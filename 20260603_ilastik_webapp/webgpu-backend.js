import { fastfilters3 as ff } from './ff.js';
import * as shaders from './shaders.js';

export class WebGpuBackend {
    constructor() {
        this.device = null;
        this.context = null;
        this.format = null;
        this.width = 0;
        this.height = 0;
        this.originalTexture = null;
        this.featureBuffer = null;
        this.probBuffer = null;
    }

    async initialize(canvas) {
        if (!navigator.gpu) throw new Error("WebGPU not supported");
        const adapter = await navigator.gpu.requestAdapter();
        
        const requiredLimits = {};
        if (adapter.limits.maxBufferSize) requiredLimits.maxBufferSize = adapter.limits.maxBufferSize;
        if (adapter.limits.maxStorageBufferBindingSize) requiredLimits.maxStorageBufferBindingSize = adapter.limits.maxStorageBufferBindingSize;
        
        this.device = await adapter.requestDevice({ requiredLimits });
        this.context = canvas.getContext('webgpu');
        this.format = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({ device: this.device, format: this.format });
    }

    async allocateImage(width, height, rgbaData) {
        this.width = width;
        this.height = height;

        if (this.originalTexture) this.originalTexture.destroy();
        this.originalTexture = this.device.createTexture({
            size: [width, height],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
        });
        this.device.queue.writeTexture({ texture: this.originalTexture }, rgbaData, { bytesPerRow: width * 4 }, [width, height]);

        if (this.probBuffer) this.probBuffer.destroy();
        this.probBuffer = this.device.createBuffer({
            size: width * height * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        });

        const initialProbs = new Float32Array(width * height).fill(-1.0);
        this.device.queue.writeBuffer(this.probBuffer, 0, initialProbs);
    }

    async updateFeatures(intensityArray, sigma) {
        if (this.featureBuffer) this.featureBuffer.destroy();
        this.featureBuffer = await ff.extract_features(this.device, intensityArray, this.width, this.height, sigma);
        this.renderComposite();
    }

    async gatherFeaturesForTraining(indicesArray) {
        const numLabels = indicesArray.length;
        const indicesBuffer = this.device.createBuffer({
            size: indicesArray.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(indicesBuffer, 0, indicesArray);

        const gatherDstSize = numLabels * 8 * 4;
        const gatherDstBuffer = this.device.createBuffer({
            size: gatherDstSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });

        const stagingBuffer = this.device.createBuffer({
            size: gatherDstSize,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });

        const module = this.device.createShaderModule({ code: shaders.GATHER_FEATURES_SHADER });
        const pipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: { module, entryPoint: 'main' }
        });

        const bindGroup = this.device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.featureBuffer } },
                { binding: 1, resource: { buffer: indicesBuffer } },
                { binding: 2, resource: { buffer: gatherDstBuffer } }
            ]
        });

        const enc = this.device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(numLabels / 64));
        pass.end();

        enc.copyBufferToBuffer(gatherDstBuffer, 0, stagingBuffer, 0, gatherDstSize);
        this.device.queue.submit([enc.finish()]);

        await stagingBuffer.mapAsync(GPUMapMode.READ);
        const features = new Float32Array(stagingBuffer.getMappedRange().slice());
        stagingBuffer.unmap();

        indicesBuffer.destroy();
        gatherDstBuffer.destroy();
        stagingBuffer.destroy();

        return features;
    }

    async runInference(rf) {
        const forestBuffer = this.device.createBuffer({
            size: rf.forestBuffer.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(forestBuffer, 0, rf.forestBuffer);

        const rootsBuffer = this.device.createBuffer({
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        const paddedRoots = new Int32Array(8);
        paddedRoots.set(rf.treeRoots);
        this.device.queue.writeBuffer(rootsBuffer, 0, paddedRoots);

        if (this.probBuffer) this.probBuffer.destroy();
        this.probBuffer = this.device.createBuffer({
            size: this.width * this.height * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });

        const code = shaders.RF_INFERENCE_SHADER
            .replace(/\${WIDTH}/g, this.width)
            .replace(/\${HEIGHT}/g, this.height);

        const module = this.device.createShaderModule({ code });
        const pipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: { module, entryPoint: "main" }
        });

        const bindGroup = this.device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.featureBuffer } },
                { binding: 1, resource: { buffer: forestBuffer } },
                { binding: 2, resource: { buffer: rootsBuffer } },
                { binding: 3, resource: { buffer: this.probBuffer } }
            ]
        });

        const enc = this.device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(this.width / 16), Math.ceil(this.height / 16));
        pass.end();
        this.device.queue.submit([enc.finish()]);

        this.renderComposite();
    }

    renderComposite() {
        if (!this.originalTexture || !this.probBuffer) return;

        const code = shaders.COMPOSITE_SHADER
            .replace(/\${WIDTH}/g, this.width)
            .replace(/\${HEIGHT}/g, this.height);

        const module = this.device.createShaderModule({ code });
        const pipeline = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: { module, entryPoint: 'vs_main' },
            fragment: { module, entryPoint: 'fs_main', targets: [{ format: this.format }] },
            primitive: { topology: 'triangle-strip' }
        });

        const bindGroup = this.device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: this.device.createSampler() },
                { binding: 1, resource: this.originalTexture.createView() },
                { binding: 2, resource: { buffer: this.probBuffer } }
            ]
        });

        const enc = this.device.createCommandEncoder();
        const pass = enc.beginRenderPass({
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                clearValue: { r: 0, g: 0, b: 0, a: 1 },
                loadOp: 'clear', storeOp: 'store'
            }]
        });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.draw(4);
        pass.end();
        this.device.queue.submit([enc.finish()]);
    }
}
