importScripts('https://cdn.jsdelivr.net/npm/h5wasm@0.7.6/dist/iife/h5wasm.js');

let isValid = true;

function logSuccess(msg) { self.postMessage({ type: 'log', level: 'success', msg: msg }); }
function logInfo(msg) { self.postMessage({ type: 'log', level: 'info', msg: msg }); }
function logWarn(msg) { self.postMessage({ type: 'log', level: 'warn', msg: msg }); }
function logError(msg) { 
    isValid = false; 
    self.postMessage({ type: 'log', level: 'error', msg: `ERROR: ${msg}` }); 
}

function parseH5String(val) {
    if (!val) return null;
    if (typeof val === 'string') return val;
    if (Array.isArray(val)) return val[0];
    return new TextDecoder().decode(val);
}

function parseH5StringArray(val) {
    if (!val) return [];
    if (Array.isArray(val)) return val;
    return [parseH5String(val)]; // Fallback
}

/**
 * Parses Ilastik blockSlice string "[0:1, 10:20, 30:40, 0:1]" 
 * into h5wasm-compatible slice: [[0,1], [10,20], [30,40], [0,1]]
 */
function parseBlockSlice(bs) {
    try {
        return bs.replace(/[\[\]]/g, '').split(',').map(part => {
            return part.split(':').map(Number);
        });
    } catch (e) {
        return null;
    }
}

self.onmessage = async function(e) {
    const file = e.data.file;
    isValid = true;
    
    try {
        logInfo(`Booting WebAssembly environment...`);
        const { FS } = await h5wasm.ready;

        logInfo(`Mounting ${file.name} to virtual WORKERFS...`);
        try { FS.mkdir('/work'); } catch (err) {} 
        FS.mount(FS.filesystems.WORKERFS, { files: [file] }, '/work');

        const filePath = '/work/' + file.name;
        const f = new h5wasm.File(filePath, 'r');
        logInfo(`Starting verification of: ${file.name}`);

        const keys = f.keys();

        // --- Root Level ---
        logInfo(`\n[ Root Level ]`);
        if (keys.includes('ilastikVersion')) logSuccess(`ilastikVersion: ${parseH5String(f.get('ilastikVersion').value)}`);
        else logError(`Missing 'ilastikVersion'`);

        if (keys.includes('workflowName')) {
            const wf = parseH5String(f.get('workflowName').value);
            if (wf === "Pixel Classification") logSuccess(`workflowName: ${wf}`);
            else logError(`Incorrect workflowName: Expected 'Pixel Classification', got '${wf}'`);
        } else logError(`Missing 'workflowName'`);

        if (keys.includes('time')) logSuccess(`time: ${parseH5String(f.get('time').value)}`);
        else logWarn(`Missing 'time' dataset`);

        // --- Input Data ---
        logInfo(`\n[ Input Data ]`);
        const rawDataRegistry = {}; // Map lane name to internal dataset if found

        if (keys.includes('Input Data')) {
            const idGroup = f.get('Input Data');
            const idKeys = idGroup.keys();
            
            if (idKeys.includes('StorageVersion')) logSuccess(`StorageVersion: ${parseH5String(idGroup.get('StorageVersion').value)}`);
            else logError(`Input Data/StorageVersion missing`);

            if (idKeys.includes('infos')) {
                const infos = idGroup.get('infos');
                const lanes = infos.keys();
                logSuccess(`Found ${lanes.length} lanes in infos`);
                
                for (let laneName of lanes) {
                    const lane = infos.get(laneName);
                    if (lane.keys().includes('Raw Data')) {
                        const raw = lane.get('Raw Data');
                        logSuccess(`  ${laneName}/Raw Data found`);
                        const rawKeys = raw.keys();
                        
                        let isInternal = false;
                        let internalPath = null;

                        ['filePath', 'nickname', 'axistags', 'shape', 'inner_path'].forEach(field => {
                            if (rawKeys.includes(field)) {
                                if (field === 'axistags') {
                                    try {
                                        const tags = JSON.parse(parseH5String(raw.get(field).value));
                                        if (tags.axes) logSuccess(`    ${field}: Valid JSON`);
                                        else logError(`    ${field}: Invalid structure (missing 'axes')`);
                                    } catch (e) { logError(`    ${field}: Failed to parse JSON`); }
                                } else if (field === 'filePath') {
                                    const path = parseH5String(raw.get(field).value);
                                    if (path === "" || path.includes(file.name)) {
                                        isInternal = true;
                                    }
                                } else if (field === 'inner_path') {
                                    internalPath = parseH5String(raw.get(field).value);
                                    isInternal = true;
                                }
                                logSuccess(`    ${field}: Present`);
                            } else if (field !== 'inner_path') {
                                logError(`    ${field}: MISSING in Raw Data`);
                            }
                        });

                        // Check for actual pixels
                        if (isInternal) {
                            let dataObj = null;
                            if (internalPath) {
                                try {
                                    dataObj = f.get(internalPath);
                                    logSuccess(`    Internal raw data found at: ${internalPath}`);
                                } catch (e) {
                                    logWarn(`    Could not follow inner_path: ${internalPath}`);
                                }
                            }

                            // Fallback to searching locally in the lane group if no inner_path or it failed
                            if (!dataObj) {
                                if (rawKeys.includes('data')) dataObj = raw.get('data');
                                else if (rawKeys.includes('volume')) dataObj = raw.get('volume');
                                
                                if (dataObj) logSuccess(`    Internal raw data found in local dataset`);
                            }

                            if (dataObj) {
                                rawDataRegistry[laneName] = dataObj;
                            } else {
                                logWarn(`    Lane ${laneName} marked as internal but no pixel data found.`);
                            }
                        } else {
                            logWarn(`    Lane ${laneName} references external file. Raw thumbnails will be unavailable.`);
                        }

                    } else logError(`  ${laneName} missing 'Raw Data' group`);
                }
            } else logError(`Input Data/infos missing`);
        } else logError(`Missing 'Input Data' group`);

        // --- Feature Selections ---
        logInfo(`\n[ Feature Selections ]`);
        if (keys.includes('FeatureSelections')) {
            const fs = f.get('FeatureSelections');
            const fsKeys = fs.keys();
            ['FeatureIds', 'Scales', 'SelectionMatrix'].forEach(field => {
                if (fsKeys.includes(field)) logSuccess(`${field}: Present`);
                else logError(`FeatureSelections/${field} missing`);
            });
            
            if (fsKeys.includes('SelectionMatrix') && fsKeys.includes('FeatureIds') && fsKeys.includes('Scales')) {
                const matrixShape = fs.get('SelectionMatrix').shape;
                const expectedShape = [fs.get('FeatureIds').shape[0], fs.get('Scales').shape[0]];
                if (matrixShape[0] === expectedShape[0] && matrixShape[1] === expectedShape[1]) logSuccess(`SelectionMatrix shape matches: ${matrixShape}`);
                else logError(`SelectionMatrix shape mismatch: Expected ${expectedShape}, got ${matrixShape}`);
            }
        } else logError(`Missing 'FeatureSelections' group`);

        // --- Pixel Classification ---
        logInfo(`\n[ Pixel Classification ]`);
        if (keys.includes('PixelClassification')) {
            const pc = f.get('PixelClassification');
            const pcKeys = pc.keys();
            
            ['LabelNames', 'LabelColors', 'LabelSets'].forEach(field => {
                if (pcKeys.includes(field)) logSuccess(`${field}: Present`);
                else logError(`PixelClassification/${field} missing`);
            });

            // Extract Labels for UI
            if (pcKeys.includes('LabelNames')) {
                const labelNames = parseH5StringArray(pc.get('LabelNames').value);
                self.postMessage({ type: 'labels', names: labelNames });
            }
            
            if (pcKeys.includes('LabelSets')) {
                const ls = pc.get('LabelSets');
                const lsNames = ls.keys();
                logSuccess(`Found ${lsNames.length} label sets`);
                
                for (let labelSetName of lsNames) {
                    const lset = ls.get(labelSetName);
                    logSuccess(`  ${labelSetName} found`);

                    // Match label set to lane (usually lane000 -> labels0000)
                    const laneIndex = labelSetName.replace(/\D/g, '').padStart(4, '0');
                    const laneName = `lane${laneIndex}`;
                    const rawDataset = rawDataRegistry[laneName];

                    for (let blockName of lset.keys()) {
                        const block = lset.get(blockName);
                        logSuccess(`    ${blockName} found`);
                        const attrs = block.attrs;
                        let order = ['Z' ,'X', 'Y']
                        let sliceInfo = null;
                        
                        if (attrs && attrs.axistags) {
                            try {
                                const axistags = JSON.parse(parseH5String(attrs.axistags.value));
                                order = axistags['axes'].map((t) => t.key);
                                logSuccess(`      axistags attr: Valid JSON ${order}`);
                            } catch(e) { logError(`      axistags attr: Invalid JSON ${e}`); }
                        } else logError(`      axistags attr: MISSING`);
                        
                        if (attrs && attrs.blockSlice) {
                            const bs = parseH5String(attrs.blockSlice.value);
                            if (bs.startsWith("[") && bs.endsWith("]") && bs.includes(":")) {
                                logSuccess(`      blockSlice attr: ${bs}`);
                                sliceInfo = parseBlockSlice(bs);
                            } else logError(`      blockSlice attr: Invalid format '${bs}'`);
                        } else logError(`      blockSlice attr: MISSING`);

                        // Extract the chunk to the main thread for thumbnail rendering
                        let rawSlice = null;
                        if (rawDataset && sliceInfo) {
                            try {
                                rawSlice = rawDataset.slice(sliceInfo);
                                logSuccess(`      Successfully extracted matching raw data slice`);
                            } catch (e) {
                                logWarn(`      Failed to slice raw data: ${e.message}`);
                            }
                        }

                        self.postMessage({
                            type: 'thumbnail',
                            name: `${labelSetName} / ${blockName}`,
                            labelData: block.value, // Copies typed array to main thread
                            rawData: rawSlice,      // Optional raw background
                            shape: block.shape,
                            order: order,
                        });
                    }
                }
            }

            if (pcKeys.includes('ClassifierFactory')) logSuccess(`ClassifierFactory: Present`);
            else logWarn(`ClassifierFactory: Missing (Ilastik will use defaults)`);
            
            if (pcKeys.includes('ClassifierForests')) logSuccess(`ClassifierForests: Present (Trained)`);
            else logWarn(`ClassifierForests: Missing (Ilastik will require retraining)`);

        } else logError(`Missing 'PixelClassification' group`);

        // Finalize
        f.close();
        FS.unmount('/work');
        
        logInfo(`\n========================================`);
        if (isValid) logSuccess(`[***] VERIFICATION SUCCESSFUL: Project meets the spec.`);
        else logError(`[!!!] VERIFICATION FAILED: Project has errors.`);
        logInfo(`========================================`);

        self.postMessage({ type: 'status', isValid: isValid });

    } catch (err) {
        logError(`Unexpected parsing error: ${err.message}`);
        console.error(err);
        self.postMessage({ type: 'status', isValid: false });
    } finally {
        self.postMessage({ type: 'done' });
    }
};