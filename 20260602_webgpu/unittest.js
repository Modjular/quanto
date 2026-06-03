import { create, globals } from 'webgpu';
import { fastfilters3 } from './fastfilters3.js'
import { FlatRandomForest } from './rf.js';

Object.assign(globalThis, globals);
const navigator = { gpu: create([]) };

const colors = {
  reset: "\x1b[0m",
  pass: "\x1b[32m",
  fail: "\x1b[31m",
  header: "\x1b[36m\x1b[1m"
};

function log(msg, status = "") {
  if (status === "PASS") console.log(`${colors.pass}[PASS] ${msg}${colors.reset}`);
  else if (status === "FAIL") console.log(`${colors.fail}[FAIL] ${msg}${colors.reset}`);
  else if (status === "HEADER") console.log(`\n${colors.header}${msg}${colors.reset}`);
  else console.log(msg);
}

function floatFromHex(hexStr) {
  const match = hexStr.toLowerCase().match(/^(-?)0x([0-9a-f]+)\.?([0-9a-f]*)p([-+]?[0-9]+)$/);
  if (!match) return parseFloat(hexStr); 

  const sign = match[1] === '-' ? -1 : 1;
  const intPart = parseInt(match[2], 16);
  const fracPart = match[3] ? parseInt(match[3], 16) * Math.pow(16, -match[3].length) : 0;
  const expPart = parseInt(match[4], 10);

  return sign * (intPart + fracPart) * Math.pow(2, expPart);
}

function assertArrayAlmostEqual(name, actual, expected, atol = 1e-6) {
  if (actual.length !== expected.length) {
    log(`${name} - Length mismatch: Expected ${expected.length}, got ${actual.length}`, "FAIL");
    return false;
  }
  let maxDiff = 0;
  for (let i = 0; i < actual.length; i++) {
    const diff = Math.abs(actual[i] - expected[i]);
    if (diff > maxDiff) maxDiff = diff;
    if (diff > atol) {
      log(`${name} - Diff ${diff.toExponential(2)} at index ${i} exceeds atol`, "FAIL");
      return false;
    }
  }
  log(`${name} (Max diff: ${maxDiff.toExponential(2)})`, "PASS");
  return true;
}

/**
 *
 *  FastFilters3 Tests
 *
 */

const KERNELS_TEXT = `
0.3 0 0x1.fc125ap-1 0x1.f6d37ap-9
0.3 1 0x0p+0 0x1.fffff8p-2 0x1.f04ep-25
0.3 2 -0x1.b297d2p-1 0x1.dc3f88p-3 0x1.88f01ep-3
0.7 0 0x1.23c2d4p-1 0x1.a4a892p-3 0x1.3b317p-7 0x1.eaf0acp-15
0.7 1 0x0p+0 0x1.ae4b2p-2 0x1.426996p-5 0x1.78a37cp-12
0.7 2 -0x1.21997cp+0 0x1.b38f4ep-2 0x1.19b9b4p-3 0x1.6367bap-9
1.0 0 0x1.98a0a4p-2 0x1.efb0bp-3 0x1.ba69eap-5 0x1.228634p-8
1.0 1 0x0p+0 0x1.ef97dcp-3 0x1.ba53c2p-4 0x1.b3b37cp-7 0x1.18aef2p-11
1.0 2 -0x1.98c6ep-2 0x1.0caf58p-17 0x1.4bf464p-3 0x1.22b3aep-5 0x1.0857dcp-9
1.6 0 0x1.fee3dap-3 0x1.a43f28p-3 0x1.d3ce16p-4 0x1.605a9ep-5 0x1.6726acp-7 0x1.ef673ep-10
1.6 1 0x0p+0 0x1.48654ap-4 0x1.6d8f34p-4 0x1.9d0346p-5 0x1.18a744p-6 0x1.e3e836p-9 0x1.0efcb4p-11
1.6 2 -0x1.8f1af8p-4 -0x1.901acap-5 0x1.9b274ap-6 0x1.5a3c98p-5 0x1.7044eep-6 0x1.a82146p-8 0x1.273694p-10 0x1.052f06p-13
3.5 0 0x1.d35556p-4 0x1.c0a47ep-4 0x1.8cefdcp-4 0x1.43a924p-4 0x1.e67286p-5 0x1.50e606p-5 0x1.ae1212p-6 0x1.f9f938p-7 0x1.124dfep-7 0x1.121abap-8 0x1.f8ddb6p-10 0x1.ac80bep-11
3.5 1 0x0p+0 0x1.253fp-7 0x1.037324p-6 0x1.3d54d8p-6 0x1.3df4dcp-6 0x1.134234p-6 0x1.a5a90ep-7 0x1.21612p-7 0x1.66965ep-8 0x1.931dc2p-9 0x1.9c7e8ep-10 0x1.811cf2p-11 0x1.48a03ap-12 0x1.00a5a4p-13
3.5 2 -0x1.3216dap-7 -0x1.0dda76p-7 -0x1.5e21f6p-8 -0x1.c19398p-10 0x1.86a3dcp-10 0x1.cba17p-9 0x1.1137f4p-8 0x1.f1675ep-9 0x1.7bc67ap-9 0x1.f84fa8p-10 0x1.2893b8p-10 0x1.385facp-11 0x1.28f0e2p-12 0x1.00aeaep-13 0x1.985f32p-15
5.0 0 0x1.476faap-4 0x1.40f3d8p-4 0x1.2e43p-4 0x1.117f6ap-4 0x1.db88f4p-5 0x1.8d334p-5 0x1.3ec2b8p-5 0x1.eb8fdp-6 0x1.6c286ep-6 0x1.033262p-6 0x1.628266p-7 0x1.d1dbc8p-8 0x1.2616cp-8 0x1.64bf72p-9 0x1.9fc9dp-10 0x1.d19932p-11
5.0 1 0x0p+0 0x1.9b6516p-9 0x1.836fe4p-8 0x1.06ecfap-7 0x1.30c4dp-7 0x1.3e3492p-7 0x1.32706p-7 0x1.13a908p-7 0x1.d2c674p-8 0x1.75c43p-8 0x1.1c013ep-8 0x1.9a87a6p-9 0x1.1ab88ap-9 0x1.73899ep-10 0x1.d255dcp-11 0x1.17c01ep-11 0x1.410bb8p-12 0x1.609b84p-13 0x1.72cca2p-14
5.0 2 -0x1.a43f58p-9 -0x1.8b718cp-9 -0x1.45d8cp-9 -0x1.c13864p-10 -0x1.b72982p-11 0x1.7874a6p-21 0x1.687732p-11 0x1.2f1446p-10 0x1.6cd17ap-10 0x1.74d9d2p-10 0x1.558118p-10 0x1.1f3dbcp-10 0x1.c1a0fap-11 0x1.4a1abp-11 0x1.c91984p-12 0x1.2b961cp-12 0x1.74d69ep-13 0x1.b9aa5ep-14 0x1.f3609ap-15 0x1.0e78dp-15 0x1.1a7ff8p-16
10.0 0 0x1.478f58p-5 0x1.45ed1ep-5 0x1.4112e6p-5 0x1.39258p-5 0x1.2e603cp-5 0x1.211216p-5 0x1.1199ep-5 0x1.0061f6p-5 0x1.dbb6f6p-6 0x1.b4f324p-6 0x1.8d59acp-6 0x1.65be86p-6 0x1.3ee18ep-6 0x1.19695cp-6 0x1.ebbf5ep-7 0x1.a95f62p-7 0x1.6c4ba8p-7 0x1.34e246p-7 0x1.034b76p-7 0x1.af008p-8 0x1.62a4b2p-8 0x1.20e8c4p-8 0x1.d208dap-9 0x1.7422dap-9 0x1.263334p-9 0x1.cc8b18p-10 0x1.64e1f4p-10 0x1.11cd7p-10 0x1.9ff20ap-11 0x1.38cc04p-11 0x1.d1c63cp-12
10.0 1 0x0p+0 0x1.a28f98p-12 0x1.9c5452p-11 0x1.2d9c96p-10 0x1.845132p-10 0x1.d009aep-10 0x1.0785ep-9 0x1.20186p-9 0x1.31760ap-9 0x1.3ba418p-9 0x1.3eed9ep-9 0x1.3bda5p-9 0x1.332292p-9 0x1.25a204p-9 0x1.144956p-9 0x1.0010ap-9 0x1.d3d5e4p-10 0x1.a5778cp-10 0x1.769d8ap-10 0x1.48a43p-10 0x1.1ca666p-10 0x1.e6f7b8p-11 0x1.9b7662p-11 0x1.577e72p-11 0x1.1b5cf2p-11 0x1.ce0fc6p-12 0x1.7461acp-12 0x1.28ae7ap-12 0x1.d3650cp-13 0x1.6c0a66p-13 0x1.1862ccp-13 0x1.ab22fcp-14 0x1.41c66ap-14 0x1.df8366p-15 0x1.61689p-15 0x1.01a71p-15
10.0 2 -0x1.a4a1ap-12 -0x1.9e58aep-12 -0x1.8bcdaep-12 -0x1.6deaa4p-12 -0x1.4623a4p-12 -0x1.165d96p-12 -0x1.c19c04p-13 -0x1.4fad4p-13 -0x1.b77efap-14 -0x1.a9a2ep-15 0x1.d61ba8p-24 0x1.82e9aep-15 0x1.68e7d2p-14 0x1.f3495cp-14 0x1.2f6b04p-13 0x1.55b86p-13 0x1.6d3784p-13 0x1.772a34p-13 0x1.7541d8p-13 0x1.6976c2p-13 0x1.55e15cp-13 0x1.3c9796p-13 0x1.1f908cp-13 0x1.008f1ap-13 0x1.c227bp-14 0x1.84abd8p-14 0x1.4a83c8p-14 0x1.14facap-14 0x1.c9b966p-15 0x1.7511e6p-15 0x1.2c0efap-15 0x1.dc783ep-16 0x1.759044p-16 0x1.215032p-16 0x1.bad25cp-17 0x1.4f024ap-17 0x1.f55176p-18 0x1.7330a8p-18 0x1.1030f8p-18 0x1.8bc5p-19 0x1.1db04p-19
`;

const EXPECTED_KERNELS = {};
KERNELS_TEXT.trim().split('\n').forEach(line => {
  const items = line.trim().split(' ');
  if (items.length > 2) {
    const scale = parseFloat(items[0]);
    const order = parseInt(items[1], 10);
    const weights = new Float32Array(items.slice(2).map(floatFromHex));
    EXPECTED_KERNELS[`${scale}_${order}`] = weights;
  }
});

async function runFiltersTest() {
  log("Running Kernel Generation Tests (CPU)", "HEADER");
  for (const [key, expected] of Object.entries(EXPECTED_KERNELS)) {
    const [scaleStr, orderStr] = key.split('_');
    const scale = parseFloat(scaleStr);
    const order = parseInt(orderStr, 10);
    const actual = fastfilters3.gaussian_kernel(scale, order);
    assertArrayAlmostEqual(`Scale ${scale}, Order ${order}`, actual, expected, 1e-4);
  }

  log("Running compute_filters WebGPU Tests", "HEADER");
  const gpu = (typeof navigator !== "undefined" && navigator.gpu);
  if (!gpu) {
    log("WebGPU is not supported or unavailable in this environment. Skipping GPU tests.", "HEADER");
    return;
  }

  try {
    const adapter = await gpu.requestAdapter();
    const device = await adapter.requestDevice();
    const width = 128, height = 128;
    const mockImage = new Float32Array(width * height).fill(1.0);
    const gpuResult = await fastfilters3.compute_filters(device, mockImage, width, height, 1.0);
    
    if (gpuResult.length === 7 * width * height) {
      log(`compute_filters output shape matched (Channels: 7, Size: ${gpuResult.length})`, "PASS");
      
      // Basic value validation for flat field
      const idx = (height / 2 * width + (width / 2)) * 7;
      const smoothing = gpuResult[idx];
      const laplacian = gpuResult[idx + 1];
      const gradient = gpuResult[idx + 2];
      
      let valuesOk = true;
      if (Math.abs(smoothing - 1.0) > 1e-5) {
        log(`Smoothing failed: Expected ~1.0, got ${smoothing}`, "FAIL");
        valuesOk = false;
      }
      if (Math.abs(laplacian) > 1e-5) {
        log(`Laplacian failed: Expected ~0.0, got ${laplacian}`, "FAIL");
        valuesOk = false;
      }
      if (Math.abs(gradient) > 1e-5) {
        log(`Gradient failed: Expected ~0.0, got ${gradient}`, "FAIL");
        valuesOk = false;
      }
      
      if (valuesOk) {
        log("compute_filters values for flat field verified.", "PASS");
      }
    } else {
      log(`compute_filters shape mismatch`, "FAIL");
    }
  } catch (e) {
    log("WebGPU Test Failed: " + e.message, "FAIL");
  }

  log("All tests completed.", "HEADER");
}

runFiltersTest().catch(e => {
    console.error(e);
    process.exit(1);
});

/**
 * 
 *  FlatRandomForest Tests
 * 
 */

function runTest(testName, testFn) {
    try {
        testFn();
        console.log(`\x1b[32m[PASS]\x1b[0m ${testName}`);
    } catch (err) {
        console.error(`\x1b[31m[FAIL]\x1b[0m ${testName}`);
        console.error(err.stack);
    }
}

// console.log("Starting Flat Random Forest verification validations...\n");
log("Starting Flat Random Forest verification validations...\n", "HEADER");

runTest("Bitwise Integrity Verification (Anti-NaN Mangle Test)", () => {
    // Force a setup that guarantees immediate leaf generation to inspect raw sign bit integrity
    const X = new Float32Array([1.0]);
    const y = new Int32Array([1]); // Class 1
    
    const rf = new FlatRandomForest({ numTrees: 1, maxDepth: 1, numClasses: 2 });
    rf.train(X, y, 1);
    
    const i32View = new Int32Array(rf.forestBuffer.buffer, rf.forestBuffer.byteOffset, rf.forestBuffer.length);
    
    if (i32View[0] !== -1) {
        throw new Error(`FeatIdx Bit Mangle: Expected -1, got: ${i32View[0]}`);
    }
    if (i32View[3] !== -2) {
        throw new Error(`LeafClass Encoding Bit Mangle: Expected -2 (for Class 1), got: ${i32View[3]}`);
    }
});

runTest("Simple classification inference match", () => {
    const X = new Float32Array([
        0.1, 0.2,
        0.15, 0.1,
        0.8, 0.9,
        0.95, 0.85
    ]);
    const y = new Int32Array([0, 0, 1, 1]);

    const rf = new FlatRandomForest({ numTrees: 5, maxDepth: 3, numClasses: 2 });
    rf.train(X, y, 2);

    const predLow = rf.predictSingle(new Float32Array([0.12, 0.15]));
    const predHigh = rf.predictSingle(new Float32Array([0.9, 0.9]));

    if (predLow !== 0) throw new Error(`Expected Class 0, got: ${predLow}`);
    if (predHigh !== 1) throw new Error(`Expected Class 1, got: ${predHigh}`);
});

runTest("Multiclass classification (3 classes, non-binary split)", () => {
    // 6 samples, 2 features, 3 distinct classes (0, 1, 2)
    const X = new Float32Array([
        0.1, 0.1,  // Class 0 (Bottom-left)
        0.2, 0.1,  // Class 0
        0.5, 0.5,  // Class 1 (Center)
        0.6, 0.5,  // Class 1
        0.9, 0.9,  // Class 2 (Top-right)
        0.8, 0.9   // Class 2
    ]);
    const y = new Int32Array([0, 0, 1, 1, 2, 2]);

    const rf = new FlatRandomForest({ numTrees: 31, maxDepth: 4, numClasses: 3 });
    rf.train(X, y, 2);

    const pred0 = rf.predictSingle(new Float32Array([0.15, 0.12]));
    const pred1 = rf.predictSingle(new Float32Array([0.55, 0.52]));
    const pred2 = rf.predictSingle(new Float32Array([0.85, 0.88]));

    if (pred0 !== 0) throw new Error(`Expected Class 0, got: ${pred0}`);
    if (pred1 !== 1) throw new Error(`Expected Class 1, got: ${pred1}`);
    if (pred2 !== 2) throw new Error(`Expected Class 2, got: ${pred2}`);
});

runTest("Graceful termination on identical features with conflicting labels", () => {
    // 3 samples with IDENTICAL features, but conflicting classes
    const X = new Float32Array([
        0.5, 0.5, 
        0.5, 0.5, 
        0.5, 0.5
    ]);
    const y = new Int32Array([0, 1, 1]); // Class 1 is the clear majority (2 vs 1)

    const rf = new FlatRandomForest({ numTrees: 31, maxDepth: 5, numClasses: 2 });
    
    // This must complete without an infinite loop / call stack overflow
    rf.train(X, y, 2);

    // Because features are identical, it must fall back to the majority class (1)
    const pred = rf.predictSingle(new Float32Array([0.5, 0.5]));
    if (pred !== 1) {
        throw new Error(`Expected majority fallback Class 1, instead predicted: ${pred}`);
    }
});

runTest("Strict enforcement of maxDepth constraint", () => {
    // Create a perfectly interleaved dataset that *wants* to split endlessly
    const X = new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
    const y = new Int32Array([0, 1, 0, 1, 0, 1, 0, 1]);
    
    // Force a strict shallow depth limit of 2
    const targetMaxDepth = 2;
    const rf = new FlatRandomForest({ numTrees: 1, maxDepth: targetMaxDepth, numClasses: 2 });
    rf.train(X, y, 1);

    const i32View = new Int32Array(rf.forestBuffer.buffer, rf.forestBuffer.byteOffset, rf.forestBuffer.length);

    // Trace down the left-most paths manually to confirm zero pointers exist beyond depth 2
    let currentDepth = 0;
    let nodeIdx = 0;

    while (currentDepth <= targetMaxDepth + 1) {
        const slotOffset = nodeIdx * 4;
        const featIdx = i32View[slotOffset + 0];

        if (featIdx === -1) {
            // Reached a leaf safely! Verify it happened at or before maxDepth
            if (currentDepth > targetMaxDepth) {
                throw new Error(`Leaf node was generated at depth ${currentDepth}, which exceeds maxDepth limit of ${targetMaxDepth}`);
            }
            break;
        }

        // Move to the left child for the next tier inspection
        nodeIdx = i32View[slotOffset + 2];
        currentDepth++;
        
        if (currentDepth > targetMaxDepth) {
            throw new Error(`Tree failed to truncate! Traversal reached depth ${currentDepth} without hitting a leaf marker.`);
        }
    }
});

runTest("Single-class dataset (Root-level purity handling)", () => {
    // User has only painted Class 1 annotations so far
    const X = new Float32Array([0.1, 0.2, 0.3, 0.4]);
    const y = new Int32Array([1, 1, 1, 1]);
    
    const rf = new FlatRandomForest({ numTrees: 2, maxDepth: 5, numClasses: 2 });
    
    // Should train instantly without trying to calculate splits
    rf.train(X, y, 1);
    
    // Every prediction must safely return the only known class
    const pred = rf.predictSingle(new Float32Array([0.9]));
    if (pred !== 1) throw new Error(`Expected automatic Class 1 prediction, got: ${pred}`);
});

runTest("Deterministic tie-breaking on even tree counts", () => {
    const rf = new FlatRandomForest({ numClasses: 3 });
    
    // Fake a forest buffer where Tree 0 votes Class 1 and Tree 1 votes Class 2
    // Struct layout: [featIdx, threshold, left, right]
    // A leaf node uses: i32[0] = -1, i32[3] = -(classId + 1)
    const buffer = new ArrayBuffer(2 * 4 * 4); // 2 nodes, 4 slots each, 4 bytes per slot
    const i32 = new Int32Array(buffer);
    const f32 = new Float32Array(buffer);
    
    // Tree 0 (Root at index 0): Direct Leaf voting for Class 1
    i32[0] = -1; f32[1] = 0.0; i32[2] = -1; i32[3] = -2; // -2 encodes Class 1
    
    // Tree 1 (Root at index 1): Direct Leaf voting for Class 2
    i32[4] = -1; f32[5] = 0.0; i32[6] = -1; i32[7] = -3; // -3 encodes Class 2
    
    rf.forestBuffer = f32;
    rf.treeRoots = new Int32Array([0, 1]); // Map roots
    rf.numTrees = 2;

    // Class 1 and Class 2 both get exactly 1 vote. 
    // Our implementation uses `>` which naturally favors the lowest-index class in a tie.
    const pred = rf.predictSingle(new Float32Array([0.0]));
    if (pred !== 1) {
        throw new Error(`Tie-breaker failed. Expected deterministic lower class index (1), got: ${pred}`);
    }
});

runTest("Empty training dataset guard verification", () => {
    const rf = new FlatRandomForest();
    let didThrow = false;
    
    try {
        rf.train(new Float32Array([]), new Int32Array([]), 2);
    } catch (e) {
        if (e.message.includes("No labeled samples provided")) {
            didThrow = true;
        }
    }
    
    if (!didThrow) {
        throw new Error("Engine failed to gracefully reject an empty training array payload.");
    }
});
