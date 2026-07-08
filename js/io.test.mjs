// Tests for the pure computeMinMax helper in io.js.
// Dependency-free — run with: node js/io.test.mjs
//
// io.js also contains browser/itk-bound code (loadFileIntoArray), but the
// top-level vendor import has no side effects that touch the DOM, so importing
// it under plain Node to reach computeMinMax is safe. loadFileIntoArray itself
// needs itk-wasm/DOM and is exercised in the browser end-to-end instead.
import { computeMinMax } from './io.js';

let failures = 0;
function assert(cond, msg) {
    if (cond) { console.log(`  ok  ${msg}`); }
    else { failures++; console.error(`FAIL  ${msg}`); }
}

// ---- Test 1: raw values are reported unchanged (no 0–1 stretch) ----
{
    // A uint16-style range: the helper must return the true magnitudes so the
    // contrast slider and stats can work in real units.
    const { dataMin, dataMax } = computeMinMax(new Float32Array([0, 4000, 65535]));
    assert(dataMin === 0, `raw min is 0 (got ${dataMin})`);
    assert(dataMax === 65535, `raw max is 65535, not normalized (got ${dataMax})`);
}

// ---- Test 2: negative values ----
{
    const { dataMin, dataMax } = computeMinMax(new Float32Array([-10, 0, 10]));
    assert(dataMin === -10 && dataMax === 10, `negatives: min -10 / max 10 (got ${dataMin}/${dataMax})`);
}

// ---- Test 3: constant image ----
{
    const { dataMin, dataMax } = computeMinMax(new Float32Array([7, 7, 7, 7]));
    assert(dataMin === 7 && dataMax === 7, `constant: min === max === 7 (got ${dataMin}/${dataMax})`);
}

// ---- Test 4: empty array falls back to 0/0 (no Infinity) ----
{
    const { dataMin, dataMax } = computeMinMax(new Float32Array(0));
    assert(dataMin === 0 && dataMax === 0, `empty: 0/0 fallback, no Infinity (got ${dataMin}/${dataMax})`);
}

console.log(failures === 0 ? '\nALL TESTS PASSED' : `\n${failures} FAILURES`);
process.exit(failures === 0 ? 0 : 1);
