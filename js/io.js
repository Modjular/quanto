import { readImage, setPipelinesBaseUrl } from './vendor/itk-wasm-image-io.min.js';

// itk-wasm fetches its WASM pipelines relative to this URL at runtime; point it at the
// vendored copy instead of the jsDelivr CDN default.
setPipelinesBaseUrl(new URL('./vendor/itk-wasm-image-io-pipelines', import.meta.url).href);

// Max value of each itk integer component type — used as the upper bound of the
// contrast slider so it spans the dtype's real range (e.g. 0–65535 for uint16).
// Float types have no fixed max (signalled by null; the slider falls back to the
// data's own max).
const DTYPE_MAX = {
    uint8: 255, int8: 127,
    uint16: 65535, int16: 32767,
    uint32: 4294967295, int32: 2147483647,
};

/**
 * Loads an image file into a raw single-channel intensity array plus display
 * range metadata. Unlike the old path, the returned intensities are the *raw*
 * pixel values (e.g. a uint16 TIFF yields values up to 65535), not a 0–1
 * min–max stretch — features, stats, and the contrast control all operate in
 * real units. Display windowing happens on the GPU from these raw values.
 * @param {File} file - The image file to load.
 * @returns {Promise<{intensityArray: Float32Array, w: number, h: number, shape: number[], range: {dataMin: number, dataMax: number, dtypeMax: number, scale: number}}>}
 */
export async function loadFileIntoArray(file) {
  let intensityArray, w, h, shape, dtypeMax, scale;

  if (file.name.endsWith('.tif') || file.name.endsWith('.tiff')) {
    // webWorker: false — the bundled worker runs from a data: URL (opaque origin), which
    // needs CORS headers to fetch the vendored pipeline WASM even same-origin. Running on
    // the main thread avoids that requirement entirely.
    const { image } = await readImage(file, { webWorker: false })

    w = image.size[0]
    h = image.size[1]
    shape = image.size
    // Keep the raw pixel magnitudes; widen to f32 for the GPU (which represents
    // integers exactly up to 2^24, so uint16 is lossless).
    intensityArray = image.data instanceof Float32Array
        ? image.data
        : Float32Array.from(image.data);

    const componentType = image.imageType?.componentType;
    const isFloat = componentType === 'float32' || componentType === 'float64';
    dtypeMax = DTYPE_MAX[componentType] ?? null; // null → float, no fixed max
    // Stats accumulate a fixed-point integer per pixel. Integer dtypes are already
    // integers (scale 1, exact); float dtypes get a documented scale so fractional
    // intensities survive the integer accumulator. 64-bit accumulation (see
    // STATS_LAYOUT) absorbs the larger magnitudes either way.
    scale = isFloat ? 1000 : 1;
  } else {
    const img = await createImageBitmap(file);
    w = img.width;
    h = img.height;
    shape = [w, h];
    const off = new OffscreenCanvas(w, h);
    const ctx = off.getContext('2d');
    ctx.drawImage(img, 0, 0);
    const rgba = ctx.getImageData(0, 0, w, h).data;

    // Raw luma in 0–255 units (these formats decode to 8-bit), keeping display
    // and stats in the source's real range rather than a 0–1 stretch.
    intensityArray = new Float32Array(w * h);
    for (let i = 0; i < w * h; i++) {
        const r = rgba[i * 4], g = rgba[i * 4 + 1], b = rgba[i * 4 + 2];
        intensityArray[i] = 0.299 * r + 0.587 * g + 0.114 * b;
    }
    dtypeMax = 255;
    scale = 1;
  }

  const { dataMin, dataMax } = computeMinMax(intensityArray);
  // Float images have no fixed dtype max, so the slider spans the data's range.
  const range = { dataMin, dataMax, dtypeMax: dtypeMax ?? dataMax, scale };

  return { intensityArray, w, h, shape, range }
}

/**
 * Scans an intensity array for its min and max. Pure and dependency-free.
 * @param {Float32Array|Array<number>} data
 * @returns {{dataMin: number, dataMax: number}} Both 0 for an empty array.
 */
export function computeMinMax(data) {
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < data.length; i++) {
        const v = data[i];
        if (v < min) min = v;
        if (v > max) max = v;
    }
    if (!Number.isFinite(min)) { min = 0; max = 0; }
    return { dataMin: min, dataMax: max };
}
