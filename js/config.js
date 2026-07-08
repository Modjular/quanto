export const LABEL_COLORS = [
    '#ff595e',
    '#ffca3a',
    '#8ac926',
    '#1982c4',
];
export const RF_CONFIG = { numTrees: 8, maxDepth: 8, numClasses: LABEL_COLORS.length };
// Number of per-pixel features the backends emit from gatherFeaturesForTraining
// (8 floats packed from two RGBA float textures — see backends/webgl2.js). The
// Random Forest's feature stride must match this value.
export const NUM_FEATURES = 8;
export const MIN_LABELS_TO_TRAIN = 5;
export const CAMERA_ZOOM_MIN = 0.1;
export const CAMERA_ZOOM_MAX = 32;
export const CAMERA_ZOOM_SENSITIVITY = 0.01;
export const TRAIN_DEBOUNCE_MS = 300;