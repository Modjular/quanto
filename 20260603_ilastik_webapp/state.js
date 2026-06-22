import { FlatRandomForest } from './rf.js';

export const LABEL_COLORS = [
    'rgba(255,0,255,1.0)',
    'rgba(0,255,0,1.0)',
    'rgba(0,0,255,1.0)',
];
export const RF_CONFIG = { numTrees: 8, maxDepth: 8, numClasses: LABEL_COLORS.length };
export const MIN_LABELS_TO_TRAIN = 5;
export const CAMERA_ZOOM_MIN = 0.1;
export const CAMERA_ZOOM_MAX = 10;
export const CAMERA_ZOOM_SENSITIVITY = 0.01;
export const TRAIN_DEBOUNCE_MS = 300;

export const state = {
    // Images — ordered array; visual order == export order
    images: [],

    // ML
    rf: new FlatRandomForest(RF_CONFIG),

    // Drawing / Tools
    currentClass: 0,
    isDrawing: false,
    activeImageId: null,
    toolMode: 'grab', // 'grab', 'paint', 'erase'

    // Features
    sigma: 1.0,

    // Live update
    liveUpdate: false,

    // Camera
    camera: { x: 0, y: 0, scale: 1 },
    isSpaceDown: false,
    isPanning: false,

    // Export
    outputDirHandle: null,
};
