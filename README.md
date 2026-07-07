# Quanto
Count cells in your browser using a pixel-classifier. Inspired by Ilastik.

### Features
 - Live preview of your labels thanks to GPU-acceleration.
 - Segmentations, probabilities, labels, and points exporting.
 - Figma-style canvas navigation.

### TODO:
 [ ] 3D support
 [ ] Contrast sliders
 [ ] Export an Ilastik-compatible `.ilp`

### Why
This started out as an exploration into how performant WebGPU could be for a more mid-level task like pixel-classification. Filter-computation and random-forest computation is all done on the GPU. The component-connecting for labeling is also done on the GPU.


