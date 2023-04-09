---
id: "rich-huang-deep-dive"
tags: []
---

Previously worked at Scale AI

Working on WebGL for image segmentation

SemSeg -> label pixelmap

Nuro -> wanted to use high res images 600M pixels

Data Infrastructure layers

problems:

- tasker machine unable to label
  - Semantic segmentation mask
    - label info is stored in pixle mask on top of image
    - Pixel are (r,g,b,a) -> encode label in r channel
    - other pixel masks layered on top of coloring effect
  - browser constraint
  - hardware constraint

- web workers (multithread js)

- splitting up image
  - poor labelling
  - tasker variance

- fixing viewport
  - spliting image into tiles
  - slow

WebGL: POC -> segment annotator -> user testing

UI -> pixi app -> standard middleware -> container (tile sprite...)

tile sprite: - Fill shader - Brush - Polygon

GLSL shader

Problems:

- Memory usage are still very high
- maintainability

Doing a lot of testing and prototyping -> so play with the tech and see what
works (approach)
