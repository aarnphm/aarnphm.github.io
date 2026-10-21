// Pixel landscape for the 404 ruin, painted the way Van Gogh painted the factories at Clichy:
// low-resolution canvases upscaled with `image-rendering: pixelated`, an ordered-dither
// underpainting, then short brush strokes laid along flow fields. Each stroke picks its ramp stop
// once, so the dither happens per stroke instead of per pixel. The ruin itself is printed into the
// same grid from the hidden SVG geometry: plates, one-pixel keylines, dial marks and chips, with
// the moss grown over it as blobs (404-overgrowth). The cat, the survey party and a passing
// triathlete (404-cat, 404-crew, 404-triathlete) stand in the same canvas as the water, which
// reflects them along with the ruin, the hill and the ranges. The ranges and the ruin paint once per size or change of light; sky, field, water,
// camp, canopy and portal repaint in stop-motion ticks.

export type { Landscape } from './404-landscape/scope'
export { setupLandscape } from './404-landscape/setup'
