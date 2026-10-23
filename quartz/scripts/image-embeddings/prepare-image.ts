import sharp from 'sharp'

export function prepareImage(source: Uint8Array) {
  // Buffer input gives SVGs no file base for resolving external resources.
  return sharp(source, { animated: false, limitInputPixels: 268402689 })
    .rotate()
    .flatten({ background: '#ffffff' })
    .toColourspace('srgb')
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true })
}
