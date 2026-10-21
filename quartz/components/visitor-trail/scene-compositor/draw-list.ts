import type { Raster } from './raster-layers'
import { invert, multiply, none, place, scaling } from './affine-geometry'
import { ambientLayers, type Clip, type Draw, type Group, type Scope } from './render-model'

export function chunkVisible(scope: Scope, owner: Group) {
  const { size, shared } = scope
  const element = owner.element
  if (!element.hasAttribute('data-trail-chunk')) return true
  const width = size.width / shared.resolution
  const height = size.height / shared.resolution
  if (element.hasAttribute('data-ground-x')) {
    const x = Number(element.getAttribute('data-ground-x'))
    const y = Number(element.getAttribute('data-ground-y'))
    return x > -256 && x < width + 256 && y > -256 && y < height + 256
  }
  if (!element.hasAttribute('data-patch-left')) return true
  return (
    Number(element.getAttribute('data-patch-left')) < width + 8 &&
    Number(element.getAttribute('data-patch-top')) < height + 8 &&
    Number(element.getAttribute('data-patch-right')) > -8 &&
    Number(element.getAttribute('data-patch-bottom')) > -8
  )
}

// The scene's draws in paint order, each with its alpha, mask and clip.
export function collectDraws(scope: Scope, root: Group, clipFor: Raster['clipFor']) {
  const { scene, coverage, shared } = scope
  const draws: Draw[] = []
  const visit = (
    owner: Group,
    alpha: number,
    mask: Float32Array,
    maskTexture: WebGLTexture | null,
    clipped: Clip[],
    clipTexture: WebGLTexture | null,
  ) => {
    if (scene.dataset.trailAmbientReady !== 'true' && owner.element.matches(ambientLayers)) return
    if (!chunkVisible(scope, owner)) return
    alpha *= owner.opacity
    if (!owner.shown || alpha <= 0) return
    if (owner.clip) {
      clipped = [...clipped, { owner, source: owner.clip }]
      clipTexture = clipFor(clipped)
    }
    // A mask still waiting for its texture would show everything it hides.
    if (owner.mask && !owner.mask.texture) return
    if (owner.mask) {
      owner.mask.placed ??= place(
        multiply(
          scaling(1 / owner.mask.width, 1 / owner.mask.height),
          multiply(owner.mask.base, invert(owner.screen)),
        ),
      )
      maskTexture = owner.mask.texture
      mask = owner.mask.placed
    }
    for (const item of owner.items) {
      if (item.kind === 'group') {
        visit(item, alpha, mask, maskTexture, clipped, clipTexture)
        continue
      }
      if (!item.texture) continue
      if (coverage.active && item.kind === 'run' && item.palette !== shared.palette) continue
      if (item.kind === 'run')
        item.placed ??= place(
          multiply(multiply(owner.screen, invert(item.base)), scaling(item.width, item.height)),
        )
      else {
        if (!item.shown) continue
        item.placed ??= place(
          multiply(
            multiply(owner.screen, item.offset),
            scaling(item.canvas.width, item.canvas.height),
          ),
        )
      }
      draws.push({
        item,
        texture: item.texture,
        placed: item.placed,
        alpha,
        mask: maskTexture ? mask : none,
        maskTexture,
        clipTexture,
      })
    }
  }
  visit(root, 1, none, null, [], null)
  return draws
}
