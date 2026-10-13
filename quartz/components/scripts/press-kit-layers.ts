export interface LayerBounds {
  x: number
  y: number
  width: number
  height: number
}

export interface CardLayer extends LayerBounds {
  id: string
  label: string
  image: CanvasImageSource
  renderer?: LayerRenderer
}

export interface LayerRenderer extends LayerBounds {
  minWidth: number
  minHeight: number
  render: (width: number, height: number) => Promise<CanvasImageSource>
}

export interface CardScene {
  width: number
  height: number
  scale: number
  background: string | null
  layers: CardLayer[]
}

export function captureLayer<T>(
  ctx: CanvasRenderingContext2D,
  layers: CardLayer[] | undefined,
  id: string,
  label: string,
  bounds: LayerBounds,
  paint: (context: CanvasRenderingContext2D) => T,
): T {
  if (!layers || bounds.width <= 0 || bounds.height <= 0) return paint(ctx)
  const scale = ctx.getTransform().a
  const x = Math.floor((bounds.x - 3) * scale) / scale
  const y = Math.floor((bounds.y - 3) * scale) / scale
  const width = Math.ceil((bounds.x + bounds.width + 3 - x) * scale) / scale
  const height = Math.ceil((bounds.y + bounds.height + 3 - y) * scale) / scale
  const canvas = document.createElement('canvas')
  canvas.width = Math.ceil(width * scale)
  canvas.height = Math.ceil(height * scale)
  const context = canvas.getContext('2d')
  if (!context) throw new Error('Canvas is unavailable')
  context.setTransform(scale, 0, 0, scale, -x * scale, -y * scale)
  const result = paint(context)
  ctx.drawImage(canvas, x, y, width, height)
  layers.push({ id, label, x, y, width, height, image: canvas })
  return result
}

/** Stable identities belong to source blocks and fields, even when another field is hidden. */
export function svgLayer(parent: SVGElement, key: string, label: string): SVGElement {
  const layer = document.createElementNS('http://www.w3.org/2000/svg', 'g')
  const block = parent.closest('[data-block]')?.getAttribute('data-block') ?? 'card'
  layer.setAttribute('data-card-layer', `${block}/${key}`)
  layer.setAttribute('data-layer-label', label)
  parent.append(layer)
  return layer
}

export async function svgLayers(
  source: string,
  width: number,
  height: number,
  scale: number,
  decode: (source: string) => Promise<HTMLImageElement>,
  signal?: AbortSignal,
  renderers?: ReadonlyMap<string, LayerRenderer>,
): Promise<CardLayer[]> {
  const parsed = new DOMParser().parseFromString(source, 'image/svg+xml').documentElement
  if (!(parsed instanceof SVGSVGElement)) throw new Error('Invalid card SVG')
  const host = document.createElement('div')
  host.style.cssText = 'position:fixed;left:-20000px;top:0;visibility:hidden;pointer-events:none'
  parsed.setAttribute('width', String(width))
  parsed.setAttribute('height', String(height))
  host.append(parsed)
  document.body.append(host)
  try {
    const root = parsed.getBoundingClientRect()
    const layers: CardLayer[] = []
    for (const element of parsed.querySelectorAll<SVGGraphicsElement>('[data-card-layer]')) {
      signal?.throwIfAborted()
      const id = element.getAttribute('data-card-layer')!
      const renderer = renderers?.get(id)
      let rect = element.getBoundingClientRect()
      const matrix = element.getScreenCTM()
      if (renderer && matrix) {
        const start = new DOMPoint(renderer.x, renderer.y).matrixTransform(matrix)
        const end = new DOMPoint(
          renderer.x + renderer.width,
          renderer.y + renderer.height,
        ).matrixTransform(matrix)
        rect = new DOMRect(start.x, start.y, end.x - start.x, end.y - start.y)
      }
      if (rect.width <= 0 && rect.height <= 0) continue
      const x = Math.floor((rect.x - root.x - 3) * scale) / scale
      const y = Math.floor((rect.y - root.y - 3) * scale) / scale
      const w = Math.ceil((rect.x - root.x + rect.width + 3 - x) * scale) / scale
      const h = Math.ceil((rect.y - root.y + rect.height + 3 - y) * scale) / scale
      const copy = parsed.cloneNode(true)
      if (!(copy instanceof SVGSVGElement)) continue
      const keep = (parent: Element) => {
        for (const child of Array.from(parent.children)) {
          if (child.matches('defs, clipPath, title')) continue
          if (child.getAttribute('data-card-layer') === id) continue
          if (
            Array.from(child.querySelectorAll('[data-card-layer]')).some(
              item => item.getAttribute('data-card-layer') === id,
            )
          )
            keep(child)
          else child.remove()
        }
      }
      keep(copy)
      copy.setAttribute('viewBox', `${x} ${y} ${w} ${h}`)
      copy.setAttribute('width', String(Math.ceil(w * scale)))
      copy.setAttribute('height', String(Math.ceil(h * scale)))
      layers.push({
        id,
        label: element.getAttribute('data-layer-label') ?? id,
        x,
        y,
        width: w,
        height: h,
        image: await decode(copy.outerHTML),
        renderer,
      })
    }
    return layers
  } finally {
    host.remove()
  }
}
