interface DrawableCanvas extends HTMLCanvasElement {
  requestPaint(): void
}

interface DrawableContext extends CanvasRenderingContext2D {
  drawElementImage(
    element: HTMLElement,
    x: number,
    y: number,
    width: number,
    height: number,
  ): DOMMatrix | void
}

export interface HtmlCanvasPreview {
  update(scale: number, visible: boolean): void
  dispose(): void
}

function drawableCanvas(canvas: HTMLCanvasElement): canvas is DrawableCanvas {
  return 'requestPaint' in canvas && typeof canvas.requestPaint === 'function'
}

function drawableContext(context: CanvasRenderingContext2D | null): context is DrawableContext {
  return (
    context !== null &&
    'drawElementImage' in context &&
    typeof context.drawElementImage === 'function'
  )
}

// The native API is still experimental. Keep the same live HTML when it is unavailable.
export function mountHtmlCanvasPreview(
  content: HTMLElement,
  width: number,
  height: number,
  signal: AbortSignal,
): HtmlCanvasPreview | null {
  const host = content.parentElement
  const candidate = document.createElement('canvas')
  content.dataset.canvasRenderer = 'html'
  if (!host || !drawableCanvas(candidate) || width <= 0 || height <= 0) return null
  const candidateContext = candidate.getContext('2d')
  if (!drawableContext(candidateContext)) return null
  const parent = host
  const canvas = candidate
  const context = candidateContext

  canvas.className = 'canvas-html-surface'
  canvas.width = 0
  canvas.height = 0
  canvas.setAttribute('layoutsubtree', '')
  canvas.setAttribute('content', 'drawable')
  content.setAttribute('drawable', '')
  content.dataset.canvasRenderer = 'native'
  let visible = false
  let disposed = false

  function dispose() {
    if (disposed) return
    disposed = true
    canvas.removeEventListener('paint', paint)
    signal.removeEventListener('abort', dispose)
    if (canvas.parentElement === parent) parent.replaceChildren(content)
    content.style.removeProperty('transform')
    content.removeAttribute('drawable')
    content.dataset.canvasRenderer = 'html'
    canvas.width = 0
    canvas.height = 0
  }

  function paint() {
    if (disposed || !visible || !content.isConnected) return
    try {
      context.reset()
      // The default snapshot size already includes device density. An explicit destination
      // keeps raster resolution from changing the card's HTML dimensions.
      const transform = context.drawElementImage(content, 0, 0, canvas.width, canvas.height)
      // Newer implementations synchronize geometry during drawing; earlier ones return it.
      if (transform) {
        const cssTransform = transform.toString()
        if (content.style.transform !== cssTransform) content.style.transform = cssTransform
      }
    } catch (error) {
      console.warn('Native canvas preview unavailable; using live HTML.', error)
      dispose()
    }
  }

  function update(scale: number, nextVisible: boolean) {
    if (disposed) return
    const wasVisible = visible
    visible = nextVisible
    if (!visible) {
      if (wasVisible) {
        canvas.width = 0
        canvas.height = 0
      }
      return
    }
    // Quantize backing resolution to avoid allocating a bitmap for every zoom event.
    const requested = Math.pow(2, Math.ceil(Math.log2(Math.max(0.5, scale) * devicePixelRatio)))
    const resolution = Math.min(4, requested, Math.sqrt(2_000_000 / (width * height)))
    const pixelWidth = Math.max(1, Math.round(width * resolution))
    const pixelHeight = Math.max(1, Math.round(height * resolution))
    if (canvas.width !== pixelWidth || canvas.height !== pixelHeight || !wasVisible) {
      canvas.width = pixelWidth
      canvas.height = pixelHeight
      canvas.requestPaint()
    }
  }

  canvas.addEventListener('paint', paint, { signal })
  signal.addEventListener('abort', dispose, { once: true })
  parent.replaceChildren(canvas)
  canvas.append(content)
  return { update, dispose }
}
