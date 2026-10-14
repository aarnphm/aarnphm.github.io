const paths = new WeakMap<SVGPathElement, { d: string; geometry: SVGPathElement }>()

export function svgPathGeometry(path: SVGPathElement) {
  const d = path.getAttribute('d') ?? ''
  const cached = paths.get(path)
  if (cached?.d === d) return cached.geometry
  // Trail geometry lives in d attributes. Measuring an isolated path avoids a layout of the
  // animated scene, while retaining the browser's exact curve and subpath calculations.
  const geometry = path.ownerDocument.createElementNS('http://www.w3.org/2000/svg', 'path')
  geometry.setAttribute('d', d)
  paths.set(path, { d, geometry })
  return geometry
}

function matrix2d(matrix: DOMMatrix) {
  return new DOMMatrix([matrix.a, matrix.b, matrix.c, matrix.d, matrix.e, matrix.f])
}

function isSvgGraphics(element: Element): element is SVGGraphicsElement {
  // Scene nodes can come from an iframe realm while belonging to the page document.
  return element.namespaceURI === 'http://www.w3.org/2000/svg' && 'transform' in element
}

export function svgLocalMatrix(element: SVGGraphicsElement, root: SVGSVGElement) {
  const view = root.ownerDocument.defaultView
  let matrix = new DOMMatrix()
  if (!view) return matrix
  for (
    let current: Element | null = element;
    current && current !== root;
    current = current.parentElement
  ) {
    if (!isSvgGraphics(current)) continue
    let local = new DOMMatrix()
    for (let index = 0; index < current.transform.baseVal.numberOfItems; index++)
      local = local.multiply(matrix2d(current.transform.baseVal.getItem(index).matrix))
    matrix = local.multiply(matrix)
  }
  return matrix
}

export function createSvgSpace(root: SVGSVGElement, signal: AbortSignal) {
  const view = root.ownerDocument.defaultView
  let screen: DOMMatrix | null | undefined
  const invalidate = () => {
    screen = undefined
  }
  const resize = new ResizeObserver(invalidate)
  resize.observe(root)
  const attributes = new MutationObserver(invalidate)
  attributes.observe(root, {
    attributes: true,
    attributeFilter: ['viewBox', 'width', 'height', 'preserveAspectRatio', 'transform'],
  })
  view?.addEventListener('resize', invalidate, { signal, passive: true })
  view?.addEventListener('scroll', invalidate, { signal, passive: true })
  signal.addEventListener(
    'abort',
    () => {
      resize.disconnect()
      attributes.disconnect()
      screen = undefined
    },
    { once: true },
  )
  return {
    matrix(element: SVGGraphicsElement = root) {
      if (screen === undefined) {
        const matrix = root.getScreenCTM()
        screen = matrix ? matrix2d(matrix) : null
      }
      return screen?.multiply(svgLocalMatrix(element, root)) ?? null
    },
  }
}
