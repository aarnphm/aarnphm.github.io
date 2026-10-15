type Point = { x: number; y: number }
type PathGeometry = { getTotalLength(): number; getPointAtLength(distance: number): Point }
const paths = new WeakMap<SVGPathElement, { d: string; geometry: PathGeometry }>()

function sampledGeometry(d: string, native: SVGPathElement): PathGeometry | null {
  // Generated trail routes use absolute moves, lines and cubics. Keep the browser's parser
  // for the other SVG commands and for paths with more than one subpath.
  if (/[^MLC\d.eE+\-,\s]/.test(d)) return null
  const tokens = d.match(/[MLC]|[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?/g)
  if (!tokens || tokens[0] !== 'M' || tokens.filter(token => token === 'M').length !== 1)
    return null
  const points: (Point & { distance: number })[] = []
  let length = 0
  const append = (point: Point) => {
    const previous = points.at(-1)
    if (previous) length += Math.hypot(point.x - previous.x, point.y - previous.y)
    points.push({ ...point, distance: length })
  }
  const midpoint = (a: Point, b: Point): Point => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 })
  const gap = (a: Point, b: Point) => Math.hypot(a.x - b.x, a.y - b.y)
  const cubic = (a: Point, b: Point, c: Point, end: Point, tolerance: number, depth = 0) => {
    const chord = gap(a, end)
    const polygon = gap(a, b) + gap(b, c) + gap(c, end)
    if (depth >= 18 || polygon - chord <= tolerance) {
      append(end)
      return
    }
    const ab = midpoint(a, b)
    const bc = midpoint(b, c)
    const cd = midpoint(c, end)
    const abc = midpoint(ab, bc)
    const bcd = midpoint(bc, cd)
    const center = midpoint(abc, bcd)
    cubic(a, ab, abc, center, tolerance / 2, depth + 1)
    cubic(center, bcd, cd, end, tolerance / 2, depth + 1)
  }
  const tolerance = 0.005 / Math.max(1, tokens.filter(token => token === 'C').length)
  for (let index = 0; index < tokens.length; ) {
    const command = tokens[index++]
    const count = command === 'C' ? 6 : command === 'M' || command === 'L' ? 2 : 0
    if (!count || index + count > tokens.length) return null
    const values = tokens.slice(index, index + count).map(Number)
    if (!values.every(Number.isFinite)) return null
    index += count
    if (command === 'C') {
      const previous = points.at(-1)
      if (!previous) return null
      cubic(
        previous,
        { x: values[0], y: values[1] },
        { x: values[2], y: values[3] },
        { x: values[4], y: values[5] },
        tolerance,
      )
    } else append({ x: values[0], y: values[1] })
  }
  if (!points.length) return null
  // Preserve the browser's route length; look up points on the bounded subdivision once.
  const nativeLength = native.getTotalLength()
  return {
    getTotalLength: () => nativeLength,
    getPointAtLength(distance) {
      const target = nativeLength
        ? (Math.max(0, Math.min(nativeLength, distance)) * length) / nativeLength
        : 0
      let low = 0
      let high = points.length - 1
      while (low < high) {
        const middle = (low + high) >>> 1
        if (points[middle].distance < target) low = middle + 1
        else high = middle
      }
      const right = points[low]
      const left = points[Math.max(0, low - 1)]
      const span = right.distance - left.distance
      const share = span ? (target - left.distance) / span : 0
      return { x: left.x + (right.x - left.x) * share, y: left.y + (right.y - left.y) * share }
    },
  }
}

export function svgPathGeometry(path: SVGPathElement) {
  const d = path.getAttribute('d') ?? ''
  const cached = paths.get(path)
  if (cached?.d === d) return cached.geometry
  // Trail geometry lives in d attributes. Measuring an isolated path avoids a layout of the
  // animated scene, while retaining the browser's exact curve and subpath calculations.
  const native = path.ownerDocument.createElementNS('http://www.w3.org/2000/svg', 'path')
  native.setAttribute('d', d)
  const geometry = sampledGeometry(d, native) ?? native
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
