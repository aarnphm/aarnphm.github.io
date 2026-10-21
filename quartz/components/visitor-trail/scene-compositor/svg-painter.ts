import type { SceneTree } from './scene-partition'
import { ctm, identity, invert, multiply, scaleOf, type Matrix } from './affine-geometry'
import { reference, shapes, unpainted, type Scope } from './render-model'

export function createSvgPaint(
  scope: Scope,
  sceneTree: SceneTree,
  current: () => CanvasRenderingContext2D,
) {
  const { view, svg, is, tree, shared } = scope
  const { localBox } = sceneTree
  // Raster: draws an SVG subtree into the scratch canvas from computed styles.
  const paths = new WeakMap<Element, { d: string; grow: number; path: Path2D }>()
  // Pixel art is one square per cell, `M x y h s v s h -s Z`. The canvas antialiases every edge,
  // and cells of different tones abut on a dither grid at fractional texels, so each shared edge
  // left a blended line and some let the layer behind through. Each cell grows by one texel to
  // the right and below, under the cells drawn after it, so solid paint covers every shared edge.
  // A stroke in the fill's paint did the same and cost half as much again per raster.
  const cells = (d: string, grow: number) => {
    if (!d.startsWith('M') || !d.endsWith('Z')) return new view.Path2D(d)
    const firstH = d.indexOf('h', 1)
    const firstV = firstH > 0 ? d.indexOf('v', firstH + 1) : -1
    if (firstV < 0) return new view.Path2D(d)
    const step = d.slice(firstH + 1, firstV)
    const stepNum = Number(step)
    if (!step || !Number.isFinite(stepNum)) return new view.Path2D(d)
    const suffix = `h${step}v${step}h-${step}Z`
    const size = stepNum + grow
    const path = new view.Path2D()
    let index = 0
    while (index < d.length) {
      if (d.charCodeAt(index) !== 77) return new view.Path2D(d)
      const space = d.indexOf(' ', index + 1)
      if (space < 0) return new view.Path2D(d)
      const h = d.indexOf('h', space + 1)
      if (h < 0 || !d.startsWith(suffix, h)) return new view.Path2D(d)
      const x = Number(d.slice(index + 1, space))
      const y = Number(d.slice(space + 1, h))
      if (!Number.isFinite(x) || !Number.isFinite(y)) return new view.Path2D(d)
      path.rect(x, y, size, size)
      index = h + suffix.length
    }
    return path
  }
  const geometry = (element: SVGGraphicsElement, style: CSSStyleDeclaration, grow = 0) => {
    const number = (value: string) => Number.parseFloat(value) || 0
    const path = new view.Path2D()
    switch (element.localName) {
      case 'path': {
        const d = element.getAttribute('d') ?? ''
        const known = paths.get(element)
        if (known?.d === d && known.grow === grow) return known.path
        const next = grow ? cells(d, grow) : new view.Path2D(d)
        paths.set(element, { d, grow, path: next })
        return next
      }
      case 'ellipse': {
        const rx = number(style.rx)
        const ry = style.ry === 'auto' ? rx : number(style.ry)
        path.ellipse(number(style.cx), number(style.cy), rx, ry, 0, 0, Math.PI * 2)
        return path
      }
      case 'circle':
        path.arc(number(style.cx), number(style.cy), number(style.r), 0, Math.PI * 2)
        return path
      case 'rect': {
        const width = number(style.width)
        const height = number(style.height)
        const rx = Math.min(width / 2, number(style.rx === 'auto' ? style.ry : style.rx))
        const ry = Math.min(height / 2, number(style.ry === 'auto' ? style.rx : style.ry))
        if (rx || ry)
          path.roundRect(number(style.x), number(style.y), width, height, [{ x: rx, y: ry }])
        else path.rect(number(style.x), number(style.y), width, height)
        return path
      }
      case 'line': {
        const line = element as SVGLineElement
        path.moveTo(line.x1.baseVal.value, line.y1.baseVal.value)
        path.lineTo(line.x2.baseVal.value, line.y2.baseVal.value)
        return path
      }
      default: {
        const points = [...(element as SVGPolylineElement).points]
        points.forEach((point, index) =>
          index ? path.lineTo(point.x, point.y) : path.moveTo(point.x, point.y),
        )
        if (element.localName === 'polygon') path.closePath()
        return path
      }
    }
  }
  const paintFor = (
    value: string,
    element: SVGGraphicsElement,
    shape: Path2D,
  ): { style: string | CanvasGradient; shape: Path2D; box: Matrix } | null => {
    const paint = current()
    const id = reference(value)
    if (!id) return value === 'none' ? null : { style: value, shape, box: identity }
    const source = tree().getElementById(id)
    if (!is(source, 'SVGGradientElement')) {
      const fallback = value.replace(/url\([^)]*\)/, '').trim()
      return fallback && fallback !== 'none' ? { style: fallback, shape, box: identity } : null
    }
    const bounding = source.getAttribute('gradientUnits') !== 'userSpaceOnUse'
    let box = identity
    let local = shape
    if (bounding) {
      const bounds = localBox(element)
      if (!bounds || !bounds.width || !bounds.height) return null
      box = [bounds.width, 0, 0, bounds.height, bounds.x, bounds.y]
      local = new view.Path2D()
      const inverse = invert(box)
      local.addPath(shape, {
        a: inverse[0],
        b: inverse[1],
        c: inverse[2],
        d: inverse[3],
        e: inverse[4],
        f: inverse[5],
      })
    }
    const length = (name: string, fallback: string) => {
      const raw = source.getAttribute(name) ?? fallback
      const value = Number.parseFloat(raw)
      if (!raw.endsWith('%')) return value
      return bounding
        ? value / 100
        : (value / 100) *
            (name.includes('y') ? svg.viewBox.baseVal.height : svg.viewBox.baseVal.width)
    }
    const gradient = is(source, 'SVGRadialGradientElement')
      ? paint.createRadialGradient(
          length('fx', source.getAttribute('cx') ?? '50%'),
          length('fy', source.getAttribute('cy') ?? '50%'),
          0,
          length('cx', '50%'),
          length('cy', '50%'),
          length('r', '50%'),
        )
      : paint.createLinearGradient(
          length('x1', '0%'),
          length('y1', '0%'),
          length('x2', '100%'),
          length('y2', '0%'),
        )
    let offset = 0
    for (const stop of source.querySelectorAll('stop')) {
      const raw = stop.getAttribute('offset') ?? '0'
      offset = Math.min(
        1,
        Math.max(offset, Number.parseFloat(raw) / (raw.endsWith('%') ? 100 : 1) || 0),
      )
      const stopStyle = view.getComputedStyle(stop)
      const opacity = Number(stopStyle.stopOpacity)
      const color = stopStyle.stopColor
      gradient.addColorStop(
        offset,
        opacity < 1 ? `color-mix(in srgb, ${color} ${opacity * 100}%, transparent)` : color,
      )
    }
    return { style: gradient, shape: local, box }
  }
  const dashes = (element: SVGGraphicsElement, style: CSSStyleDeclaration) => {
    if (style.strokeDasharray === 'none') return { list: [] as number[], offset: 0 }
    const length = Number(element.getAttribute('pathLength'))
    const factor =
      length > 0 && is(element, 'SVGGeometryElement') ? element.getTotalLength() / length : 1
    const list = style.strokeDasharray
      .split(/[\s,]+/)
      .map(part => (Number.parseFloat(part) || 0) * factor)
    return {
      list: list.length % 2 ? [...list, ...list] : list,
      offset: (Number.parseFloat(style.strokeDashoffset) || 0) * factor,
    }
  }
  const draw = (
    matrix: Matrix,
    alpha: number,
    style: CSSStyleDeclaration,
    element: SVGGraphicsElement,
    shape: Path2D,
  ) => {
    const paint = current()
    let painted = false
    const fill = paintFor(style.fill, element, shape)
    if (fill) {
      const opacity = alpha * Number(style.fillOpacity)
      if (opacity > 0) {
        paint.setTransform(...multiply(matrix, fill.box))
        paint.globalAlpha = opacity
        paint.fillStyle = fill.style
        paint.fill(fill.shape, style.fillRule === 'evenodd' ? 'evenodd' : 'nonzero')
        painted = true
      }
    }
    const width = Number.parseFloat(style.strokeWidth) || 0
    const stroke = width > 0 ? paintFor(style.stroke, element, shape) : null
    if (stroke) {
      const opacity = alpha * Number(style.strokeOpacity)
      if (opacity > 0) {
        paint.setTransform(...multiply(matrix, stroke.box))
        paint.globalAlpha = opacity
        paint.strokeStyle = stroke.style
        const fixed = style.vectorEffect === 'non-scaling-stroke'
        paint.lineWidth =
          (fixed ? (width * shared.resolution) / scaleOf(matrix) : width) / scaleOf(stroke.box)
        paint.lineCap = style.strokeLinecap as CanvasLineCap
        paint.lineJoin = (
          style.strokeLinejoin === 'arcs' || style.strokeLinejoin === 'miter-clip'
            ? 'miter'
            : style.strokeLinejoin
        ) as CanvasLineJoin
        paint.miterLimit = Number(style.strokeMiterlimit) || 4
        const dash = dashes(element, style)
        paint.setLineDash(dash.list)
        paint.lineDashOffset = dash.offset
        paint.stroke(stroke.shape)
        paint.setLineDash([])
        painted = true
      }
    }
    return painted
  }
  const write = (
    element: SVGTextElement,
    style: CSSStyleDeclaration,
    matrix: Matrix,
    alpha: number,
  ) => {
    const paint = current()
    const along = element.querySelector('textPath')
    const textStyle = along ? view.getComputedStyle(along) : style
    const fill = paintFor(textStyle.fill, element, new view.Path2D())
    const content = element.textContent ?? ''
    if (!fill || !content.trim() || textStyle.visibility !== 'visible') return false
    paint.setTransform(...matrix)
    paint.globalAlpha = alpha * Number(textStyle.fillOpacity)
    paint.fillStyle = fill.style
    paint.font = `${textStyle.fontStyle} ${textStyle.fontWeight} ${textStyle.fontSize} ${textStyle.fontFamily}`
    paint.textBaseline = 'alphabetic'
    const anchor = textStyle.textAnchor
    const first = (list: SVGAnimatedLengthList) =>
      list.baseVal.numberOfItems ? list.baseVal.getItem(0).value : 0
    const dx = first(element.dx)
    const dy = first(element.dy)
    if (!along) {
      paint.textAlign = anchor === 'middle' ? 'center' : anchor === 'end' ? 'right' : 'left'
      paint.fillText(content, first(element.x) + dx, first(element.y) + dy)
      return true
    }
    const id = reference(`url(${along.getAttribute('href') ?? ''})`)
    const route = id ? tree().getElementById(id) : null
    if (!is(route, 'SVGGeometryElement') || !route.getAttribute('d')) return false
    const length = route.getTotalLength()
    const startRaw = along.getAttribute('startOffset') ?? '0'
    const start = startRaw.endsWith('%')
      ? (Number.parseFloat(startRaw) / 100) * length
      : Number.parseFloat(startRaw) || 0
    const total = paint.measureText(content).width
    let distance = start + dx - (anchor === 'middle' ? total / 2 : anchor === 'end' ? total : 0)
    paint.textAlign = 'center'
    for (const glyph of content) {
      const advance = paint.measureText(glyph).width
      const middle = distance + advance / 2
      distance += advance
      if (middle < 0 || middle > length) continue
      const point = route.getPointAtLength(middle)
      const ahead = route.getPointAtLength(Math.min(length, middle + 0.5))
      const behind = route.getPointAtLength(Math.max(0, middle - 0.5))
      const angle = Math.atan2(ahead.y - behind.y, ahead.x - behind.x)
      const cos = Math.cos(angle)
      const sin = Math.sin(angle)
      paint.setTransform(...multiply(matrix, [cos, sin, -sin, cos, point.x, point.y]))
      paint.fillText(glyph, 0, dy)
    }
    return true
  }
  const clipPathOf = (source: SVGClipPathElement, matrix: Matrix) => {
    const shape = new view.Path2D()
    let rule: CanvasFillRule = 'nonzero'
    for (const child of source.children) {
      if (!is(child, 'SVGGeometryElement')) continue
      const style = view.getComputedStyle(child)
      if (style.display === 'none') continue
      if (style.clipRule === 'evenodd') rule = 'evenodd'
      shape.addPath(geometry(child, style), {
        a: matrix[0],
        b: matrix[1],
        c: matrix[2],
        d: matrix[3],
        e: matrix[4],
        f: matrix[5],
      })
    }
    return { shape, rule }
  }
  const render = (
    element: Element,
    matrix: Matrix,
    alpha: number,
    toTexels: Matrix,
    self: boolean,
  ): boolean => {
    const paint = current()
    if (
      !is(element, 'SVGGraphicsElement') ||
      unpainted.has(element.localName) ||
      element.hasAttribute('data-trail-model')
    )
      return false
    if (is(element, 'SVGForeignObjectElement')) return false
    const style = view.getComputedStyle(element)
    if (style.display === 'none') return false
    const local = !self && style.transform !== 'none' ? multiply(toTexels, ctm(element)) : matrix
    const opacity = self ? alpha : alpha * Number(style.opacity)
    if (opacity <= 0) return false
    const clipId = reference(style.clipPath)
    const clipSource = clipId ? tree().getElementById(clipId) : null
    if (is(clipSource, 'SVGClipPathElement')) {
      paint.save()
      paint.setTransform(1, 0, 0, 1, 0, 0)
      const { shape, rule } = clipPathOf(clipSource, local)
      paint.clip(shape, rule)
    }
    let painted = false
    if (shapes.has(element.localName)) {
      if (style.visibility === 'visible')
        painted = draw(
          local,
          opacity,
          style,
          element,
          geometry(element, style, style.shapeRendering === 'crispedges' ? 1 / scaleOf(local) : 0),
        )
    } else if (is(element, 'SVGTextElement')) {
      painted = write(element, style, local, opacity)
    } else
      for (const child of element.children)
        painted = render(child, local, opacity, toTexels, false) || painted
    if (is(clipSource, 'SVGClipPathElement')) paint.restore()
    return painted
  }
  return { render, clipPathOf }
}
