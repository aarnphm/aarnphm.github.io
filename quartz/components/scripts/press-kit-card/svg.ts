import type { Fonts } from '../press-kit-card'
import { SPORT_ICON, isActivityKind } from '../../../plugins/stores/strava'

const NS = 'http://www.w3.org/2000/svg'

export function node(tag: string, attributes: Record<string, string | number> = {}): SVGElement {
  const element = document.createElementNS(NS, tag)
  for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, String(value))
  return element
}

export function svgImage(svg: string): Promise<HTMLImageElement> {
  const url = URL.createObjectURL(new Blob([svg], { type: 'image/svg+xml' }))
  return new Promise((resolve, reject) => {
    const image = new Image()
    image.onload = () => {
      URL.revokeObjectURL(url)
      resolve(image)
    }
    image.onerror = () => {
      URL.revokeObjectURL(url)
      reject(new Error('Unable to render card image'))
    }
    image.src = url
  })
}

export function createSvgText(fonts: Fonts, scale: number) {
  function measure(value: string, size: number, weight = 400, family = fonts.body) {
    const context = document.createElement('canvas').getContext('2d')
    if (!context) return null
    context.font = `${weight} ${Math.round(size)}px ${family}`
    return context.measureText(value)
  }
  function text(
    parent: SVGElement,
    value: string,
    x: number,
    y: number,
    size: number,
    color: string,
    weight = 400,
    family = fonts.body,
  ): void {
    if (!value) return
    const canvas = document.createElement('canvas')
    const context = canvas.getContext('2d')
    if (!context) return
    const font = `${weight} ${Math.round(size)}px ${family}`
    context.font = font
    const metrics = context.measureText(value)
    const pad = 2
    const left = Math.ceil(Math.max(0, metrics.actualBoundingBoxLeft)) + pad
    const top = Math.ceil(metrics.actualBoundingBoxAscent) + pad
    const width = Math.ceil(Math.max(metrics.width, metrics.actualBoundingBoxRight)) + left + pad
    const height = top + Math.ceil(metrics.actualBoundingBoxDescent) + pad
    canvas.width = Math.ceil(width * scale)
    canvas.height = Math.ceil(height * scale)
    context.scale(scale, scale)
    context.font = font
    context.fillStyle = color
    context.fillText(value, left, top)
    const anchor = parent.getAttribute('text-anchor')
    const offset = anchor === 'middle' ? metrics.width / 2 : anchor === 'end' ? metrics.width : 0
    // SVG images cannot use the document's webfonts. Paint text with the same loaded
    // canvas faces at export density before composing the final PNG.
    parent.append(
      node('image', {
        x: x - left - offset,
        y: y - top,
        width: canvas.width / scale,
        height: canvas.height / scale,
        href: canvas.toDataURL('image/png'),
      }),
    )
  }

  function lines(
    value: string,
    width: number,
    size: number,
    weight = 400,
    family = fonts.body,
  ): string[] {
    const measure = document.createElement('canvas').getContext('2d')
    if (!measure) return [value]
    measure.font = `${weight} ${Math.round(size)}px ${family}`
    const result: string[] = []
    let line = ''
    for (const word of value.split(/\s+/).filter(Boolean)) {
      const next = line ? `${line} ${word}` : word
      if (line && measure.measureText(next).width > width) {
        result.push(line)
        line = word
      } else line = next
    }
    if (line) result.push(line)
    return result
  }

  function fitText(
    parent: SVGElement,
    value: string,
    x: number,
    y: number,
    width: number,
    size: number,
    color: string,
    weight = 400,
    family = fonts.body,
  ): void {
    const measure = document.createElement('canvas').getContext('2d')
    if (measure) {
      measure.font = `${weight} ${Math.round(size)}px ${family}`
      const measured = measure.measureText(value).width
      if (measured > width) size *= width / measured
    }
    text(parent, value, x, y, size, color, weight, family)
  }
  function fitValue(
    parent: SVGElement,
    value: string,
    x: number,
    y: number,
    width: number,
    size: number,
    color: string,
  ): void {
    fitText(parent, value, x, y, width, size, color, 400, fonts.mono)
  }

  function metricRow(
    parent: SVGElement,
    label: string,
    value: string,
    x: number,
    baseline: number,
    width: number,
    size: number,
    color: string,
    ruled = true,
  ): void {
    if (ruled)
      parent.append(
        node('path', {
          d: `M${x} ${baseline + size * 0.45}H${x + width}`,
          stroke: color,
          opacity: 0.12,
        }),
      )
    fitText(parent, label, x, baseline, width * 0.42, size, color)
    const values = node('g', { 'text-anchor': 'end' })
    fitValue(values, value, x + width, baseline, width * 0.54, size, color)
    parent.append(values)
  }

  function titleRow(
    parent: SVGElement,
    value: string,
    sport: string | undefined,
    x: number,
    y: number,
    width: number,
    size: number,
    color: string,
    align: 'left' | 'center' = 'left',
  ): number {
    const iconSize = size * 0.95
    const icon = sport ? sportIcon(sport, 0, 0, iconSize, color) : undefined
    const offset = icon ? iconSize + size * 0.3 : 0
    const titleLines = lines(value, width - offset, size, 500)
    const metrics = titleLines.map(line => measure(line, size, 500))
    const titleWidth = Math.min(
      width - offset,
      Math.max(0, ...metrics.map(item => item?.width ?? 0)),
    )
    const ascent = Math.max(size * 0.6, ...metrics.map(item => item?.actualBoundingBoxAscent ?? 0))
    const descent = Math.max(0, ...metrics.map(item => item?.actualBoundingBoxDescent ?? 0))
    const textHeight = ascent + descent + Math.max(0, titleLines.length - 1) * size * 1.2
    const height = Math.max(icon ? iconSize : 0, textHeight)
    const left = x + (align === 'center' ? Math.max(0, (width - titleWidth - offset) / 2) : 0)
    if (icon) {
      icon.setAttribute('x', String(left))
      icon.setAttribute('y', String(y + (height - iconSize) / 2))
      parent.append(icon)
    }
    titleLines.forEach((line, index) =>
      fitText(
        parent,
        line,
        left + offset,
        y + (height - textHeight) / 2 + ascent + index * size * 1.2,
        titleWidth,
        size,
        color,
        500,
      ),
    )
    return height
  }
  return { text, lines, fitText, fitValue, metricRow, titleRow, measure }
}

export function graphic(
  parent: SVGElement,
  markup: string,
  x: number,
  y: number,
  width: number,
  height: number,
): void {
  const parsed = new DOMParser().parseFromString(markup, 'image/svg+xml').documentElement
  if (parsed.localName !== 'svg') return
  parsed.setAttribute('x', String(x))
  parsed.setAttribute('y', String(y))
  parsed.setAttribute('width', String(width))
  parsed.setAttribute('height', String(height))
  parsed.setAttribute('preserveAspectRatio', 'xMidYMid meet')
  parent.append(document.importNode(parsed, true))
}

export function sportIcon(
  sport: string,
  x: number,
  y: number,
  size: number,
  ink: string,
): SVGElement | undefined {
  if (!isActivityKind(sport)) return undefined
  const solid = sport === 'treatment' || sport === 'yoga'
  const icon = node('svg', {
    x,
    y,
    width: size,
    height: size,
    viewBox: '0 0 24 24',
    fill: solid ? ink : 'none',
    stroke: solid ? 'none' : ink,
    'stroke-width': 1.5,
    'stroke-linecap': 'round',
    'stroke-linejoin': 'round',
  })
  for (const d of SPORT_ICON[sport]) icon.append(node('path', { d }))
  return icon
}
