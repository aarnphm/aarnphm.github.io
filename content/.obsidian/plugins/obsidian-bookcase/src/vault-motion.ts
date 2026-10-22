import { lidRibs } from './vault-model'

const SVG_NS = 'http://www.w3.org/2000/svg'

interface Bounds {
  x: number
  y: number
  width: number
  height: number
}

const overlaps = (a: Bounds, b: Bounds) =>
  a.x < b.x + b.width && b.x < a.x + a.width && a.y < b.y + b.height && b.y < a.y + a.height

const swept = (box: Bounds, can: SVGGElement): Bounds => {
  const dx = parseFloat(can.style.getPropertyValue('--px')) || 0
  const dy = parseFloat(can.style.getPropertyValue('--py')) || 0
  return {
    x: box.x + Math.min(0, dx) - 2,
    y: box.y + Math.min(0, dy) - 2,
    width: box.width + Math.abs(dx) + 4,
    height: box.height + Math.abs(dy) + 4,
  }
}

const element = <K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string>) => {
  const node = document.createElementNS(SVG_NS, tag)
  for (const [name, value] of Object.entries(attrs)) node.setAttribute(name, value)
  return node
}

/** Two lines of at most `max` characters for the round lid label, as the garden's. */
function lidLines(title: string, max = 12): string[] {
  const lines = ['']
  for (const word of title.toUpperCase().split(/\s+/)) {
    const line = lines[lines.length - 1]
    if (!line || line.length + word.length + 1 <= max)
      lines[lines.length - 1] = line ? `${line} ${word}` : word
    else if (lines.length < 2) lines.push(word)
    else {
      lines[1] = `${lines[1]}…`
      break
    }
  }
  return lines.map(line => (line.length > max + 1 ? `${line.slice(0, max - 1)}…` : line))
}

/** Embossed rings and the round lid label, laid on the pulled can's lid. */
function lidDetail(can: SVGGElement, title: string): SVGGElement | null {
  const lid = can.querySelector('.cn-lid')
  const transform = lid?.getAttribute('transform')
  const r = Number(lid?.getAttribute('r'))
  if (!transform || !Number.isFinite(r)) return null
  const group = element('g', { class: 'cn-lid-detail', transform })
  const size = Math.round(r * 0.075 * 10) / 10
  const text = element('text', { class: 'cn-lid-text', 'font-size': String(size) })
  const lines = lidLines(title)
  lines.forEach((line, i) => {
    const span = element('tspan', {
      x: '0',
      y: String((i - (lines.length - 1) / 2) * size * 1.25 + size * 0.35),
    })
    span.textContent = line
    text.append(span)
  })
  const relief = element('g', { class: 'cn-rings' })
  relief.append(
    element('circle', { class: 'cn-lid-ring', r: String(Math.round(r * 9) / 10) }),
    element('path', { class: 'cn-lid-ribs', d: lidRibs(r) }),
  )
  group.append(
    relief,
    element('circle', { class: 'cn-lid-label', r: String(Math.round(r * 0.28)) }),
    text,
  )
  return group
}

/** Isolate paint to the moving can's sweep, with the original scene's foreground as its mask. */
export function createVaultMotion(svg: SVGSVGElement) {
  const prefix = `vault-${crypto.randomUUID()}`
  const assigned = new Set<SVGElement>()
  const ids = new WeakMap<SVGElement, string>()
  let nextId = 0
  const idOf = (node: SVGElement) => {
    let id = ids.get(node)
    if (!id) {
      id = node.id || `${prefix}-${nextId++}`
      if (!node.id) {
        node.id = id
        assigned.add(node)
      }
      ids.set(node, id)
    }
    return id
  }
  const contour = (node: SVGGraphicsElement) => {
    const group = element('g', {})
    const shapes =
      node instanceof SVGGElement
        ? Array.from(node.querySelectorAll<SVGGraphicsElement>('.cn-band, .cn-lid'))
        : [node]
    for (const shape of shapes) {
      const copy = shape.cloneNode(false)
      if (!(copy instanceof SVGElement)) continue
      copy.removeAttribute('id')
      copy.removeAttribute('class')
      copy.removeAttribute('style')
      copy.setAttribute('fill', 'black')
      copy.setAttribute('stroke', 'black')
      copy.setAttribute('stroke-width', '0.75')
      group.append(copy)
    }
    return group
  }
  const bounds = new WeakMap<SVGGraphicsElement, Bounds>()
  const boundsOf = (node: SVGGraphicsElement): Bounds => {
    const previous = bounds.get(node)
    if (previous) return previous
    let box: Bounds
    if (node instanceof SVGGElement && node.classList.contains('cn-can')) {
      // Conservative projected bounds also work before this can's contents are hydrated.
      const r = Number(node.dataset.r)
      box = swept(
        {
          x: Number(node.dataset.cx) - 50,
          y: Number(node.dataset.cy) - 2 * r,
          width: 2 * r + 100,
          height: 3 * r,
        },
        node,
      )
    } else box = node.getBBox()
    bounds.set(node, box)
    return box
  }
  const originals = Array.from(svg.children).filter(
    (node): node is SVGGraphicsElement =>
      node instanceof SVGGraphicsElement &&
      (node.classList.contains('cn-can') ||
        node.classList.contains('cn-face') ||
        node.classList.contains('cn-holder')),
  )
  const moving = new Map<
    SVGGElement,
    {
      can: SVGGElement
      layer: SVGSVGElement
      foreground: SVGGElement
      inkId: string
      front: Map<SVGGraphicsElement, SVGGElement>
      references: Map<SVGGElement, SVGUseElement>
    }
  >()
  const addMoving = (
    motion: NonNullable<ReturnType<typeof moving.get>>,
    can: SVGGElement,
    clone: SVGGElement,
  ) => {
    const shape = motion.front.get(can)
    if (!shape) return
    shape.style.display = 'none'
    const reference = element('use', { href: `#${idOf(clone)}`, filter: `url(#${motion.inkId})` })
    motion.references.set(clone, reference)
    motion.foreground.append(reference)
  }
  const mount = (can: SVGGElement, title: string) => {
    const clone = can.cloneNode(true)
    if (!(clone instanceof SVGGElement)) return undefined
    clone.removeAttribute('id')
    clone.querySelector('.cn-rings')?.remove()
    const detail = lidDetail(clone, title)
    if (detail) clone.append(detail)
    clone.classList.add('cn-pulled')
    const box = swept(can.getBBox(), can)
    const { x, y, width, height } = box
    const layer = element('svg', {
      class: 'vault-scene vault-motion',
      viewBox: `${x} ${y} ${width} ${height}`,
      width: String(width),
      height: String(height),
      'aria-hidden': 'true',
    })
    layer.style.left = `${x - svg.viewBox.baseVal.x}px`
    layer.style.top = `${y - svg.viewBox.baseVal.y}px`
    const region = { x: String(x), y: String(y), width: String(width), height: String(height) }
    const maskId = `${prefix}-mask-${nextId++}`
    const inkId = `${maskId}-ink`
    const ink = element('filter', { id: inkId, filterUnits: 'userSpaceOnUse', ...region })
    ink.append(
      element('feColorMatrix', {
        type: 'matrix',
        values: '0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 1 0',
      }),
    )
    const mask = element('mask', {
      id: maskId,
      maskUnits: 'userSpaceOnUse',
      'mask-type': 'luminance',
      ...region,
    })
    const foreground = element('g', {})
    const front = new Map<SVGGraphicsElement, SVGGElement>()
    for (const node of originals.slice(originals.indexOf(can) + 1)) {
      if (overlaps(box, boundsOf(node))) front.set(node, contour(node))
    }
    foreground.append(...front.values())
    mask.append(element('rect', { ...region, fill: 'white' }), foreground)
    const defs = element('defs', {})
    defs.append(ink, mask)
    const visible = element('g', { mask: `url(#${maskId})` })
    visible.append(clone)
    layer.append(defs, visible)
    const motion = {
      can,
      layer,
      foreground,
      inkId,
      front,
      references: new Map<SVGGElement, SVGUseElement>(),
    }
    for (const [other, existing] of moving) {
      addMoving(motion, existing.can, other)
      addMoving(existing, can, clone)
    }
    moving.set(clone, motion)
    can.style.opacity = '0'
    svg.parentElement?.append(layer)
    return clone
  }
  const remove = (clone: SVGGElement) => {
    const motion = moving.get(clone)
    if (!motion) return
    motion.layer.remove()
    moving.delete(clone)
    assigned.delete(clone)
    for (const other of moving.values()) {
      other.references.get(clone)?.remove()
      other.references.delete(clone)
      const shape = other.front.get(motion.can)
      if (shape) shape.style.display = ''
    }
  }
  const dispose = () => {
    for (const clone of moving.keys()) remove(clone)
    for (const node of assigned) node.removeAttribute('id')
    assigned.clear()
  }
  return { mount, remove, dispose }
}

export type VaultMotion = ReturnType<typeof createVaultMotion>
