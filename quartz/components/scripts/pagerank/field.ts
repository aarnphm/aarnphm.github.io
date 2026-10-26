import type { PageRankModel, PageRankNode } from '../../../util/pagerank-view'
import { fieldLayout, type FieldLayout } from '../../../util/pagerank-field'
import { formatTimes } from './views'

const SVG = 'http://www.w3.org/2000/svg'
const TAU = Math.PI * 2
/** Pointer distance, in px, within which a dot counts as hovered or tapped. */
const reach = 8
const tapReach = 14

const svgNode = <K extends keyof SVGElementTagNameMap>(
  tag: K,
  attributes: Record<string, string | number>,
  text?: string,
): SVGElementTagNameMap[K] => {
  const node = document.createElementNS(SVG, tag)
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value))
  if (text !== undefined) node.textContent = text
  return node
}

const span = (className: string, text: string) => {
  const node = document.createElement('span')
  if (className) node.className = className
  node.textContent = text
  return node
}

export interface FieldEvents {
  /** The hovered dot changed; `null` when the pointer leaves every dot. */
  point(node: PageRankNode | null): void
  select(node: PageRankNode): void
}

export interface Field {
  render(width: number): void
  mark(selected: PageRankNode | null, pointer: PageRankNode | null): void
  /** Repaints with the current theme colours. */
  paint(): void
  layoutMs: number
}

/** Every note as one dot on the log score axis; canvas marks, SVG axis, HTML annotations. */
export function pagerankField(
  host: HTMLElement,
  model: PageRankModel,
  events: FieldEvents,
  signal: AbortSignal,
): Field {
  const canvas = host.querySelector<HTMLCanvasElement>('canvas')!
  const overlay = host.querySelector<SVGSVGElement>('svg')!
  const notes = host.querySelector<HTMLElement>('[data-pr-field-notes]')!
  const context = canvas.getContext('2d')!
  const { incoming, outgoing } = model.graph
  let layout: FieldLayout | null = null
  let selected: PageRankNode | null = null
  let pointer: PageRankNode | null = null
  let sources = new Set<number>()
  let targets = new Set<number>()

  const paint = () => {
    if (!layout) return
    const { x, y, radius: r, width, height } = layout
    const style = getComputedStyle(host)
    const ink = style.getPropertyValue('--dark').trim()
    const paper = style.getPropertyValue('--light').trim()
    const dpr = Math.min(2, devicePixelRatio || 1)
    context.setTransform(dpr, 0, 0, dpr, 0, 0)
    context.clearRect(0, 0, width, height)
    const dots = (ids: Iterable<number>, radius: number) => {
      context.beginPath()
      for (const id of ids) {
        context.moveTo(x[id] + radius, y[id])
        context.arc(x[id], y[id], radius, 0, TAU)
      }
    }
    const lit = (id: number) => id === selected?.id || sources.has(id) || targets.has(id)

    context.globalAlpha = 0.5
    context.fillStyle = style.getPropertyValue('--darkgray').trim()
    context.beginPath()
    for (let id = 0; id < x.length; id++) {
      if (lit(id)) continue
      context.moveTo(x[id] + r, y[id])
      context.arc(x[id], y[id], r, 0, TAU)
    }
    context.fill()
    context.globalAlpha = 1

    // Shape carries the role, so it reads without colour: solid links to it, hollow it links
    // to, solid with an outer ring both ways.
    const both = [...sources].filter(id => targets.has(id))
    context.fillStyle = ink
    dots(sources, r + 1.6)
    context.fill()
    context.fillStyle = paper
    dots(
      [...targets].filter(id => !sources.has(id)),
      r + 1.6,
    )
    context.fill()
    context.strokeStyle = ink
    context.lineWidth = 1.5
    context.stroke()
    context.lineWidth = 1
    dots(both, r + 3.4)
    context.stroke()

    if (selected) {
      dots([selected.id], r + 3.2)
      context.fillStyle = style.getPropertyValue('--pr-salmon').trim()
      context.fill()
      context.lineWidth = 1.5
      context.stroke()
    }
    if (pointer && pointer !== selected) {
      dots([pointer.id], lit(pointer.id) ? r + 5 : r + 3)
      context.lineWidth = 1.5
      context.stroke()
    }
  }

  const annotate = (field: FieldLayout) => {
    const { axisY, axisX0, axisX1, ticks, height } = field
    const axis = svgNode('g', { class: 'pagerank-ticks' })
    const lines = svgNode('g', { class: 'pagerank-field-lines' })
    lines.append(svgNode('line', { x1: axisX0, x2: axisX1, y1: axisY, y2: axisY }))
    for (const tick of ticks) {
      lines.append(svgNode('line', { x1: tick.x, x2: tick.x, y1: axisY, y2: axisY + 4 }))
      axis.append(
        svgNode('text', { x: tick.x, y: axisY + 16, 'text-anchor': 'middle' }, tick.label),
      )
    }
    if (field.blockCount) {
      const x0 = field.blockX0 - field.radius
      lines.append(
        svgNode('line', { x1: x0, x2: field.blockX1 + field.radius, y1: axisY, y2: axisY }),
      )
      for (const [index, line] of field.blockLabel.entries())
        axis.append(svgNode('text', { x: x0, y: axisY + 16 + index * 13 }, line))
    }
    const title = svgNode(
      'text',
      { x: axisX1, y: height - 4, 'text-anchor': 'end', class: 'pagerank-axis-title' },
      'score, ×mean (log)',
    )
    overlay.replaceChildren(lines, axis, title)
    overlay.setAttribute('viewBox', `0 0 ${field.width} ${field.height}`)

    const first = model.byPosition[0]
    const unlinked = model.byPosition.find(node => node.inbound === 0)
    const items: HTMLElement[] = []
    const count = span('pagerank-field-number', field.blockCount.toLocaleString('en'))
    const floor = unlinked && span('pagerank-field-number', `${formatTimes(unlinked.times)}×`)
    const block = document.createElement('p')
    if (floor) {
      block.className = 'pagerank-field-note'
      block.append(count, ' notes, each at ', floor)
      block.style.left = `${field.blockX0 - field.radius}px`
      items.push(block)
    }
    const top = document.createElement('p')
    const name = span('pagerank-field-title', first?.title ?? '')
    const mean = span('', 'the mean')
    if (first && first.inbound > 0) {
      top.className = 'pagerank-field-note pagerank-field-top'
      top.append(
        span('pagerank-field-number', '#1'),
        name,
        span('pagerank-field-number', `${formatTimes(first.times)}×`),
        mean,
      )
      top.style.right = `${field.width - field.x[first.id] - field.radius}px`
      items.push(top)
    }
    notes.replaceChildren(...items)
    if (top.isConnected && first) {
      const right = field.x[first.id] + field.radius
      // Both notes keep one line. While they collide, the #1 title truncates first, then
      // "the mean" goes, then the block note drops its words, then the title goes.
      const fits = () => {
        const blockRight = block.isConnected ? block.offsetLeft + block.offsetWidth : 0
        top.style.maxWidth = `${Math.max(0, right - blockRight - 16)}px`
        return top.scrollWidth <= top.clientWidth && (name.hidden || name.clientWidth >= 48)
      }
      const steps = [
        () => mean.remove(),
        () => floor && block.replaceChildren(count, ' at ', floor),
        () => (name.hidden = true),
      ]
      for (const step of steps) {
        if (fits()) break
        step()
      }
      fits()
      // Leader from under the label to just above the dot.
      const from = top.offsetTop + top.offsetHeight + 4
      const to = field.y[first.id] - field.radius - 5
      if (to > from) {
        const x = field.x[first.id]
        lines.append(svgNode('line', { x1: x, x2: x, y1: from, y2: to, class: 'pagerank-leader' }))
      }
    }
  }

  const field: Field = {
    layoutMs: 0,
    render(width) {
      if (width <= 0 || width === layout?.width) return
      const start = performance.now()
      layout = fieldLayout(model, width)
      field.layoutMs = performance.now() - start
      const dpr = Math.min(2, devicePixelRatio || 1)
      canvas.width = Math.round(width * dpr)
      canvas.height = Math.round(layout.height * dpr)
      host.style.setProperty('--field-height', `${layout.height}px`)
      annotate(layout)
      paint()
    },
    mark(nextSelected, nextPointer) {
      if (nextSelected !== selected) {
        selected = nextSelected
        sources = new Set(selected ? incoming[selected.id] : [])
        targets = new Set(selected ? outgoing[selected.id] : [])
      }
      pointer = nextPointer
      paint()
    },
    paint,
  }

  const local = (event: MouseEvent) => {
    const box = canvas.getBoundingClientRect()
    return [event.clientX - box.left, event.clientY - box.top] as const
  }
  let frame = 0
  let last: PointerEvent | null = null
  canvas.addEventListener(
    'pointermove',
    event => {
      if (event.pointerType === 'touch') return
      last = event
      if (frame) return
      frame = requestAnimationFrame(() => {
        frame = 0
        if (!layout || !last) return
        const id = layout.nearest(...local(last), reach)
        const node = id >= 0 ? model.nodes[id] : null
        host.toggleAttribute('data-hit', !!node)
        if (node !== pointer) events.point(node)
      })
    },
    { signal },
  )
  canvas.addEventListener(
    'pointerleave',
    () => {
      cancelAnimationFrame(frame)
      frame = 0
      last = null
      host.removeAttribute('data-hit')
      if (pointer) events.point(null)
    },
    { signal },
  )
  canvas.addEventListener(
    'click',
    event => {
      if (!layout) return
      const touch = (event as PointerEvent).pointerType === 'touch'
      const id = layout.nearest(...local(event), touch ? tapReach : reach)
      if (id >= 0) events.select(model.nodes[id])
    },
    { signal },
  )
  host.addEventListener(
    'keydown',
    event => {
      if (!selected || (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight')) return
      // Right is up the axis, so it moves to the next higher score.
      const index = selected.position - 1 + (event.key === 'ArrowRight' ? -1 : 1)
      const node = model.byPosition[index]
      if (!node) return
      event.preventDefault()
      events.select(node)
    },
    { signal },
  )
  signal.addEventListener('abort', () => cancelAnimationFrame(frame), { once: true })
  return field
}
