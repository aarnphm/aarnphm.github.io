import { registerMdxContent } from './mdx-content'

const ROOT = '[data-mtp-modules]'

const visibleCols = (root: HTMLElement) =>
  Array.from(root.querySelectorAll<HTMLElement>('[data-mtp-col]')).filter(
    col => col.getAttribute('data-mtp-off') !== 'true',
  )

const rel = (stage: DOMRect, el: Element) => {
  const r = el.getBoundingClientRect()
  return {
    left: r.left - stage.left,
    top: r.top - stage.top,
    right: r.right - stage.left,
    bottom: r.bottom - stage.top,
    midX: r.left - stage.left + r.width / 2,
    midY: r.top - stage.top + r.height / 2,
    width: r.width,
    height: r.height,
  }
}

const paint = (root: HTMLElement) => {
  const stage = root.querySelector<HTMLElement>('[data-mtp-stage]')
  const svg = root.querySelector<SVGSVGElement>('[data-mtp-links]')
  if (!stage || !svg) return

  const stageBox = stage.getBoundingClientRect()
  const w = Math.max(1, stage.clientWidth)
  const h = Math.max(1, stage.clientHeight)
  svg.setAttribute('viewBox', `0 0 ${w} ${h}`)
  svg.setAttribute('width', String(w))
  svg.setAttribute('height', String(h))

  while (svg.firstChild) svg.removeChild(svg.firstChild)

  const ns = 'http://www.w3.org/2000/svg'
  const defs = document.createElementNS(ns, 'defs')
  const marker = document.createElementNS(ns, 'marker')
  const markerId = `mtp-arrow-${Math.random().toString(36).slice(2, 8)}`
  marker.setAttribute('id', markerId)
  // Open chevron, tip on the path end; user-space units keep it 6x8px at any figure size.
  marker.setAttribute('markerUnits', 'userSpaceOnUse')
  marker.setAttribute('viewBox', '0 0 6 8')
  marker.setAttribute('refX', '5.5')
  marker.setAttribute('refY', '4')
  marker.setAttribute('markerWidth', '6')
  marker.setAttribute('markerHeight', '8')
  marker.setAttribute('orient', 'auto')
  const tip = document.createElementNS(ns, 'path')
  tip.setAttribute('d', 'M0.5,0.5 L5.5,4 L0.5,7.5')
  tip.setAttribute('class', 'mtp-link-arrow')
  marker.appendChild(tip)
  defs.appendChild(marker)
  svg.appendChild(defs)

  const cols = visibleCols(root)
  const boxes = cols.map(col => rel(stageBox, col))
  // Stacked when the second column sits below the first: links then run in side lanes.
  const stacked = boxes.length > 1 && boxes[1].top >= boxes[0].bottom - 1
  const pathEl = (d: string, cls: string, arrow = false) => {
    const path = document.createElementNS(ns, 'path')
    path.setAttribute('d', d)
    path.setAttribute('class', cls)
    path.setAttribute('vector-effect', 'non-scaling-stroke')
    if (arrow) path.setAttribute('marker-end', `url(#${markerId})`)
    svg.appendChild(path)
  }
  const labelEl = (x: number, y: number, rotate = false) => {
    const label = document.createElementNS(ns, 'text')
    label.setAttribute('x', String(x))
    label.setAttribute('y', String(y))
    label.setAttribute('text-anchor', 'middle')
    label.setAttribute('class', 'mtp-share-label')
    if (rotate) label.setAttribute('transform', `rotate(-90 ${x} ${y})`)
    label.textContent = 'shared'
    svg.appendChild(label)
  }

  for (let i = 0; i < cols.length - 1; i += 1) {
    const from = cols[i].querySelector('[data-mtp-out-h]')
    const to = cols[i + 1].querySelector('[data-mtp-in-h]')
    if (!from || !to) continue
    const a = rel(stageBox, from)
    const b = rel(stageBox, to)
    if (stacked) {
      // Left lane: out of the transformer's left edge, down, into the RMSNorm's left edge.
      const lane = boxes[i].left - 9
      pathEl(`M ${a.left} ${a.midY} H ${lane} V ${b.midY} H ${b.left}`, 'mtp-link-causal', true)
    } else {
      // The elbow runs in the gutter, clear of boxes and "shared" links.
      const gutterX = boxes[i].right + (boxes[i + 1].left - boxes[i].right) * 0.7
      pathEl(`M ${a.right} ${a.midY} H ${gutterX} V ${b.midY} H ${b.left}`, 'mtp-link-causal', true)
    }
  }

  const paintShared = (key: 'emb' | 'head', lane: number) => {
    const nodes = cols
      .map(col => col.querySelector(`[data-mtp-shared="${key}"]`))
      .filter((el): el is Element => Boolean(el))
    for (let i = 0; i < nodes.length - 1; i += 1) {
      const a = rel(stageBox, nodes[i])
      const b = rel(stageBox, nodes[i + 1])
      if (stacked) {
        const x = Math.max(a.right, b.right) + lane
        pathEl(`M ${a.right} ${a.midY} H ${x} V ${b.midY} H ${b.right}`, 'mtp-link-share')
      } else {
        const y = a.midY
        pathEl(`M ${a.right} ${y} H ${b.left}`, 'mtp-link-share')
        labelEl((a.right + b.left) / 2, y - 4)
      }
    }
  }

  paintShared('head', 8)
  paintShared('emb', 16)
  if (stacked) {
    // One label per column gap, outside the two lanes, centred in the gap.
    for (let i = 0; i < cols.length - 1; i += 1) {
      labelEl(boxes[i].right + 25, (boxes[i].bottom + boxes[i + 1].top) / 2, true)
    }
  }
}

const setup = (root: HTMLElement): (() => void) | null => {
  const out = (key: string) => root.querySelector<HTMLElement>(`[data-mtp-out="${key}"]`)
  let frame = 0

  const schedulePaint = () => {
    cancelAnimationFrame(frame)
    frame = requestAnimationFrame(() => paint(root))
  }

  const render = (depth: number) => {
    root.setAttribute('data-mtp-depth', String(depth))
    for (const col of root.querySelectorAll<HTMLElement>('[data-mtp-col]')) {
      const id = col.getAttribute('data-mtp-col')
      if (id === 'main') continue
      if (Number(id) > depth) col.setAttribute('data-mtp-off', 'true')
      else col.removeAttribute('data-mtp-off')
    }
    const chain = out('chain')
    if (chain) {
      const parts = ['main', ...Array.from({ length: depth }, (_, i) => `MTP ${i + 1}`)]
      chain.textContent = parts.join(' → ')
    }
    const count = out('count')
    if (count) count.textContent = String(depth)
    schedulePaint()
  }

  const onChange = (event: Event) => {
    const input = event.target as HTMLInputElement | null
    if (!input?.matches('[data-mtp-depth-input]')) return
    render(Number(input.value))
  }

  const onResize = () => schedulePaint()
  const ro = new ResizeObserver(onResize)
  const stage = root.querySelector('[data-mtp-stage]')
  if (stage) ro.observe(stage)
  ro.observe(root)
  const rail = root.querySelector('[data-mtp-rail]')
  if (rail) ro.observe(rail)

  root.addEventListener('change', onChange)
  window.addEventListener('resize', onResize)
  render(Number(root.getAttribute('data-mtp-depth') ?? '2'))

  return () => {
    cancelAnimationFrame(frame)
    root.removeEventListener('change', onChange)
    window.removeEventListener('resize', onResize)
    ro.disconnect()
  }
}

registerMdxContent(ROOT, root => {
  const cleanup = setup(root)
  if (cleanup) window.addCleanup(cleanup)
})
