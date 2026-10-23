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
  marker.setAttribute('viewBox', '0 0 10 10')
  marker.setAttribute('refX', '9')
  marker.setAttribute('refY', '5')
  marker.setAttribute('markerWidth', '6')
  marker.setAttribute('markerHeight', '6')
  marker.setAttribute('orient', 'auto')
  const tip = document.createElementNS(ns, 'path')
  tip.setAttribute('d', 'M0,0 L10,5 L0,10 z')
  tip.setAttribute('class', 'mtp-link-arrow')
  marker.appendChild(tip)
  defs.appendChild(marker)
  svg.appendChild(defs)

  const cols = visibleCols(root)

  for (let i = 0; i < cols.length - 1; i += 1) {
    const from = cols[i].querySelector('[data-mtp-out-h]')
    const to = cols[i + 1].querySelector('[data-mtp-in-h]')
    if (!from || !to) continue
    const a = rel(stageBox, from)
    const b = rel(stageBox, to)
    const x1 = a.right
    const y1 = a.midY
    const x2 = b.midX
    const y2 = b.bottom
    const midX = a.right + Math.max(10, (b.left - a.right) * 0.45)
    const dropY = Math.max(y1, b.bottom + 8)
    const path = document.createElementNS(ns, 'path')
    path.setAttribute('d', `M ${x1} ${y1} H ${midX} V ${dropY} H ${x2} V ${y2}`)
    path.setAttribute('class', 'mtp-link-causal')
    path.setAttribute('marker-end', `url(#${markerId})`)
    svg.appendChild(path)
  }

  const paintShared = (key: 'emb' | 'head') => {
    const nodes = cols
      .map(col => col.querySelector(`[data-mtp-shared="${key}"]`))
      .filter((el): el is Element => Boolean(el))
    if (nodes.length < 2) return
    for (let i = 0; i < nodes.length - 1; i += 1) {
      const a = rel(stageBox, nodes[i])
      const b = rel(stageBox, nodes[i + 1])
      const y = a.midY
      const x1 = a.right
      const x2 = b.left
      const line = document.createElementNS(ns, 'line')
      line.setAttribute('x1', String(x1))
      line.setAttribute('y1', String(y))
      line.setAttribute('x2', String(x2))
      line.setAttribute('y2', String(y))
      line.setAttribute('class', 'mtp-link-share')
      svg.appendChild(line)
      const label = document.createElementNS(ns, 'text')
      label.setAttribute('x', String((x1 + x2) / 2))
      label.setAttribute('y', String(y - 4))
      label.setAttribute('text-anchor', 'middle')
      label.setAttribute('class', 'mtp-share-label')
      label.textContent = 'shared'
      svg.appendChild(label)
    }
  }

  paintShared('emb')
  paintShared('head')
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
