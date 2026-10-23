import { registerMdxContent } from './mdx-content'

const DCH_ROOT_SELECTOR = '[data-data-chart]'

// The component precomputes every state; this script only selects one.
const setupPositions = (root: HTMLElement): (() => void) | null => {
  const svg = root.querySelector<SVGSVGElement>('[data-dch-svg]')
  const listbox = root.querySelector<SVGGElement>('[data-dch-listbox]')
  if (!svg || !listbox) return null
  const options = Array.from(listbox.querySelectorAll<SVGRectElement>('[data-dch-pos]'))
  const bands = Array.from(root.querySelectorAll<SVGGElement>('[data-dch-band]'))
  const states = Array.from(root.querySelectorAll<HTMLElement>('[data-dch-state]'))
  const centers = options.map(o => Number(o.getAttribute('data-dch-cx')))
  let active = Math.max(
    0,
    options.findIndex(o => o.getAttribute('aria-selected') === 'true'),
  )

  const select = (i: number, focus: boolean) => {
    const next = Math.min(options.length - 1, Math.max(0, i))
    if (next !== active) {
      options.forEach((o, j) => {
        o.setAttribute('aria-selected', j === next ? 'true' : 'false')
        o.setAttribute('tabindex', j === next ? '0' : '-1')
      })
      bands.forEach((b, j) => b.toggleAttribute('data-dch-on', j === next))
      states.forEach((s, j) => (s.hidden = j !== next))
      active = next
    }
    if (focus) options[next].focus()
  }

  const onKey = (e: KeyboardEvent) => {
    const step: Record<string, number> = {
      ArrowRight: active + 1,
      ArrowDown: active + 1,
      ArrowLeft: active - 1,
      ArrowUp: active - 1,
      Home: 0,
      End: options.length - 1,
    }
    if (!(e.key in step)) return
    e.preventDefault()
    select(step[e.key], true)
  }

  const nearest = (clientX: number) => {
    const ctm = svg.getScreenCTM()
    if (!ctm) return active
    const x = new DOMPoint(clientX, 0).matrixTransform(ctm.inverse()).x
    let best = 0
    centers.forEach((c, j) => {
      if (Math.abs(c - x) < Math.abs(centers[best] - x)) best = j
    })
    return best
  }
  const onMove = (e: PointerEvent) => select(nearest(e.clientX), false)
  const onDown = (e: PointerEvent) => {
    if (e.target instanceof SVGRectElement && e.target.hasAttribute('data-dch-pos'))
      select(nearest(e.clientX), true)
  }

  listbox.addEventListener('keydown', onKey)
  svg.addEventListener('pointermove', onMove)
  svg.addEventListener('pointerdown', onDown)
  return () => {
    listbox.removeEventListener('keydown', onKey)
    svg.removeEventListener('pointermove', onMove)
    svg.removeEventListener('pointerdown', onDown)
  }
}

const setupProbe = (root: HTMLElement): (() => void) | null => {
  const raw = root.getAttribute('data-dch-probe')
  const svg = root.querySelector<SVGSVGElement>('[data-dch-svg]')
  const input = root.querySelector<HTMLInputElement>('[data-dch-probe-input]')
  const hit = root.querySelector<SVGRectElement>('[data-dch-probe-hit]')
  const dot = root.querySelector<SVGCircleElement>('[data-dch-dot]')
  const drop = root.querySelector<SVGPathElement>('[data-dch-drop]')
  if (!raw || !svg || !input || !hit || !dot || !drop) return null
  const points = JSON.parse(raw) as [number, number][]
  const valuetext = JSON.parse(input.getAttribute('data-dch-valuetext') ?? '[]') as string[]
  const states = Array.from(root.querySelectorAll<HTMLElement>('[data-dch-probe-state]'))
  const x0 = Number(hit.getAttribute('x'))
  const y0 = Number(hit.getAttribute('y')) + Number(hit.getAttribute('height'))

  const apply = () => {
    const i = Number(input.value)
    const p = points[i]
    if (!p) return
    dot.setAttribute('cx', String(p[0]))
    dot.setAttribute('cy', String(p[1]))
    drop.setAttribute('d', `M${x0} ${p[1]}H${p[0]}V${y0}`)
    states.forEach((s, j) => (s.hidden = j !== i))
    if (valuetext[i]) input.setAttribute('aria-valuetext', valuetext[i])
  }

  const fromPointer = (e: PointerEvent) => {
    const ctm = svg.getScreenCTM()
    if (!ctm) return
    const x = new DOMPoint(e.clientX, 0).matrixTransform(ctm.inverse()).x
    let best = 0
    points.forEach((p, j) => {
      if (Math.abs(p[0] - x) < Math.abs(points[best][0] - x)) best = j
    })
    if (String(best) !== input.value) {
      input.value = String(best)
      apply()
    }
  }
  const onDown = (e: PointerEvent) => {
    hit.setPointerCapture(e.pointerId)
    root.setAttribute('data-dch-dragging', '')
    fromPointer(e)
  }
  const onMove = (e: PointerEvent) => {
    if (hit.hasPointerCapture(e.pointerId)) fromPointer(e)
  }
  const onUp = (e: PointerEvent) => {
    if (hit.hasPointerCapture(e.pointerId)) hit.releasePointerCapture(e.pointerId)
    root.removeAttribute('data-dch-dragging')
  }

  input.addEventListener('input', apply)
  hit.addEventListener('pointerdown', onDown)
  hit.addEventListener('pointermove', onMove)
  hit.addEventListener('pointerup', onUp)
  hit.addEventListener('pointercancel', onUp)
  apply()
  return () => {
    input.removeEventListener('input', apply)
    hit.removeEventListener('pointerdown', onDown)
    hit.removeEventListener('pointermove', onMove)
    hit.removeEventListener('pointerup', onUp)
    hit.removeEventListener('pointercancel', onUp)
  }
}

// The SVG scales with its frame. CSS multiplies type and marker sizes by user units per rendered
// pixel, so axis titles stay at the meta size inline and in the Zoomable view.
const setupScale = (root: HTMLElement): (() => void) | null => {
  const svg = root.querySelector<SVGSVGElement>('[data-dch-svg]')
  if (!svg) return null
  // A hover band narrower than 24 rendered px grows symmetrically about its point. Bands overlap,
  // and pointer selection picks the nearest point by centre, so the overlap changes no choice.
  const hits = Array.from(svg.querySelectorAll<SVGRectElement>('[data-dch-pos]')).map(rect => ({
    rect,
    lo: Number(rect.getAttribute('x')),
    width: Number(rect.getAttribute('width')),
    cx: Number(rect.getAttribute('data-dch-cx')),
  }))
  const observer = new ResizeObserver(entries => {
    const width = entries[entries.length - 1].contentRect.width
    if (width <= 0) return
    const u = svg.viewBox.baseVal.width / width
    svg.style.setProperty('--dch-u', u.toFixed(4))
    for (const h of hits) {
      const wide = h.width >= 24 * u
      h.rect.setAttribute('x', String(wide ? h.lo : h.cx - 12 * u))
      h.rect.setAttribute('width', String(wide ? h.width : 24 * u))
    }
  })
  observer.observe(svg)
  return () => observer.disconnect()
}

registerMdxContent(DCH_ROOT_SELECTOR, root => {
  for (const cleanup of [setupScale(root), setupPositions(root), setupProbe(root)])
    if (cleanup) window.addCleanup?.(cleanup)
})
