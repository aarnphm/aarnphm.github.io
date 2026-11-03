import type { Runtime } from './runtime'
import {
  LABEL_X,
  layoutRail,
  RAIL_WIDTH,
  TRACK_X,
  truncate,
  type RailLayout,
} from '../../../util/latent/rail'
import { titleText } from '../../../util/latent/title'
import { yearOfDay } from '../../../util/latent/types'
import { goTo } from './reader'
import { measureRail, renderRail } from './search'

// ---- the rail as a control -----------------------------------------------------------------

function railLayoutOf(rt: Runtime): RailLayout {
  rt.railLayout ??= layoutRail(rt.track, rt.railHeight)
  return rt.railLayout
}

/** The document whose slot is nearest a y in rail viewBox units. */
function docNearY(rt: Runtime, y: number): number {
  const layout = railLayoutOf(rt)
  let best = 0
  let bestDist = Infinity
  for (let i = 0; i < layout.y.length; i++) {
    const dist = Math.abs(layout.y[i] - y)
    if (dist < bestDist) {
      bestDist = dist
      best = i
    }
  }
  return best
}

export function bindRail(rt: Runtime): void {
  let stepping = false
  let wheelBudget = 0
  const observer = new ResizeObserver(() => {
    if (measureRail(rt)) renderRail(rt, rt.railItems)
  })
  observer.observe(rt.rail)
  rt.signal.addEventListener('abort', () => observer.disconnect())
  // a click on the track opens the document nearest that point in time
  rt.rail.addEventListener(
    'click',
    event => {
      const target = event.target as Element | null
      if (!target || target.closest('text.lat-label')) return
      const svg = target.closest<SVGSVGElement>('svg.lat-track')
      if (!svg) return
      const ctm = svg.getScreenCTM()
      if (!ctm) return
      const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(ctm.inverse())
      if (point.y < 0 || point.y > rt.railHeight) return
      event.preventDefault()
      const d = docNearY(rt, point.y)
      if (d !== rt.open) void goTo(rt, rt.docs[d].r0)
    },
    { signal: rt.signal },
  )
  bindRailHover(rt, () => stepping)
  // the wheel over the rail steps through documents in time order instead of scrolling the page
  rt.rail.addEventListener(
    'wheel',
    event => {
      event.preventDefault()
      if (stepping) return
      wheelBudget += event.deltaY
      if (Math.abs(wheelBudget) < 40) return
      const direction = Math.sign(wheelBudget)
      wheelBudget = 0
      const next = rt.open + direction
      if (next < 0 || next >= rt.docs.length) return
      stepping = true
      void goTo(rt, rt.docs[next].r0).finally(() => {
        stepping = false
      })
    },
    { passive: false, signal: rt.signal },
  )
}

const HOVER_DWELL_MS = 260

/**
 * Hovering the dated track scrubs it: a probe line names the document under the pointer, and
 * resting on one for a beat opens it. The undated band is one row per kind, so it has no probe.
 */
function bindRailHover(rt: Runtime, busy: () => boolean): void {
  let hovered = -1
  let dwell: number | null = null
  let frame = 0
  let pending: { x: number; y: number } | null = null
  const svgOf = () => rt.rail.querySelector<SVGSVGElement>('svg.lat-track')
  const clear = () => {
    hovered = -1
    pending = null
    if (dwell !== null) window.clearTimeout(dwell)
    dwell = null
    svgOf()?.querySelector('.lat-probe')?.remove()
  }
  const probe = (d: number) => {
    const svg = svgOf()
    if (!svg) return
    const layout = railLayoutOf(rt)
    const y = layout.y[d]
    let group = svg.querySelector<SVGGElement>('g.lat-probe')
    if (!group) {
      group = document.createElementNS('http://www.w3.org/2000/svg', 'g')
      group.setAttribute('class', 'lat-probe')
      group.innerHTML = '<line></line><text></text>'
      svg.append(group)
    }
    const line = group.querySelector('line')!
    line.setAttribute('x1', String(TRACK_X - 6))
    line.setAttribute('x2', String(RAIL_WIDTH))
    line.setAttribute('y1', y.toFixed(1))
    line.setAttribute('y2', y.toFixed(1))
    const text = group.querySelector('text')!
    const year = yearOfDay(rt.track[d].day)
    text.setAttribute('x', String(LABEL_X))
    // the probe reads above the line near the bottom of the track, below it elsewhere
    text.setAttribute('y', (y > rt.railHeight * 0.8 ? y - 4 : y + 10).toFixed(1))
    text.textContent = `${truncate(titleText(rt.docs[d]?.t ?? ''), 24)}${year ? ` · ${year}` : ''}`
  }
  const settle = () => {
    frame = 0
    const at = pending
    pending = null
    const svg = svgOf()
    if (!at || !svg) return
    const ctm = svg.getScreenCTM()
    if (!ctm) return
    const point = new DOMPoint(at.x, at.y).matrixTransform(ctm.inverse())
    const layout = railLayoutOf(rt)
    const floor = layout.datedCount < rt.track.length ? layout.undatedTop - 8 : rt.railHeight
    if (point.y < 0 || point.y > floor) {
      clear()
      return
    }
    const d = docNearY(rt, point.y)
    if (rt.track[d].day === 0) {
      clear()
      return
    }
    // a rail re-render (the dwell opening a document) drops the probe; draw it again in place
    if (d === hovered && svg.querySelector('.lat-probe')) return
    const same = d === hovered
    hovered = d
    probe(d)
    if (same) return
    if (dwell !== null) window.clearTimeout(dwell)
    dwell = window.setTimeout(() => {
      dwell = null
      if (hovered !== d || d === rt.open || busy()) return
      void goTo(rt, rt.docs[d].r0)
    }, HOVER_DWELL_MS)
  }
  rt.rail.addEventListener(
    'pointermove',
    event => {
      if (event.pointerType === 'touch') return
      const target = event.target as Element | null
      if (!target?.closest('svg.lat-track') || target.closest('text.lat-label')) {
        clear()
        return
      }
      pending = { x: event.clientX, y: event.clientY }
      if (!frame) frame = requestAnimationFrame(settle)
    },
    { signal: rt.signal },
  )
  rt.rail.addEventListener('pointerleave', clear, { signal: rt.signal })
  rt.signal.addEventListener('abort', clear)
}
