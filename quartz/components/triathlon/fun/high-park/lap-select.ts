import { PICK_SLOTS, toggleLap } from '../../../../util/high-park-lap-compare'
import { LAP_BINS } from '../../../../util/high-park-lap-metrics'
import {
  LAP_CHART,
  lapChartX,
  PROFILE,
  profileLine,
  profileX,
  profileY,
  type HighParkUnits,
} from '../../../pages/triathlon-panels/high-park-laps'
import { rafEvent } from '../../analytics/scrub-primitives'
import { clampN } from '../../analytics/shared'
import { collectView, type LapRide, type ProfileNodes, type View, type Who } from './lap-nodes'

export type { LapRide } from './lap-nodes'

/**
 * Lap picking and the hover readout inside a ride's drop-down.
 *
 * A click on a lap (a lap-by-lap position, a profile line, a table row) picks it or drops it;
 * picks hold a colour slot each and the first pick is the reference. Hovering a profile reads
 * every pick at that point of the lap, on all four charts at once.
 *
 * The page carries hundreds of `:has()` rules, so a node insertion or a text write restyles the
 * whole document. Pointer moves therefore only write `Text.data`, inline style and `d` on nodes
 * the markup pooled, and only when the value changed. Attributes change on click alone.
 */
const setText = (node: Text, value: string) => {
  if (node.data !== value) node.data = value
}
const setPath = (path: Element, d: string) => {
  if (path.getAttribute('d') !== d) path.setAttribute('d', d)
}
const setStyle = (
  node: HTMLElement | SVGElement,
  property: 'display' | 'order' | 'left' | 'top',
  value: string,
) => {
  if (node.style[property] !== value) node.style[property] = value
}
/** Show a pooled mark at [x, y] percent of its plot box, or hide it. */
const place = (node: HTMLElement, at: readonly [number, number] | null) => {
  setStyle(node, 'display', at ? '' : 'none')
  if (!at) return
  setStyle(node, 'left', `${at[0]}%`)
  setStyle(node, 'top', `${at[1]}%`)
}
export const setupLapSelect = (panel: HTMLElement) => {
  const views = new Map<HTMLElement, View>()
  const byElement = new WeakMap<HTMLElement, View>()
  let hot: View | null = null

  const rowsOf = (view: View, profile: ProfileNodes) =>
    view.even ? profile.model.even : profile.model.rows
  // The hovered lap gets its own line and readout unless a pick or the fastest lap already has.
  const hoverShown = (view: View) =>
    view.hoverLap >= 0 &&
    !view.picks.some(pick => pick.lap === view.hoverLap) &&
    !(view.picks.length === 0 && view.hoverLap === view.best)

  const paintRead = (view: View) => {
    const { bin, units } = view
    const entries: { who: Who; lap: number }[] = view.picks.length
      ? view.picks.map(pick => ({ who: String(pick.slot) as Who, lap: pick.lap }))
      : [{ who: 'best', lap: view.best }]
    if (hoverShown(view)) entries.push({ who: 'hover', lap: view.hoverLap })
    const at =
      bin < 0
        ? 'lap avg'
        : `${units.km((bin / (LAP_BINS - 1)) * view.lapKm).toFixed(2)} ${units.distance}`
    for (const profile of view.profiles) {
      const { field, model } = profile
      const rows = rowsOf(view, profile)
      const whole = view.even ? model.restEven : model.rest
      const valueOf = (lap: number) => (bin < 0 ? whole[lap] : (rows[lap]?.[bin] ?? null))
      const base = view.even ? (bin < 0 ? model.restBase : (model.base?.mean[bin] ?? null)) : null
      // Against the baseline when it is on, else against the first pick (or the fastest lap).
      const reference = base ?? valueOf(entries[0].lap)
      setText(profile.at, at)
      const put = (who: Who, order: number, label: string, value: number | null, own: boolean) => {
        const nodes = profile.read.get(who)
        if (!nodes) return
        setStyle(nodes.el, 'display', '')
        setStyle(nodes.el, 'order', String(order))
        setText(nodes.lap, label)
        setText(nodes.value, value === null ? '–' : field.format(value))
        setText(
          nodes.delta,
          own || value === null || reference === null ? '' : field.delta(value, reference),
        )
        place(
          nodes.dot,
          bin < 0 || value === null
            ? null
            : [
                (profileX(bin) / PROFILE.w) * 100,
                (profileY(value, model.lo, model.hi) / PROFILE.h) * 100,
              ],
        )
      }
      const used = new Set<Who>()
      entries.forEach(({ who, lap }, i) => {
        used.add(who)
        put(who, i, String(lap + 1), valueOf(lap), base === null && i === 0)
      })
      if (base !== null) {
        used.add('base')
        put('base', entries.length, 'mean', base, true)
      }
      for (const [who, nodes] of profile.read)
        if (!used.has(who)) {
          setStyle(nodes.el, 'display', 'none')
          place(nodes.dot, null)
        }
    }
  }

  const paintHover = (view: View) => {
    const lap = view.hoverLap
    const cursor = lap < 0 ? '' : `M${lapChartX(lap, view.ride.laps.length)} 0V${LAP_CHART.h}`
    for (const chart of view.charts) {
      setPath(chart.cursor, cursor)
      setText(chart.range, lap < 0 ? chart.idle : chart.scale.read(lap))
    }
    setText(view.lapName, lap < 0 ? '' : `lap ${lap + 1} of ${view.ride.laps.length}`)
    const shown = hoverShown(view)
    for (const profile of view.profiles) {
      const { lo, hi } = profile.model
      setPath(profile.hoverLine, shown ? profileLine(rowsOf(view, profile)[lap] ?? [], lo, hi) : '')
      setPath(profile.cursor, view.bin < 0 ? '' : `M${profileX(view.bin)} 0V${PROFILE.h}`)
    }
    paintRead(view)
  }

  // Everything a click can change: chips, pins, pick lines, table rows, then the readout.
  const paintPicks = (view: View) => {
    const { picks } = view
    view.el.toggleAttribute('data-picked', picks.length > 0)
    view.el.toggleAttribute('data-even', view.even)
    view.mode?.setAttribute('aria-pressed', String(view.even))
    const bySlot = (slot: number) => picks.find(pick => pick.slot === slot)
    view.chips.forEach((chip, slot) => {
      const pick = bySlot(slot)
      setStyle(chip.el, 'display', pick ? '' : 'none')
      if (!pick) return
      setStyle(chip.el, 'order', String(picks.indexOf(pick)))
      setText(chip.lap, String(pick.lap + 1))
    })
    if (view.clear) setStyle(view.clear, 'display', picks.length ? '' : 'none')
    for (const lap of view.marked)
      if (!picks.some(pick => pick.lap === lap)) view.rows[lap]?.removeAttribute('data-who')
    view.marked = new Set(picks.map(pick => pick.lap))
    for (const pick of picks) view.rows[pick.lap]?.setAttribute('data-who', String(pick.slot))
    for (let slot = 0; slot < PICK_SLOTS; slot++) {
      const pick = bySlot(slot)
      for (const chart of view.charts) {
        const points = pick ? chart.scale.pointsAt(pick.lap) : []
        chart.pins[slot]?.forEach((pin, s) => place(pin, points[s] ?? null))
      }
      for (const profile of view.profiles) {
        const row = pick ? rowsOf(view, profile)[pick.lap] : null
        setPath(
          profile.pickLines[slot],
          row ? profileLine(row, profile.model.lo, profile.model.hi) : '',
        )
      }
    }
    paintHover(view)
  }

  const setHover = (view: View | null, lap = -1, bin = -1) => {
    if (hot && hot !== view) {
      hot.hoverLap = hot.bin = -1
      paintHover(hot)
    }
    hot = view
    if (!view || (view.hoverLap === lap && view.bin === bin)) return
    view.hoverLap = lap
    view.bin = bin
    paintHover(view)
  }

  const viewAt = (target: Element | null) => {
    const el = target?.closest<HTMLElement>('.tri-hp-lapview')
    return (el && byElement.get(el)) ?? null
  }
  // The lap (and, on a profile, the bin) a pointer event lands on. Reads layout, writes nothing.
  const locate = (event: MouseEvent, view: View): { lap: number; bin: number } | null => {
    const target = event.target as Element
    const count = view.ride.laps.length
    const row = target.closest<HTMLElement>('.tri-hp-lapgrid tr[data-lap]')
    if (row) return { lap: Number(row.dataset.lap), bin: -1 }
    const chart = target.closest<SVGSVGElement>('.tri-hp-lapchart svg')
    if (chart) {
      const { w, pad } = LAP_CHART
      const rect = chart.getBoundingClientRect()
      const at = ((event.clientX - rect.left) / rect.width) * w
      const lap = Math.round(((at - pad) / (w - 2 * pad)) * (count - 1))
      return { lap: clampN(lap, 0, count - 1), bin: -1 }
    }
    const svg = target.closest<SVGSVGElement>('.tri-hp-profile svg')
    const profile = svg && view.profiles.find(candidate => candidate.svg === svg)
    if (!svg || !profile) return null
    const rect = svg.getBoundingClientRect()
    const bin = clampN(
      Math.round(((event.clientX - rect.left) / rect.width) * (LAP_BINS - 1)),
      0,
      LAP_BINS - 1,
    )
    // The lap whose line passes nearest the pointer at that bin.
    const y = ((event.clientY - rect.top) / rect.height) * PROFILE.h
    let lap = -1
    let nearest = Infinity
    rowsOf(view, profile).forEach((values, i) => {
      const value = values[bin]
      if (value == null) return
      const distance = Math.abs(profileY(value, profile.model.lo, profile.model.hi) - y)
      if (distance < nearest) [lap, nearest] = [i, distance]
    })
    return lap < 0 ? null : { lap, bin }
  }

  const point = (event: MouseEvent) => {
    const view = viewAt(event.target as Element | null)
    const hit = view && locate(event, view)
    if (view && hit) setHover(view, hit.lap, hit.bin)
    else setHover(view)
  }
  const onMove = rafEvent(point)
  const onLeave = () => {
    onMove.cancel()
    setHover(null)
  }
  const onClick = (event: MouseEvent) => {
    const target = event.target as Element | null
    const view = viewAt(target)
    if (!view || !target) return
    const chip = target.closest<HTMLElement>('button.tri-hp-pick')
    if (chip) {
      const slot = Number(chip.dataset.who)
      view.picks = view.picks.filter(pick => pick.slot !== slot)
    } else if (target.closest('.tri-hp-clear')) view.picks = []
    else if (target.closest('.tri-hp-mode')) view.even = !view.even
    else {
      const hit = locate(event, view)
      if (!hit) return
      onMove.cancel()
      // A tap has no hover before it: the click also places the cursor.
      hot = view
      view.hoverLap = hit.lap
      view.bin = hit.bin
      view.picks = toggleLap(view.picks, hit.lap)
    }
    paintPicks(view)
  }
  const onKey = (event: KeyboardEvent) => {
    if (event.key !== 'Escape') return
    for (const view of views.values())
      if (view.picks.length) {
        view.picks = []
        paintPicks(view)
      }
  }

  panel.addEventListener('pointermove', onMove.onEvent)
  panel.addEventListener('pointerleave', onLeave)
  panel.addEventListener('click', onClick)
  document.addEventListener('keydown', onKey)

  return {
    /** After a drop-down renders (or re-renders in other units): find its nodes, keep its picks. */
    bind(slot: HTMLElement, ride: LapRide, units: HighParkUnits) {
      const el = slot.querySelector<HTMLElement>('.tri-hp-lapview')
      if (!el) return
      const previous = views.get(slot)
      const view = collectView(el, ride, units, previous)
      if (hot === previous) hot = null
      views.set(slot, view)
      byElement.set(el, view)
      paintPicks(view)
    },
    destroy() {
      onMove.cancel()
      panel.removeEventListener('pointermove', onMove.onEvent)
      panel.removeEventListener('pointerleave', onLeave)
      panel.removeEventListener('click', onClick)
      document.removeEventListener('keydown', onKey)
      views.clear()
    },
  }
}
