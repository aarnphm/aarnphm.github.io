import { h, render } from 'preact'
import type { HighParkLap } from '../../../util/high-park-loops'
import { LAP_BINS } from '../../../util/high-park-lap-metrics'
import {
  highParkUnits,
  LAP_CHART,
  LapView,
  PROFILE,
  profileX,
  profileY,
  type ProfileKey,
} from '../../pages/triathlon-panels/high-park-laps'
import { rafEvent } from '../analytics/scrub-primitives'
import { clampN } from '../analytics/shared'
import { el } from '../runtime/dom'

type N = number | null
type Profile = [w: N[], hr: N[], cad: N[], spd: N[]]
/**
 * One lap as the page ships it: seconds, km, watts, bpm, rpm, torque N·m, torque effectiveness
 * L/R %, pedal smoothness L/R %, left balance %, breaths/min, core °C, shifts, climb m, and the
 * in-lap profile (null when unrecorded).
 */
type Lap = [number, number, N, N, N, N, N, N, N, N, N, N, N, N, N, Profile | null]
/**
 * Rides newest first, as the log lists them, each with its laps, mean lap altitude and activity
 * link (for the drop-down, which renders on first open); points are [ride, lap, x, y] in
 * lap-clock units.
 */
export interface HighParkHoverData {
  rides: [date: string, name: string, laps: Lap[], altitude: number[] | null, href: string][]
  points: [number, number, number, number][]
}
export interface HighParkUnitControl {
  imperial: () => boolean
  toggle: () => void
}

const MONTHS = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
const clock = (seconds: number): string =>
  `${Math.floor(seconds / 60)}:${String(Math.round(seconds) % 60).padStart(2, '0')}`
// Nearest dot within this many lap-clock units (the chart is 640 wide) counts as hovered.
const REACH = 16
const pair = (left: N, right: N) => `${left ?? '–'}/${right ?? '–'}`
const PROFILE_INDEX: Record<ProfileKey, number> = { w: 0, hr: 1, cad: 2, spd: 3 }
const lapOf = ([
  s,
  km,
  w,
  hr,
  cad,
  nm,
  teL,
  teR,
  psL,
  psR,
  balL,
  resp,
  core,
  shifts,
  gain,
  p,
]: Lap): HighParkLap => ({
  s,
  km,
  w,
  hr,
  cad,
  nm,
  teL,
  teR,
  psL,
  psR,
  balL,
  resp,
  core,
  shifts,
  gain,
  profile: p && { w: p[0], hr: p[1], cad: p[2], spd: p[3] },
})

export const setupHighParkHover = (
  root: HTMLElement,
  unitControl: HighParkUnitControl,
): (() => void) | null => {
  const panel = root.querySelector<HTMLElement>('.tri-hp')
  const source = panel?.querySelector('script.tri-hp-data')
  if (!panel || !source?.textContent) return null
  let data: HighParkHoverData
  try {
    data = JSON.parse(source.textContent) as HighParkHoverData
  } catch {
    return null
  }
  const units = () => highParkUnits(unitControl.imperial())
  const clockSvg = panel.querySelector<SVGSVGElement>('.tri-hp-clock svg')
  const marker = clockSvg?.querySelector<SVGRectElement>('.tri-hp-hover') ?? null
  const tip = el('div', 'tri-gloss tri-hp-tip')
  tip.setAttribute('role', 'tooltip')
  const head = el('span', 'tri-gloss-h')
  const name = el('span', 'tri-hp-tip-name')
  const rows = el('dl', 'tri-hp-tip-rows')
  tip.append(head, name, rows)
  root.appendChild(tip)
  let shown = ''

  const point = (ride: number, lap: number) =>
    data.points.find(([r, l]) => r === ride && l === lap) ?? null
  const fill = (ride: number, lap: number) => {
    const entry = data.rides[ride]
    const detail = entry?.[2][lap]
    if (!entry || !detail) return false
    const u = units()
    const key = `${ride}:${lap}:${u.imperial}`
    if (key === shown) return true
    shown = key
    const [date, title, laps] = entry
    const [s, km, w, hr, cad, nm, teL, teR, psL, psR, balL, resp, core] = detail
    head.textContent = `${date.slice(8, 10)} ${MONTHS[Number(date.slice(5, 7)) - 1] ?? ''} · lap ${lap + 1} of ${laps.length}`
    name.textContent = title
    const values: [string, string][] = [['time', clock(s)]]
    if (km > 0 && s > 0) values.push(['speed', `${u.kmh((km / s) * 3600).toFixed(1)} ${u.speed}`])
    if (w != null) values.push(['power', `${w} W`])
    if (hr != null) values.push(['heart rate', `${hr} bpm`])
    if (cad != null) values.push(['cadence', `${cad} rpm`])
    if (nm != null) values.push(['torque', `${nm.toFixed(1)} N·m`])
    if (teL != null || teR != null) values.push(['torque eff. L/R', `${pair(teL, teR)} %`])
    if (psL != null || psR != null) values.push(['smoothness L/R', `${pair(psL, psR)} %`])
    if (balL != null) values.push(['balance L/R', `${pair(balL, 100 - balL)} %`])
    if (resp != null) values.push(['breathing', `${resp} br/min`])
    if (core != null) values.push(['core temp', `${core.toFixed(2)} °C`])
    rows.replaceChildren(...values.flatMap(([k, v]) => [el('dt', '', k), el('dd', '', v)]))
    return true
  }
  const place = (x: number, y: number) => {
    const { width, height } = tip.getBoundingClientRect()
    let left = x + 14
    if (left + width > window.innerWidth - 8) left = x - 14 - width
    let top = y + 14
    if (top + height > window.innerHeight - 8) top = y - 14 - height
    tip.style.left = `${clampN(left, 8, Math.max(8, window.innerWidth - 8 - width))}px`
    tip.style.top = `${clampN(top, 8, Math.max(8, window.innerHeight - 8 - height))}px`
  }
  const mark = (hit: [number, number, number, number] | null) => {
    if (!marker) return
    marker.classList.toggle('is-on', hit != null)
    if (!hit) return
    marker.setAttribute('x', String(hit[2] - 4.5))
    marker.setAttribute('y', String(hit[3] - 4.5))
  }
  const show = (ride: number, lap: number, event: PointerEvent) => {
    if (!fill(ride, lap)) return hide()
    tip.classList.add('tri-gloss--on')
    place(event.clientX, event.clientY)
    mark(point(ride, lap))
  }

  let hotView: HTMLElement | null = null
  let hotLap = -1
  let readout: HTMLElement | null = null
  // A clicked lap stays lit (charts, table row, readout, tooltip) until clicked again or Esc.
  let locked: { view: HTMLElement; lap: number } | null = null
  const release = () => {
    locked?.view.classList.remove('is-locked')
    locked = null
  }
  const cool = () => {
    if (readout) readout.textContent = readout.dataset.idle ?? ''
    readout = null
    if (!hotView) return
    for (const node of hotView.querySelectorAll('.is-hot')) node.classList.remove('is-hot')
    for (const path of hotView.querySelectorAll('.tri-hp-cursor, .tri-hp-marker'))
      path.setAttribute('d', '')
    hotView = null
    hotLap = -1
  }
  // One lap lit across the drop-down: its line in each profile, its table row, a cursor per chart.
  const heat = (view: HTMLElement, lap: number) => {
    cool()
    hotView = view
    hotLap = lap
    const laps = Number(view.dataset.laps)
    for (const node of view.querySelectorAll(`[data-lap="${lap}"]`)) node.classList.add('is-hot')
    const { w, h, pad } = LAP_CHART
    const x = pad + (lap / Math.max(1, laps - 1)) * (w - 2 * pad)
    for (const cursor of view.querySelectorAll('.tri-hp-lapchart .tri-hp-cursor'))
      cursor.setAttribute('d', `M${x.toFixed(1)} 0V${h}`)
  }
  const clear = () => {
    shown = ''
    tip.classList.remove('tri-gloss--on')
    mark(null)
    cool()
  }
  const hide = () => {
    if (!locked) clear()
  }

  const onClock = (event: PointerEvent) => {
    cool()
    const matrix = clockSvg?.getScreenCTM()
    if (!matrix) return
    const p = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse())
    let best: [number, number, number, number] | null = null
    let bestD = REACH * REACH
    for (const candidate of data.points) {
      const d = (candidate[2] - p.x) ** 2 + (candidate[3] - p.y) ** 2
      if (d < bestD) [best, bestD] = [candidate, d]
    }
    if (best) show(best[0], best[1], event)
    else hide()
  }
  const rideOf = (view: HTMLElement) =>
    view.dataset.ride === undefined ? NaN : Number(view.dataset.ride)
  const onLapChart = (event: PointerEvent, chart: SVGSVGElement, view: HTMLElement) => {
    const { w, pad } = LAP_CHART
    const laps = Number(view.dataset.laps)
    const rect = chart.getBoundingClientRect()
    const at = ((event.clientX - rect.left) / rect.width) * w
    const lap = clampN(Math.round(((at - pad) / (w - 2 * pad)) * (laps - 1)), 0, laps - 1)
    heat(view, lap)
    const ride = rideOf(view)
    if (Number.isFinite(ride)) show(ride, lap, event)
  }
  // Per-lap profile: the bin under the cursor, and the lap whose line passes nearest the pointer.
  const onProfile = (event: PointerEvent, chart: SVGSVGElement, view: HTMLElement) => {
    const ride = rideOf(view)
    const figure = chart.closest<HTMLElement>('.tri-hp-profile')
    const key = figure?.dataset.metric as ProfileKey | undefined
    const laps = data.rides[ride]?.[2]
    if (!figure || !key || !laps) return
    const u = units()
    const scale = key === 'spd' ? u.kmh : (v: number) => v
    const lo = Number(chart.dataset.lo)
    const hi = Number(chart.dataset.hi)
    const rect = chart.getBoundingClientRect()
    const bin = clampN(
      Math.round(((event.clientX - rect.left) / rect.width) * (LAP_BINS - 1)),
      0,
      LAP_BINS - 1,
    )
    const py = ((event.clientY - rect.top) / rect.height) * PROFILE.h
    let lap = -1
    let nearest = Infinity
    let value = 0
    laps.forEach((entry, i) => {
      const raw = entry[15]?.[PROFILE_INDEX[key]]?.[bin]
      if (raw == null) return
      const y = profileY(scale(raw), lo, hi)
      if (Math.abs(y - py) < nearest) [lap, nearest, value] = [i, Math.abs(y - py), scale(raw)]
    })
    if (lap < 0) return
    heat(view, lap)
    const x = profileX(bin)
    chart.querySelector('.tri-hp-cursor')?.setAttribute('d', `M${x} 0V${PROFILE.h}`)
    chart.querySelector('.tri-hp-marker')?.setAttribute('d', `M${x} ${profileY(value, lo, hi)}h0`)
    readout = figure.querySelector<HTMLElement>('.tri-hp-readout')
    if (readout) {
      const lapKm = laps[lap][1]
      const at = u.km((bin / (LAP_BINS - 1)) * lapKm)
      const unit = readout.dataset.idle ?? ''
      readout.textContent = `lap ${lap + 1} · ${at.toFixed(2)} ${u.distance} · ${value.toFixed(key === 'spd' ? 1 : 0)} ${unit}`
    }
    show(ride, lap, event)
  }
  const route = (event: PointerEvent) => {
    const target = event.target as Element | null
    const view = target?.closest<HTMLElement>('.tri-hp-lapview') ?? null
    const chart = target?.closest<SVGSVGElement>('.tri-hp-lapchart svg')
    const profile = target?.closest<SVGSVGElement>('.tri-hp-profile svg')
    const row = target?.closest<HTMLElement>('.tri-hp-lapgrid tr[data-lap]')
    if (clockSvg && target && clockSvg.contains(target)) onClock(event)
    else if (view && chart) onLapChart(event, chart, view)
    else if (view && profile && Number.isFinite(rideOf(view))) onProfile(event, profile, view)
    else if (view && row) {
      if (shown) hide()
      heat(view, Number(row.dataset.lap))
    } else if (shown || hotView) hide()
  }
  const onMove = rafEvent((event: PointerEvent) => {
    if (!locked) route(event)
  })
  const onClick = (event: MouseEvent) => {
    const target = event.target as Element | null
    const view = target?.closest<HTMLElement>('.tri-hp-lapview')
    if (
      !view ||
      !target?.closest('.tri-hp-lapchart svg, .tri-hp-profile svg, .tri-hp-lapgrid tr[data-lap]')
    )
      return
    const previous = locked
    release()
    onMove.cancel()
    route(event as PointerEvent)
    if (hotView !== view || hotLap < 0) return
    if (previous?.view === view && previous.lap === hotLap) return clear()
    locked = { view, lap: hotLap }
    view.classList.add('is-locked')
  }
  const onScroll = () => {
    // The tooltip is fixed to the viewport; a pinned lap keeps its lines and readout.
    if (locked) tip.classList.remove('tri-gloss--on')
    else clear()
  }
  const onKey = (event: KeyboardEvent) => {
    if (event.key !== 'Escape') return
    release()
    clear()
  }

  // Drop-downs render on first open; `toggle` does not bubble, so listen in the capture phase.
  const slots = new Map<HTMLElement, number>()
  const draw = (slot: HTMLElement, ride: number) => {
    const entry = data.rides[ride]
    if (!entry) return
    render(
      h(LapView, {
        ride: { lapDetail: entry[2].map(lapOf), altitude: entry[3], href: entry[4] },
        index: ride,
        imperial: unitControl.imperial(),
      }),
      slot,
    )
  }
  const onToggle = (event: Event) => {
    const details = event.target as HTMLDetailsElement | null
    if (!details?.open) return
    const slot = details.querySelector<HTMLElement>(':scope > .tri-hp-lapslot')
    const ride = Number(slot?.dataset.ride)
    if (!slot || slots.has(slot) || !data.rides[ride]) return
    slots.set(slot, ride)
    draw(slot, ride)
  }

  // Units: server-rendered text and the map's scale bar swap in place; open drop-downs redraw.
  const unitButton = panel.querySelector<HTMLButtonElement>('.tri-hp-unit')
  const syncUnits = () => {
    const imperial = unitControl.imperial()
    if (unitButton) unitButton.textContent = imperial ? 'mph' : 'km/h'
    for (const node of panel.querySelectorAll<HTMLElement>('.tri-hp-u'))
      node.textContent = (imperial ? node.dataset.imperial : node.dataset.metric) ?? ''
    const bar = panel.querySelector<HTMLElement>('.tri-hp-scale-bar')
    if (bar) bar.style.width = (imperial ? bar.dataset.imperialW : bar.dataset.metricW) ?? ''
    release()
    clear()
    for (const [slot, ride] of slots) draw(slot, ride)
  }
  const onUnitClick = () => unitControl.toggle()

  panel.addEventListener('pointermove', onMove.onEvent)
  panel.addEventListener('pointerleave', hide)
  panel.addEventListener('scroll', onScroll, { passive: true })
  panel.addEventListener('click', onClick)
  panel.addEventListener('toggle', onToggle, true)
  unitButton?.addEventListener('click', onUnitClick)
  window.addEventListener('tri:unit', syncUnits)
  document.addEventListener('keydown', onKey)
  syncUnits()
  // A drop-down left open (a back/forward restore) renders now.
  for (const details of panel.querySelectorAll<HTMLDetailsElement>('.tri-hp-log details[open]'))
    onToggle({ target: details } as unknown as Event)
  return () => {
    onMove.cancel()
    panel.removeEventListener('pointermove', onMove.onEvent)
    panel.removeEventListener('pointerleave', hide)
    panel.removeEventListener('scroll', onScroll)
    panel.removeEventListener('click', onClick)
    panel.removeEventListener('toggle', onToggle, true)
    unitButton?.removeEventListener('click', onUnitClick)
    window.removeEventListener('tri:unit', syncUnits)
    document.removeEventListener('keydown', onKey)
    for (const slot of slots.keys()) render(null, slot)
    tip.remove()
  }
}
