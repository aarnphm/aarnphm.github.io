import { h, render } from 'preact'
import type { HighParkLap } from '../../../util/high-park-loops'
import { highParkUnits, LapView, type LapSport } from '../../pages/triathlon-panels/high-park-laps'
import { rafEvent } from '../analytics/scrub-primitives'
import { clampN } from '../analytics/shared'
import { el } from '../runtime/dom'
import { setupLapSelect } from './high-park/lap-select'

type N = number | null
type Profile = [w: N[], hr: N[], cad: N[], spd: N[]]
/**
 * One lap as the page ships it: seconds, km, watts, bpm, rpm, torque N·m, torque effectiveness
 * L/R %, pedal smoothness L/R %, left balance %, breaths/min, core °C, shifts, climb m, and the
 * in-lap profile (null when unrecorded).
 */
type Lap = [number, number, N, N, N, N, N, N, N, N, N, N, N, N, N, Profile | null]
/**
 * One sport's rides newest first, as its log lists them, each with its laps, mean lap altitude
 * and activity link (for the drop-down, which renders on first open); points are
 * [ride, lap, x, y] in lap-clock units.
 */
export interface HighParkHoverData {
  sport?: LapSport
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
  if (!panel) return null
  const sports = new Map<LapSport, HighParkHoverData>()
  for (const source of panel.querySelectorAll('script.tri-hp-data'))
    try {
      const parsed = JSON.parse(source.textContent ?? '') as HighParkHoverData
      sports.set(parsed.sport ?? 'bike', parsed)
    } catch {
      continue
    }
  // The lap clock plots the cycling laps.
  const data = sports.get('bike') ?? { rides: [], points: [] }
  if (sports.size === 0) return null
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
    if (nm != null) values.push(['torque', `${u.nm(nm).toFixed(1)} ${u.torque}`])
    if (teL != null || teR != null) values.push(['torque eff. L/R', `${pair(teL, teR)} %`])
    if (psL != null || psR != null) values.push(['smoothness L/R', `${pair(psL, psR)} %`])
    if (balL != null) values.push(['balance L/R', `${pair(balL, 100 - balL)} %`])
    if (resp != null) values.push(['breathing', `${resp} br/min`])
    if (core != null) values.push(['core temp', `${u.celsius(core).toFixed(2)} ${u.temp}`])
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

  const hide = () => {
    if (!shown) return
    shown = ''
    tip.classList.remove('tri-gloss--on')
    mark(null)
  }

  const onClock = (event: PointerEvent) => {
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
  const onMove = rafEvent((event: PointerEvent) => {
    const target = event.target as Element | null
    if (clockSvg && target && clockSvg.contains(target)) onClock(event)
    else hide()
  })
  const onLeave = () => {
    onMove.cancel()
    hide()
  }

  // Drop-downs render on first open; `toggle` does not bubble, so listen in the capture phase.
  const lapSelect = setupLapSelect(panel)
  const slots = new Map<HTMLElement, { sport: LapSport; ride: number }>()
  const draw = (slot: HTMLElement, sport: LapSport, ride: number) => {
    const entry = sports.get(sport)?.rides[ride]
    if (!entry) return false
    const laps = entry[2].map(lapOf)
    const imperial = unitControl.imperial()
    render(
      h(LapView, {
        ride: { lapDetail: laps, altitude: entry[3], href: entry[4] },
        index: ride,
        imperial,
        sport,
      }),
      slot,
    )
    lapSelect.bind(slot, { laps, altitude: entry[3], sport }, highParkUnits(imperial))
    return true
  }
  const onToggle = (event: Event) => {
    const details = event.target as HTMLDetailsElement | null
    if (!details?.open) return
    const slot = details.querySelector<HTMLElement>(':scope > .tri-hp-lapslot')
    if (!slot || slots.has(slot)) return
    const sport: LapSport = slot.dataset.sport === 'run' ? 'run' : 'bike'
    const ride = Number(slot.dataset.ride)
    if (draw(slot, sport, ride)) slots.set(slot, { sport, ride })
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
    hide()
    for (const [slot, { sport, ride }] of slots) draw(slot, sport, ride)
  }
  const onUnitClick = () => unitControl.toggle()

  panel.addEventListener('pointermove', onMove.onEvent)
  panel.addEventListener('pointerleave', onLeave)
  panel.addEventListener('scroll', hide, { passive: true })
  panel.addEventListener('toggle', onToggle, true)
  unitButton?.addEventListener('click', onUnitClick)
  window.addEventListener('tri:unit', syncUnits)
  syncUnits()
  // A drop-down left open (a back/forward restore) renders now.
  for (const details of panel.querySelectorAll<HTMLDetailsElement>('.tri-hp-log details[open]'))
    onToggle({ target: details } as unknown as Event)
  return () => {
    onMove.cancel()
    panel.removeEventListener('pointermove', onMove.onEvent)
    panel.removeEventListener('pointerleave', onLeave)
    panel.removeEventListener('scroll', hide)
    panel.removeEventListener('toggle', onToggle, true)
    unitButton?.removeEventListener('click', onUnitClick)
    window.removeEventListener('tri:unit', syncUnits)
    lapSelect.destroy()
    for (const slot of slots.keys()) render(null, slot)
    tip.remove()
  }
}
