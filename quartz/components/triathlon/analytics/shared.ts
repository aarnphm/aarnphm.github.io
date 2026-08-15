import type { Analytics } from '../../../plugins/stores/analytics'
import type { RaceLeg } from '../../../plugins/stores/analytics'
import type { ActivityKind } from '../../../plugins/stores/strava'
import type { Sport } from '../../../plugins/stores/strava'
import type { AxisXTick } from '../../../util/triathlon-card'
import type { TriathlonContext } from '../runtime/context'
import type { TriathlonFormatter } from '../runtime/formatter'
import { clock } from '../../../util/triathlon-card'
import { KM_TO_MI } from '../../../util/triathlon-card'
import { buildIcon } from '../activity/primitives'
import { el } from '../runtime/dom'
import { svg } from '../runtime/dom'
import { toggleTriUnit } from '../runtime/preferences'
import { fmtKm } from './panels/thresholds'

export const ANA_W = 100

export const ANA_H = 30

export const KG_PER_LB = 0.45359237

export const weightUnitLabel = (formatter: TriathlonFormatter): string =>
  formatter.presentation.distance === 'imperial' ? 'lb' : 'kg'

export const wConv = (formatter: TriathlonFormatter, kg: number): number =>
  formatter.presentation.distance === 'imperial' ? kg / KG_PER_LB : kg

export const wNum = (formatter: TriathlonFormatter, kg: number, kgDp = 1, lbDp = 0): string => {
  const digits = formatter.presentation.distance === 'imperial' ? lbDp : kgDp
  return formatter.number(wConv(formatter, kg), digits, digits)
}

export const wFmt = (formatter: TriathlonFormatter, kg: number, kgDp = 1, lbDp = 0): string =>
  `${wNum(formatter, kg, kgDp, lbDp)} ${weightUnitLabel(formatter)}`

export const wSigned = (formatter: TriathlonFormatter, kg: number, dp: number): string => {
  const v = wConv(formatter, kg)
  return `${v > 0 ? '+' : ''}${formatter.number(v, dp, dp)}`
}

export const pctFmt = (formatter: TriathlonFormatter, value: number, digits: number): string =>
  `${formatter.number(value, digits, digits)}${formatter.presentation.locale === 'fr' ? '\u00a0' : ''}%`

export const weightSwitch = (context: TriathlonContext): HTMLElement => {
  const g = el('div', 'tri-unit-switch', undefined, { role: 'group', 'aria-label': 'weight unit' })
  for (const u of ['kg', 'lb'] as const) {
    const on = (u === 'lb') === (context.presentation.distance === 'imperial')
    const opt = el('button', on ? 'tri-unit-opt tri-unit-opt--on' : 'tri-unit-opt', u, {
      type: 'button',
      'aria-pressed': String(on),
      'data-unit': u,
    })
    g.appendChild(opt)
  }
  return g
}

export const mountWeightSwitch = (root: HTMLElement, context: TriathlonContext): (() => void) => {
  const onClick = (event: MouseEvent): void => {
    const target = event.target
    if (!(target instanceof Element)) return
    const option = target.closest<HTMLButtonElement>('.tri-unit-opt[data-unit]')
    if (!option || !root.contains(option)) return
    const unit = option.dataset.unit
    if (unit !== 'kg' && unit !== 'lb') return
    if ((unit === 'lb') !== (context.presentation.distance === 'imperial'))
      toggleTriUnit(context.preferences)
  }
  root.addEventListener('click', onClick)
  return () => root.removeEventListener('click', onClick)
}

export const RACE_LABEL: Record<string, string> = {
  sprint: 'sprint',
  olympic: 'olympic',
  '70.3': '70.3',
  ironman: 'ironman',
}

export const clampN = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x))

export const polyD = (pts: [number, number][]): string =>
  pts.map(([x, y], i) => `${i ? 'L' : 'M'} ${x.toFixed(2)} ${y.toFixed(2)}`).join(' ')

export const signed = (n: number): string => (n > 0 ? `+${n}` : `${n}`)

export const signedFixed = (n: number, dp: number): string => `${n > 0 ? '+' : ''}${n.toFixed(dp)}`

export const hms = (sec: number): string => {
  const t = Math.max(0, Math.round(sec))
  const h = Math.floor(t / 3600)
  const m = Math.floor((t % 3600) / 60)
  const s = t % 60
  return h > 0
    ? `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`
    : `${m}:${s.toString().padStart(2, '0')}`
}

export const speedUnitLabel = (formatter: TriathlonFormatter): 'mph' | 'km/h' =>
  formatter.presentation.distance === 'imperial' ? 'mph' : 'km/h'

export const speedFromKmh = (formatter: TriathlonFormatter, kmh: number): number =>
  formatter.presentation.distance === 'imperial' ? kmh * KM_TO_MI : kmh

export const fmtSpeedKmh = (
  formatter: TriathlonFormatter,
  kmh: number,
  dp = 1,
  gap = ' ',
): string => `${speedFromKmh(formatter, kmh).toFixed(dp)}${gap}${speedUnitLabel(formatter)}`

export type RaceLegSplit = Pick<RaceLeg, 'sport' | 'legKm' | 'splitS'>

export const raceLegDistance = (formatter: TriathlonFormatter, leg: RaceLegSplit): string =>
  leg.sport === 'swim'
    ? `${Math.round(leg.legKm * 1000).toLocaleString('en-US')} m`
    : fmtKm(formatter.presentation, leg.legKm)

export const raceLegPace = (formatter: TriathlonFormatter, leg: RaceLegSplit): string => {
  if (leg.legKm <= 0 || leg.splitS <= 0) return '—'
  if (leg.sport === 'swim') return `${clock(leg.splitS / (leg.legKm * 10))} /100m`
  if (leg.sport === 'bike') {
    const kmh = leg.legKm / (leg.splitS / 3600)
    return fmtSpeedKmh(formatter, kmh)
  }
  const secKm = leg.splitS / leg.legKm
  return formatter.presentation.distance === 'imperial'
    ? `${clock(secKm / KM_TO_MI)} /mi`
    : `${clock(secKm)} /km`
}

export const raceLegTip = (formatter: TriathlonFormatter, leg: RaceLegSplit): string =>
  `${formatter.text(leg.sport)} · ${hms(leg.splitS)} · ${raceLegDistance(formatter, leg)} · ${raceLegPace(formatter, leg)}`

export const markGloss = (e: HTMLElement, key: string): HTMLElement => {
  e.dataset.gloss = key
  e.tabIndex = 0
  return e
}

export const markGlossDefinition = (e: HTMLElement, definition: string): HTMLElement => {
  e.dataset.gloss = ''
  e.dataset.glossDef = definition
  e.tabIndex = 0
  return e
}

export const anaTitle = (
  formatter: TriathlonFormatter,
  text: string,
  key?: string,
): HTMLElement => {
  const e = el('div', 'tri-ana-block-title', formatter.text(text))
  if (key) markGloss(e, key)
  return e
}

export const bySport = <T extends { sport: Sport }>(arr: T[], sport: Sport): T | undefined =>
  arr.find(x => x.sport === sport)

export const thLabel = (
  formatter: TriathlonFormatter,
  th: { paceLabel: string; unit: string; vThr?: number },
): string => {
  if (th.unit === 'km/h' && th.vThr != null) return fmtSpeedKmh(formatter, th.vThr * 3.6, 0)
  if (th.unit === 's/km' && th.vThr != null && formatter.presentation.distance === 'imperial')
    return `${clock(1609.344 / th.vThr)} /mi`
  return th.unit === 'km/h' ? `${th.paceLabel} km/h` : `${th.paceLabel}${th.unit.slice(1)}`
}

export const buildIconLeg = (formatter: TriathlonFormatter, sport: ActivityKind): HTMLElement => {
  const wrap = el('span', `tri-ana-ico tri-leg-${sport}`)
  wrap.appendChild(buildIcon(formatter.presentation, sport))
  return wrap
}

export const trendDir = (invert: boolean, slope: number | null): number => {
  if (slope == null || slope === 0) return 0
  return (invert ? slope < 0 : slope > 0) ? 1 : -1
}

export type TrendGlyphDirection = 'up' | 'down' | 'flat'

export const buildTrendGlyph = (
  direction: TrendGlyphDirection,
  label: string,
  className: string,
): SVGElement => {
  const glyph = svg('svg', {
    class: `tri-trend-glyph tri-dir-${direction} ${className}`,
    viewBox: '0 0 8 8',
    role: 'img',
    'aria-label': label,
  })
  glyph.appendChild(
    svg('path', {
      d:
        direction === 'up'
          ? 'M4 1.25 7 6.75H1Z'
          : direction === 'down'
            ? 'M1 1.25h6L4 6.75Z'
            : 'M1.5 1.5h5v5h-5Z',
    }),
  )
  return glyph
}

export const buildGauge = (data: Analytics, context: TriathlonContext): HTMLElement => {
  const text = (key: string): string => context.formatter.text(key)
  const block = el('div', 'tri-ana-gauge')
  block.appendChild(anaTitle(context.formatter, 'training load · injury risk', 'acwr'))
  const r = data.risk
  const chips = el('div', 'tri-gauge-chips')
  chips.append(
    markGloss(
      el('span', 'tri-ana-chip', `${text('ramp')} ${signed(Math.round((r.rampWeek || 0) * 100))}%`),
      'ramp',
    ),
    markGloss(
      el(
        'span',
        'tri-ana-chip',
        r.monotony != null
          ? `${text('monotony')} ${r.monotony.toFixed(2)}`
          : `${text('monotony')} —`,
      ),
      'monotony',
    ),
    markGloss(
      el(
        'span',
        'tri-ana-chip',
        r.strain != null ? `${text('strain')} ${Math.round(r.strain)}` : `${text('strain')} —`,
      ),
      'strain',
    ),
  )
  if (r.acwr == null) {
    block.appendChild(el('div', 'tri-ana-empty', text('building base — ACWR needs ~4 weeks')))
    block.appendChild(chips)
    return block
  }
  const W = 100
  const H = 12
  const lo = 0.5
  const hi = 1.8
  const xf = (v: number): number => ((clampN(v, lo, hi) - lo) / (hi - lo)) * W
  const s = svg('svg', {
    class: 'tri-gauge-svg',
    viewBox: `0 0 ${W} ${H}`,
    preserveAspectRatio: 'none',
  })
  const zone = (a: number, b: number, cls: string): void => {
    s.appendChild(
      svg('rect', {
        x: xf(a),
        y: 4,
        width: xf(b) - xf(a),
        height: 4,
        class: `tri-gauge-zone ${cls}`,
      }),
    )
  }
  zone(lo, 0.8, 'tri-acwr-z-under')
  zone(0.8, 1.3, 'tri-acwr-z-sweet')
  zone(1.3, 1.5, 'tri-acwr-z-caution')
  zone(1.5, hi, 'tri-acwr-z-high')
  const track = el('div', 'tri-gauge-track')
  const needle = el('span', 'tri-gauge-needle')
  needle.style.left = `${clampN(xf(r.acwr), 1.5, 98.5)}%`
  track.append(s, needle)
  block.appendChild(track)
  const scale = el('div', 'tri-gauge-scale')
  for (const v of [0.8, 1.3, 1.5]) {
    const tick = el('span', 'tri-gauge-tick', v.toFixed(1))
    tick.style.left = `${xf(v).toFixed(2)}%`
    scale.appendChild(tick)
  }
  block.appendChild(scale)
  const val = el('div', 'tri-gauge-val')
  val.append(
    el('span', 'tri-gauge-num', r.acwr.toFixed(2)),
    el('span', `tri-gauge-state tri-acwr-${r.acwrState}`, r.acwrState),
  )
  block.appendChild(val)
  block.appendChild(chips)
  return block
}

export const PMC_H = 82

export const PMC_TOP = 4

export const PMC_BOT = 52

export const PMC_TSB_ZERO = 32

export const PMC_TSB_HALF = PMC_BOT - PMC_TSB_ZERO - 2

export const PMC_BAR_TOP = 61

export const PMC_BAR_BOT = 80

export const niceUp = (v: number): number => {
  const step = v <= 20 ? 5 : v <= 60 ? 10 : v <= 200 ? 20 : 50
  return Math.max(step, Math.ceil(v / step) * step)
}

export const monthTicks = (
  formatter: TriathlonFormatter,
  dates: string[],
  xPct: (i: number) => number,
): AxisXTick[] => {
  const out: AxisXTick[] = []
  const seen = new Set<string>()
  for (let i = 0; i < dates.length; i++) {
    const mo = dates[i].slice(0, 7)
    if (seen.has(mo)) continue
    seen.add(mo)
    out.push({
      label: new Date(`${dates[i]}T00:00:00Z`).toLocaleDateString(
        formatter.presentation.locale === 'fr' ? 'fr-CA' : 'en-US',
        { month: 'short', timeZone: 'UTC' },
      ),
      pct: xPct(i),
      cls: i === 0 ? 'tri-cax-xt--first' : undefined,
    })
  }
  return out
}

export const PMC_PROJ_DAYS = 14

export const K42 = 1 - Math.exp(-1 / 42)

export const K7 = 1 - Math.exp(-1 / 7)

export function normCdf(z: number): number {
  const t = 1 / (1 + 0.2316419 * Math.abs(z))
  const d = 0.3989423 * Math.exp((-z * z) / 2)
  const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))
  return z >= 0 ? 1 - p : p
}
