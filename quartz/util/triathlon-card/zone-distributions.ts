import type { StravaActivityDetail } from '../../plugins/stores/strava'
import type { DetailCtx, TriNodeFactory } from './construction-contracts'
import { SWIM_POWER_BIN_SIZE } from '../swim-power'
import { triathlonTraceName } from '../triathlon-trace-settings'
import { axisFrame, type AxisXTick } from './chart-geometry'
import { HR_ZONE_NAMES, POWER_ZONE_NAMES, zoneClock, zoneRange } from './metric-text'
import { swimDragPowerDefinition } from './swim-power-chart'

export const zoneDuo = <N>(f: TriNodeFactory<N>, a: N | null, b: N | null): N | null => {
  if (!a || !b) return a ?? b
  const duo = f.el('div', 'tri-zone-duo')
  f.add(duo, a, b)
  return duo
}

const zoneTable = <N>(
  f: TriNodeFactory<N>,
  title: string,
  times: number[],
  bounds: number[],
  names: string[],
  unit: string,
  caption: string,
): N => {
  const wrap = f.el('div', 'tri-zone', undefined, { 'data-tri-trace': triathlonTraceName(title) })
  f.add(wrap, f.el('div', 'tri-zone-title', title, { 'data-i18n': title }))
  const total = times.reduce((s, x) => s + x, 0) || 1
  let mx = 1
  for (const t of times) if (t > mx) mx = t
  const grid = f.el('div', 'tri-zone-grid', undefined, { role: 'list' })
  for (let i = times.length - 1; i >= 0; i--) {
    const name = names[i] ?? `Z${i + 1}`
    const range = `${zoneRange(bounds, i)}${unit}`
    const time = zoneClock(times[i])
    const percentage = `${((times[i] / total) * 100).toFixed(1)}%`
    const row = f.el('div', 'tri-zone-row', undefined, {
      role: 'listitem',
      'aria-label': `Z${i + 1}, ${name}, ${range}, ${time}, ${percentage}`,
    })
    const track = f.el('span', 'tri-zone-bar')
    f.add(
      track,
      f.el('span', `tri-zone-fill tri-zone-fill--${i + 1}`, undefined, {
        style: `width:${(times[i] / mx) * 100}%`,
      }),
    )
    f.add(
      row,
      f.el('span', 'tri-zone-z', `Z${i + 1}`),
      f.el('span', 'tri-zone-name', name),
      f.el('span', 'tri-zone-range', range),
      f.el('span', 'tri-zone-time', time),
      f.el('span', 'tri-zone-pct', percentage),
      track,
    )
    f.add(grid, row)
  }
  f.add(wrap, grid)
  if (caption) f.add(wrap, f.el('div', 'tri-zone-cap', caption))
  return wrap
}

export const buildHrZones = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  ctx: DetailCtx,
): N | null => {
  if (!d.hrZones || !ctx.zones?.hr.length) return null
  return zoneTable(
    f,
    'heart rate zones',
    d.hrZones,
    ctx.zones.hr,
    HR_ZONE_NAMES,
    '',
    ctx.vt1 != null ? `based on vt1 ${ctx.vt1} bpm` : '',
  )
}

export const buildPowerZones = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  ctx: DetailCtx,
): N | null => {
  if (!d.powerZones || !ctx.zones?.power.length) return null
  const ftp = ctx.zones.ftp
  return zoneTable(
    f,
    'power zones',
    d.powerZones,
    ctx.zones.power,
    POWER_ZONE_NAMES,
    'w',
    ftp != null ? `based on FTP ${ftp} W` : '',
  )
}

export const powerHistogramReadout = (
  histogram: readonly number[],
  bin: number,
  unit: 'W' | 'idx',
): string => {
  const lower = bin * SWIM_POWER_BIN_SIZE
  const upper = lower + SWIM_POWER_BIN_SIZE - (unit === 'W' ? 1 : 0)
  const total = histogram.reduce((sum, seconds) => sum + seconds, 0)
  const seconds = histogram[bin] ?? 0
  return `${lower}–${upper} ${unit} · ${zoneClock(seconds)} (${(total > 0 ? (seconds / total) * 100 : 0).toFixed(1)}%)`
}

export const buildPowerHist = <N>(f: TriNodeFactory<N>, d: StravaActivityDetail): N | null => {
  const swim = d.sport === 'swim'
  const hist = swim ? d.swimPower?.histogramS : d.powerHist
  if (!hist || hist.length < (swim ? 1 : 2)) return null
  const unit = swim ? 'idx' : 'W'
  const title = swim ? '25 idx drag distribution' : '25W power distribution'
  const attrs: Record<string, string> = {
    'data-tri-trace': swim ? 'swim-drag-distribution' : triathlonTraceName(title),
  }
  if (swim && d.swimPower) attrs['data-swim-power-source'] = d.swimPower.source
  const wrap = f.el(
    'div',
    swim ? 'tri-zone tri-swim-power-distribution' : 'tri-zone',
    undefined,
    attrs,
  )
  const titleAttrs: Record<string, string> =
    swim && d.swimPower
      ? {
          'data-gloss': '',
          'data-gloss-def': `Valid swim time in 25 idx bands, with the upper boundary included in the next band. ${swimDragPowerDefinition(d.swimPower.speedBasis)}`,
          tabindex: '0',
        }
      : { 'data-i18n': title }
  f.add(wrap, f.el('div', 'tri-zone-title', title, titleAttrs))
  const H = 34
  const n = hist.length
  const binSize = SWIM_POWER_BIN_SIZE
  const histMax = n * binSize
  const selected = hist.reduce((best, seconds, bin) => (seconds > hist[best] ? bin : best), 0)
  let mx = 1
  for (const t of hist) if (t > mx) mx = t
  const s = f.svg('svg', {
    class: 'tri-hist-svg',
    viewBox: `0 0 ${n} ${H}`,
    preserveAspectRatio: 'none',
    'data-hist': JSON.stringify(hist),
    'data-hist-unit': unit,
    role: 'slider',
    tabindex: 0,
    'aria-label': swim ? 'modeled swim drag distribution' : 'power distribution',
    'aria-orientation': 'horizontal',
    'aria-valuemin': 0,
    'aria-valuemax': n - 1,
    'aria-valuenow': selected,
    'aria-valuetext': powerHistogramReadout(hist, selected, unit),
  })
  hist.forEach((t, i) => {
    if (t <= 0) return
    const h = (t / mx) * (H - 1)
    f.add(
      s,
      f.svg('rect', {
        x: i + 0.1,
        y: H - h,
        width: 0.8,
        height: h,
        class: 'tri-hist-bar',
        'data-bin': i,
      }),
    )
  })
  const np = swim ? d.swimPower?.averageIndex : (d.npWatts ?? d.avgWatts)
  if (np != null) {
    const position = np / binSize + (swim ? 0 : 0.5)
    f.add(s, f.svg('line', { x1: position, y1: 0, x2: position, y2: H, class: 'tri-hist-avg' }))
  }
  f.add(s, f.svg('line', { class: 'tri-chart-cursor', x1: 0, y1: 0, x2: 0, y2: H }))
  const histStep = histMax <= 300 ? 100 : histMax <= 700 ? 200 : 300
  const histXTicks: AxisXTick[] = []
  for (let value = 0; value < histMax; value += histStep)
    histXTicks.push({
      label: swim ? `${value} idx` : `${value}w`,
      pct: (value / histMax) * 100,
      cls: value === 0 ? 'tri-cax-xt--first' : undefined,
    })
  f.add(
    wrap,
    axisFrame(
      f,
      s,
      [
        { label: zoneClock(mx), vbY: H - (H - 1) },
        { label: zoneClock(Math.round(mx / 2)), vbY: H - (H - 1) / 2 },
        { label: '0', vbY: H },
      ],
      H,
      histXTicks,
    ),
  )
  f.add(wrap, f.el('div', 'tri-chart-readout'))
  const cap = f.el('div', 'tri-elev-cap')
  f.add(cap, f.el('span', 'tri-ana-k', `0–${histMax - (swim ? 0 : 1)} ${unit}`))
  if (np != null)
    f.add(cap, f.el('span', 'tri-ana-k', swim ? `avg ${np.toFixed(1)} idx` : `wtd avg ${np} W`))
  f.add(wrap, cap)
  return wrap
}
