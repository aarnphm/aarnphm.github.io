import type { Analytics } from '../../../../plugins/stores/analytics'
import type { BodyBlock } from '../../../../plugins/stores/analytics'
import type { TriathlonContext } from '../../runtime/context'
import { axisFrame } from '../../../../util/triathlon-card'
import { createDomFactory } from '../../runtime/dom'
import { el } from '../../runtime/dom'
import { mathK } from '../../runtime/dom'
import { svg } from '../../runtime/dom'
import { ANA_W } from '../shared'
import { anaTitle } from '../shared'
import { clampN } from '../shared'
import { markGloss } from '../shared'
import { monthTicks } from '../shared'
import { pctFmt } from '../shared'
import { polyD } from '../shared'
import { weightSwitch } from '../shared'
import { weightUnitLabel } from '../shared'
import { wFmt } from '../shared'
import { wNum } from '../shared'
import { wSigned } from '../shared'
import { buildWeekTrend } from './performance'

export interface BodyDay {
  date: string
  ts: number
  samples: { date: string; ts: number; kg: number }[]
  min: number
  max: number
  first: number
  last: number
}

export const groupBodyByDay = (series: { date: string; ts: number; kg: number }[]): BodyDay[] => {
  const byDay = new Map<string, { date: string; ts: number; kg: number }[]>()
  for (const p of series) {
    const arr = byDay.get(p.date)
    if (arr) arr.push(p)
    else byDay.set(p.date, [p])
  }
  return [...byDay.values()]
    .map(arr => {
      const sorted = arr.slice().sort((a, b) => a.ts - b.ts)
      let mn = Infinity
      let mx = -Infinity
      for (const q of sorted) {
        if (q.kg < mn) mn = q.kg
        if (q.kg > mx) mx = q.kg
      }
      const last = sorted[sorted.length - 1]
      return {
        date: sorted[0].date,
        ts: last.ts,
        samples: sorted,
        min: mn,
        max: mx,
        first: sorted[0].kg,
        last: last.kg,
      }
    })
    .sort((a, b) => a.ts - b.ts)
}

export const buildBody = (data: Analytics, context: TriathlonContext): HTMLElement => {
  const text = (key: string): string => context.formatter.text(key)
  const block = el('div', 'tri-ana-bodywt')
  const title = anaTitle(context.formatter, 'body weight', 'weight')
  const b: BodyBlock = data.body
  if (b.latestKg == null) {
    block.appendChild(title)
    block.appendChild(el('div', 'tri-ana-empty', text('no weight logged')))
    return block
  }
  const titleRow = el('div', 'tri-bodywt-titlerow')
  titleRow.append(title, weightSwitch(context))
  block.appendChild(titleRow)
  const head = el('div', 'tri-bodywt-head')
  head.append(el('span', 'tri-bodywt-kg', wFmt(context.formatter, b.latestKg)))
  block.appendChild(head)
  const pts = b.series
  if (pts.length >= 2) {
    const bmr = Array.isArray(b.bmrSeries) ? b.bmrSeries : []
    let min = Infinity
    let max = -Infinity
    for (const p of pts) {
      if (p.kg < min) min = p.kg
      if (p.kg > max) max = p.kg
    }
    if (b.goalKg != null) {
      if (b.goalKg < min) min = b.goalKg
      if (b.goalKg > max) max = b.goalKg
    }
    const range = Math.max(0.5, max - min)
    const lo = min - range * 0.18
    const hi = max + range * 0.18
    const days = groupBodyByDay(pts)
    const nd = days.length
    const t0 = pts[0].ts
    const t1 = pts[pts.length - 1].ts
    const xPct = (ts: number): number => (t1 > t0 ? ((ts - t0) / (t1 - t0)) * 100 : 50)
    const yPct = (kg: number): number => (1 - (kg - lo) / (hi - lo)) * 100
    const chart = el('div', 'tri-bodywt-chart')
    const yax = el('div', 'tri-bodywt-yax')
    yax.append(
      el('span', '', wNum(context.formatter, hi)),
      el('span', '', wNum(context.formatter, lo)),
    )
    const plot = el('div', 'tri-bodywt-plot')
    const s = svg('svg', {
      class: 'tri-bodywt-svg',
      viewBox: '0 0 100 100',
      preserveAspectRatio: 'none',
    })
    for (const gy of [0, 50])
      s.appendChild(svg('line', { x1: 0, y1: gy, x2: 100, y2: gy, class: 'tri-bodywt-grid' }))
    s.appendChild(svg('line', { x1: 0, y1: 0, x2: 0, y2: 100, class: 'tri-bodywt-axis' }))
    s.appendChild(svg('line', { x1: 0, y1: 100, x2: 100, y2: 100, class: 'tri-bodywt-axis' }))
    if (bmr.length >= 2)
      s.appendChild(
        svg('line', {
          x1: 100,
          y1: 0,
          x2: 100,
          y2: 100,
          class: 'tri-bodywt-axis tri-bodywt-axis--bmr',
        }),
      )
    if (b.goalKg != null)
      s.appendChild(
        svg('line', {
          x1: 0,
          y1: yPct(b.goalKg),
          x2: 100,
          y2: yPct(b.goalKg),
          class: 'tri-bodywt-goal',
        }),
      )
    for (const d of days) {
      if (d.samples.length < 2) continue
      const dx = xPct(d.ts).toFixed(2)
      s.appendChild(
        svg('line', {
          x1: dx,
          y1: yPct(d.max),
          x2: dx,
          y2: yPct(d.min),
          class: 'tri-bodywt-range',
          'data-day': d.date,
        }),
      )
    }
    s.appendChild(
      svg('path', {
        d: polyD(days.map(d => [xPct(d.ts), yPct(d.last)])),
        class: 'tri-bodywt-line',
      }),
    )
    s.appendChild(svg('line', { x1: 0, y1: 0, x2: 0, y2: 100, class: 'tri-ana-cursor' }))
    plot.appendChild(s)
    days.forEach((d, di) => {
      const left = `${xPct(d.ts).toFixed(2)}%`
      d.samples.forEach((sample, si) => {
        const dayLast = si === d.samples.length - 1
        const cls =
          di === nd - 1 && dayLast
            ? 'tri-bodywt-pt tri-bodywt-pt--last'
            : dayLast
              ? 'tri-bodywt-pt'
              : 'tri-bodywt-pt tri-bodywt-pt--sub'
        const m = el('span', cls)
        m.style.left = left
        m.style.top = `${yPct(sample.kg).toFixed(2)}%`
        plot.appendChild(m)
      })
    })
    let yaxR: HTMLElement | null = null
    if (bmr.length >= 2) {
      const byDayB = new Map<string, { ts: number; bmr: number }>()
      for (const p of bmr) {
        const e = byDayB.get(p.date)
        if (!e || p.ts > e.ts) byDayB.set(p.date, { ts: p.ts, bmr: p.bmr })
      }
      const bd = [...byDayB.values()].sort((p, q) => p.ts - q.ts)
      let blo = Infinity
      let bhi = -Infinity
      for (const p of bd) {
        if (p.bmr < blo) blo = p.bmr
        if (p.bmr > bhi) bhi = p.bmr
      }
      const goalBmr = b.goalBmr ?? b.goalLeanBmr
      if (goalBmr != null) {
        if (goalBmr < blo) blo = goalBmr
        if (goalBmr > bhi) bhi = goalBmr
      }
      const brange = Math.max(40, bhi - blo)
      let bLoP = blo - brange * 0.18
      let bHiP = bhi + brange * 0.18
      if (goalBmr != null && b.goalKg != null) {
        const weightGoalY = yPct(b.goalKg)
        const goalY = (1 - (goalBmr - bLoP) / (bHiP - bLoP)) * 100
        if (Math.abs(goalY - weightGoalY) < 10) {
          const targetY = clampN(weightGoalY - 10, 12, 88)
          bLoP = Math.min(bLoP, bHiP - ((bHiP - goalBmr) * 100) / targetY)
        }
      }
      const bY = (v: number): number => (1 - (v - bLoP) / (bHiP - bLoP)) * 100
      if (goalBmr != null)
        s.appendChild(
          svg('line', {
            x1: 0,
            y1: bY(goalBmr),
            x2: 100,
            y2: bY(goalBmr),
            class: 'tri-bodywt-bmr-goal',
          }),
        )
      const firstB = bd[0]
      const firstBx = xPct(firstB.ts)
      if (firstBx > 0)
        s.appendChild(
          svg('line', {
            x1: 0,
            y1: bY(firstB.bmr),
            x2: firstBx.toFixed(2),
            y2: bY(firstB.bmr),
            class: 'tri-bodywt-bmr-missing',
          }),
        )
      s.appendChild(
        svg('path', { d: polyD(bd.map(p => [xPct(p.ts), bY(p.bmr)])), class: 'tri-bodywt-bmr' }),
      )
      const lastB = bd[bd.length - 1]
      const bm = el('span', 'tri-bodywt-bpt')
      bm.style.left = `${xPct(lastB.ts).toFixed(2)}%`
      bm.style.top = `${bY(lastB.bmr).toFixed(2)}%`
      plot.appendChild(bm)
      yaxR = el('div', 'tri-bodywt-yax tri-bodywt-yax-r')
      yaxR.append(el('span', '', `${Math.round(bHiP)}`), el('span', '', `${Math.round(bLoP)}`))
    }
    chart.append(yax, plot)
    if (yaxR) {
      chart.appendChild(yaxR)
      block.classList.add('tri-bodywt--bmr')
    }
    const xax = el('div', 'tri-bodywt-xax')
    xax.append(
      el('span', '', context.formatter.shortDate(days[0].date)),
      el('span', '', context.formatter.shortDate(days[nd - 1].date)),
    )
    block.append(chart, xax)
    block.appendChild(el('div', 'tri-chart-readout'))
  }
  const cap = el('div', 'tri-elev-cap')
  if (b.trendKgPerWeek != null)
    cap.appendChild(
      markGloss(
        el(
          'span',
          'tri-ana-k',
          `${wSigned(context.formatter, b.trendKgPerWeek, 2)} ${weightUnitLabel(context.formatter)}/${text('wk')}`,
        ),
        'wtrend',
      ),
    )
  if (b.goalKg != null) {
    const delta =
      b.goalDeltaKg != null
        ? ` (${wSigned(context.formatter, b.goalDeltaKg, 1)} ${weightUnitLabel(context.formatter)})`
        : ''
    const eta = b.goalEtaWeeks != null ? ` · $\\approx${b.goalEtaWeeks}$ ${text('wk')}` : ''
    cap.appendChild(
      markGloss(
        mathK('tri-ana-k', `${text('goal')} ${wFmt(context.formatter, b.goalKg)}${delta}${eta}`),
        'wgoal',
      ),
    )
  }
  if (b.goalBmr != null || b.goalLeanBmr != null) {
    const lean = b.goalLeanBmr != null ? ` · ${text('FFM')} ${b.goalLeanBmr} kcal` : ''
    const bmrText =
      b.goalBmr != null
        ? `${text('goal')} ${text('BMR')} ${b.goalBmr} kcal${lean}`
        : `${text('goal')} ${text('BMR')} ${b.goalLeanBmr} kcal`
    cap.appendChild(markGloss(el('span', 'tri-ana-k tri-bmr-k', bmrText), 'bmr'))
  }
  if (b.bodyFatPct != null)
    cap.appendChild(
      markGloss(
        el('span', 'tri-ana-k', `${text('fat')} ${pctFmt(context.formatter, b.bodyFatPct, 1)}`),
        'bodyfat',
      ),
    )
  if (b.ffmi != null)
    cap.appendChild(
      markGloss(
        el('span', 'tri-ana-k', `${text('FFMI')} ${context.formatter.number(b.ffmi, 1, 1)}`),
        'ffmi',
      ),
    )
  if (b.bmi != null)
    cap.appendChild(
      markGloss(
        el('span', 'tri-ana-k', `${text('bmi')} ${context.formatter.number(b.bmi, 1, 1)}`),
        'bmi',
      ),
    )
  if (b.latestBmr != null)
    cap.appendChild(
      markGloss(el('span', 'tri-ana-k tri-bmr-k', `${text('BMR')} ${b.latestBmr} kcal`), 'bmr'),
    )
  if (b.muscleMassKg != null)
    cap.appendChild(
      el('span', 'tri-ana-k', `${text('muscle')} ${wFmt(context.formatter, b.muscleMassKg, 1, 1)}`),
    )
  if (b.boneMassKg != null)
    cap.appendChild(
      el('span', 'tri-ana-k', `${text('bone')} ${wFmt(context.formatter, b.boneMassKg, 1, 1)}`),
    )
  if (b.bodyWaterPct != null)
    cap.appendChild(
      el('span', 'tri-ana-k', `${text('water')} ${pctFmt(context.formatter, b.bodyWaterPct, 1)}`),
    )
  const next = (data.events ?? [])
    .filter(e => e.date >= data.meta.today)
    .sort((a, b2) => a.date.localeCompare(b2.date))[0]
  if (next)
    cap.appendChild(
      el(
        'span',
        'tri-ana-k',
        `${next.event ?? text('race')} · ${context.formatter.longDate(next.date)}`,
      ),
    )
  block.appendChild(cap)
  return block
}

export const buildEffort = (data: Analytics, context: TriathlonContext): HTMLElement => {
  const block = buildWeekTrend(data, 'effort', context)
  const all = data.weekly
  if (!all.some(w => w.effort > 0)) return block
  let mx = 1
  for (const w of all) if (w.effort > mx) mx = w.effort
  const cap = el('div', 'tri-elev-cap')
  cap.appendChild(el('span', 'tri-ana-k', `${context.formatter.text('peak')} ${Math.round(mx)}`))
  block.appendChild(cap)
  return block
}

export const segRuns = <T>(
  rows: T[],
  sel: (r: T) => number | null,
  x: (i: number) => number,
  y: (v: number) => number,
): [number, number][][] => {
  const out: [number, number][][] = []
  let cur: [number, number][] = []
  rows.forEach((r, i) => {
    const v = sel(r)
    if (v == null) {
      if (cur.length > 1) out.push(cur)
      cur = []
      return
    }
    cur.push([x(i), y(v)])
  })
  if (cur.length > 1) out.push(cur)
  return out
}

export const missingBridges = <T>(
  rows: T[],
  sel: (r: T) => number | null,
  x: (i: number) => number,
  y: (v: number) => number,
): [number, number][][] => {
  const out: [number, number][][] = []
  let previous: { index: number; value: number } | null = null
  for (const [index, row] of rows.entries()) {
    const value = sel(row)
    if (value == null) continue
    if (previous != null && index > previous.index + 1)
      out.push([
        [x(previous.index), y(previous.value)],
        [x(index), y(value)],
      ])
    previous = { index, value }
  }
  return out
}

export const missingRuns = <T>(
  rows: T[],
  sel: (r: T) => number | null,
  x: (i: number) => number,
  y: (v: number) => number,
): [number, number][][] => {
  const out: [number, number][][] = []
  let previous: { i: number; value: number } | null = null
  for (const [i, row] of rows.entries()) {
    const value = sel(row)
    if (value == null) continue
    if (previous == null) {
      if (i > 0)
        out.push([
          [x(0), y(value)],
          [x(i), y(value)],
        ])
    } else if (i > previous.i + 1) {
      out.push([
        [x(previous.i), y(previous.value)],
        [x(i), y(value)],
      ])
    }
    previous = { i, value }
  }
  if (previous != null && previous.i < rows.length - 1)
    out.push([
      [x(previous.i), y(previous.value)],
      [x(rows.length - 1), y(previous.value)],
    ])
  return out
}

export const buildHeatAcclimatisation = (
  data: Analytics,
  context: TriathlonContext,
): HTMLElement => {
  const text = (key: string): string => context.formatter.text(key)
  const block = el('div', 'tri-ana-accl')
  const heat = data.heat
  const coreActivities = heat.activities.filter(activity => activity.heatStrainIndex != null)
  const usesCore = coreActivities.length > 0
  block.appendChild(
    anaTitle(
      context.formatter,
      usesCore ? 'heat strain · acclimatisation' : 'ambient heat · acclimatisation',
      'heatacclimation',
    ),
  )
  if (!heat.series.length || heat.currentPct == null) {
    block.appendChild(el('div', 'tri-ana-empty', text('no thermal data')))
    return block
  }

  const temperature = (celsius: number): number =>
    context.presentation.distance === 'imperial' ? (celsius * 9) / 5 + 32 : celsius
  const temperatureUnit = context.presentation.distance === 'imperial' ? '°F' : '°C'
  const temperatureText = (celsius: number, digits = 0): string =>
    `${temperature(celsius).toFixed(digits)}${temperatureUnit}`
  const summary = el('div', 'tri-accl-summary')
  summary.append(
    el('span', 'tri-accl-summary-v', `${heat.heatDays14d}`),
    el('span', 'tri-accl-summary-k', text('heat days')),
    el('span', 'tri-accl-summary-k tri-accl-summary-dot', '·'),
    el('span', 'tri-accl-summary-k', text('14d')),
    el('span', 'tri-accl-summary-v', `${heat.heatMinutes14d}`),
    el('span', 'tri-accl-summary-k', text('hot min')),
    el('span', 'tri-accl-summary-k tri-accl-summary-dot', '·'),
    el('span', 'tri-accl-summary-k', text('14d')),
  )
  block.appendChild(summary)

  const legend = el('div', 'tri-accl-legend')
  const legendItem = (cls: string, text: string): HTMLElement => {
    const item = el('span', 'tri-accl-legitem')
    item.append(el('span', `tri-accl-legmark ${cls}`), el('span', '', context.formatter.text(text)))
    return item
  }
  legend.append(
    legendItem('tri-accl-leg-temp', usesCore ? 'heat strain index' : 'activity temperature'),
    legendItem('tri-accl-leg-proxy', 'acclimatisation proxy'),
    legendItem('tri-accl-leg-dose', 'heat exposure'),
  )
  block.appendChild(legend)

  const rows = heat.series
  const activities = usesCore ? coreActivities : heat.activities
  const n = rows.length
  const H = 70
  const tempTop = 3
  const tempBottom = 28
  const acclTop = 40
  const acclBottom = 66
  const signal = (activity: (typeof activities)[number]): number =>
    usesCore ? (activity.heatStrainIndex ?? 0) : temperature(activity.temperatureC)
  const observed = activities.map(signal)
  const threshold = usesCore
    ? heat.method.heatStrainThreshold
    : temperature(heat.method.hotThresholdC)
  const signalStep = usesCore ? 1 : 5
  const minimumSpan = usesCore ? 2 : 10
  let signalLo = Math.floor(Math.min(...observed, threshold) / signalStep) * signalStep
  let signalHi = Math.ceil(Math.max(...observed, threshold) / signalStep) * signalStep
  if (signalHi - signalLo < minimumSpan) {
    signalLo -= signalStep
    signalHi += signalStep
  }
  const x = (i: number): number => (n > 1 ? (i / (n - 1)) * ANA_W : ANA_W / 2)
  const fromMs = Date.parse(`${rows[0].date}T00:00:00Z`)
  const toMs = Date.parse(`${rows[rows.length - 1].date}T23:59:59Z`)
  const xActivity = (startedAt: string): number => {
    const timestamp = Date.parse(startedAt)
    return Number.isFinite(timestamp) && toMs > fromMs
      ? clampN((timestamp - fromMs) / (toMs - fromMs), 0, 1) * ANA_W
      : ANA_W / 2
  }
  const ySignal = (value: number): number =>
    tempBottom - ((value - signalLo) / (signalHi - signalLo)) * (tempBottom - tempTop)
  const yAccl = (value: number): number =>
    acclBottom - (clampN(value, 0, 100) / 100) * (acclBottom - acclTop)
  const s = svg('svg', {
    class: 'tri-ana-svg tri-accl-svg',
    viewBox: `0 0 ${ANA_W} ${H}`,
    preserveAspectRatio: 'none',
    role: 'img',
    'aria-label': text(
      usesCore
        ? 'CORE heat strain and heat acclimatisation over time'
        : 'ambient workout temperature and heat acclimatisation proxy over time',
    ),
  })
  s.appendChild(
    svg('rect', {
      x: 0,
      y: tempTop,
      width: ANA_W,
      height: Math.max(0, ySignal(threshold) - tempTop),
      class: 'tri-accl-hot-zone',
    }),
  )
  s.appendChild(
    svg('line', {
      x1: 0,
      y1: ySignal(threshold),
      x2: ANA_W,
      y2: ySignal(threshold),
      class: 'tri-accl-threshold',
    }),
  )
  const barWidth = clampN(72 / Math.max(1, n), 0.45, 1.35)
  for (const [i, day] of rows.entries()) {
    if (day.dose > 0)
      s.appendChild(
        svg('rect', {
          x: x(i) - barWidth / 2,
          y: yAccl(day.dose * 100),
          width: barWidth,
          height: acclBottom - yAccl(day.dose * 100),
          class: 'tri-accl-dose',
        }),
      )
  }
  if (activities.length)
    s.appendChild(
      svg('path', {
        d: polyD(
          activities.map(activity => [xActivity(activity.startedAt), ySignal(signal(activity))]),
        ),
        class: 'tri-accl-temp',
      }),
    )
  s.appendChild(
    svg('path', {
      d: polyD(rows.map((day, i) => [x(i), yAccl(day.acclimatisationPct)])),
      class: 'tri-accl-proxy',
    }),
  )
  s.appendChild(
    svg('line', {
      x1: 0,
      y1: (tempBottom + acclTop) / 2,
      x2: ANA_W,
      y2: (tempBottom + acclTop) / 2,
      class: 'tri-accl-divider',
    }),
  )
  s.appendChild(svg('line', { x1: 0, y1: 0, x2: 0, y2: H, class: 'tri-ana-cursor' }))
  if (usesCore)
    for (const activity of activities) {
      const pointX = xActivity(activity.startedAt)
      const pointY = ySignal(signal(activity))
      s.appendChild(
        svg('line', {
          x1: pointX - 0.001,
          y1: pointY,
          x2: pointX + 0.001,
          y2: pointY,
          class: 'tri-accl-temp-point',
        }),
      )
    }
  const frame = axisFrame(
    createDomFactory(context.presentation),
    s,
    [
      { label: `${signalHi}${usesCore ? '' : temperatureUnit}`, vbY: tempTop },
      {
        label: `${usesCore ? threshold.toFixed(1) : Math.round(threshold)}${usesCore ? '' : temperatureUnit}`,
        vbY: ySignal(threshold),
      },
      { label: `${signalLo}${usesCore ? '' : temperatureUnit}`, vbY: tempBottom },
      { label: '100%', vbY: acclTop },
      { label: '50%', vbY: yAccl(50) },
      { label: '0%', vbY: acclBottom },
    ],
    H,
    monthTicks(
      context.formatter,
      rows.map(day => day.date),
      i => x(i),
    ),
    false,
  )
  frame.classList.add('tri-accl-frame')
  block.append(frame, el('div', 'tri-chart-readout'))

  const cap = el('div', 'tri-elev-cap tri-accl-cap')
  const latestCore = coreActivities.at(-1)
  if (usesCore && latestCore)
    cap.appendChild(
      markGloss(
        el(
          'span',
          'tri-ana-k',
          `${text('latest')} ${latestCore.coreOrigin === 'app' ? 'CORE app' : 'CORE FIT'} ${temperatureText(latestCore.temperatureC, 2)} · HSI ${latestCore.heatStrainIndex?.toFixed(1) ?? '—'} · ${context.formatter.shortDate(latestCore.date)}`,
        ),
        'heatstrain',
      ),
    )
  else if (heat.latestTemperatureC != null && heat.lastObservedDate)
    cap.appendChild(
      markGloss(
        el(
          'span',
          'tri-ana-k',
          `${text('latest')} ${temperatureText(heat.latestTemperatureC)} · ${context.formatter.shortDate(heat.lastObservedDate)}`,
        ),
        'ambienttemp',
      ),
    )
  cap.append(
    markGloss(
      el('span', 'tri-ana-k', `${heat.coveragePct}% ${text('thermal coverage')}`),
      usesCore ? 'heatstrain' : 'ambienttemp',
    ),
    el('span', 'tri-ana-k', `${text(heat.confidence)} ${text('confidence')}`),
    el(
      'span',
      'tri-ana-k',
      `CORE app ${heat.coreSourceCounts.app} · CORE FIT ${heat.coreSourceCounts.fit} · WeatherKit ${heat.sourceCounts.weatherkit} · Strava ${heat.sourceCounts.strava}`,
    ),
  )
  block.appendChild(cap)
  const method = markGloss(
    mathK(
      'tri-accl-method',
      `${usesCore ? `HSI ≥${heat.method.heatStrainThreshold.toFixed(1)} · ${text('fallback')} >${temperatureText(heat.method.hotThresholdC)}` : `>${temperatureText(heat.method.hotThresholdC)}`} · ${heat.method.targetMinutesPerDay} min = 1 ${text('exposure')} · ${heat.method.targetDays} ${text('exposures')} = 100% · ${heat.method.decayPerDay * 100}%/${text('day')} ${text('decay after')} ${heat.method.decayGraceDays} ${text('days')}`,
    ),
    'heatdose',
  )
  block.appendChild(method)
  return block
}
