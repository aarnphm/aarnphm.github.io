import type { ActivitySummary } from '../../../../plugins/stores/analytics'
import type { Analytics } from '../../../../plugins/stores/analytics'
import type { DailyPoint } from '../../../../plugins/stores/analytics'
import type { Sport } from '../../../../plugins/stores/strava'
import type { TriathlonContext } from '../../runtime/context'
import { axisFrame } from '../../../../util/triathlon-card'
import { dist } from '../../../../util/triathlon-card'
import { weeklyChartX } from '../../../../util/weekly-target-range'
import { buildIcon } from '../../activity/primitives'
import { createDomFactory } from '../../runtime/dom'
import { el } from '../../runtime/dom'
import { mathK } from '../../runtime/dom'
import { svg } from '../../runtime/dom'
import { ANA_W } from '../shared'
import { anaTitle } from '../shared'
import { buildIconLeg } from '../shared'
import { bySport } from '../shared'
import { clampN } from '../shared'
import { hms } from '../shared'
import { K42 } from '../shared'
import { K7 } from '../shared'
import { markGloss } from '../shared'
import { monthTicks } from '../shared'
import { niceUp } from '../shared'
import { PMC_BAR_BOT } from '../shared'
import { PMC_BAR_TOP } from '../shared'
import { PMC_BOT } from '../shared'
import { PMC_H } from '../shared'
import { PMC_PROJ_DAYS } from '../shared'
import { PMC_TOP } from '../shared'
import { PMC_TSB_HALF } from '../shared'
import { PMC_TSB_ZERO } from '../shared'
import { polyD } from '../shared'
import { signed } from '../shared'
import { signedFixed } from '../shared'
import { initialPerformanceModel, updatePerformance } from './performance-model'
import { fmtKm } from './thresholds'
import { fmtSignedKm } from './thresholds'

export const buildPmc = (
  data: Analytics,
  context: TriathlonContext,
): { element: HTMLElement; mount?: () => () => void } => {
  const block = el('div', 'tri-ana-pmc')
  const daily = data.daily
  const n = daily.length
  if (n < 2) {
    block.appendChild(el('div', 'tri-ana-empty', context.formatter.text('not enough data')))
    return { element: block }
  }
  const activitiesByDate = new Map<string, ActivitySummary[]>()
  for (const activity of data.activities) {
    const current = activitiesByDate.get(activity.date)
    if (current) current.push(activity)
    else activitiesByDate.set(activity.date, [activity])
  }
  const raceDates = new Set(data.events.map(event => event.date))
  const r = data.risk
  const ago = Math.max(0, n - 8)
  const delta = (get: (d: DailyPoint) => number): number =>
    Math.round(get(daily[n - 1]) - get(daily[ago]))
  const stat = (
    cls: string,
    label: string,
    value: string,
    d: number,
    gloss: string,
    zone?: string,
  ): HTMLElement => {
    const wrap = el('div', `tri-pmc-stat ${cls}`)
    const head = el('div', 'tri-pmc-stat-head')
    head.append(
      el('span', 'tri-pmc-dot'),
      el('span', 'tri-pmc-stat-k', context.formatter.text(label)),
    )
    wrap.append(
      head,
      el('div', `tri-pmc-stat-v${zone ? ` tri-zone-${zone}` : ''}`, value),
      el('div', 'tri-pmc-stat-d', `${signed(d)} · 7d`),
    )
    return markGloss(wrap, gloss)
  }
  const readout = el('div', 'tri-pmc-now-row')
  readout.append(
    stat(
      'tri-pmc-fit',
      'fitness',
      String(Math.round(r.ctl)),
      delta(d => d.ctl),
      'ctl',
    ),
    stat(
      'tri-pmc-fat',
      'fatigue',
      String(Math.round(r.atl)),
      delta(d => d.atl),
      'atl',
    ),
    stat(
      'tri-pmc-form',
      'form',
      signed(Math.round(r.tsb)),
      delta(d => d.tsb),
      'tsb',
      r.tsbZone,
    ),
  )
  block.appendChild(readout)

  const H = PMC_PROJ_DAYS
  const N = n + H
  const lastMs = Date.parse(`${daily[n - 1].date}T00:00:00Z`)
  const projDate = (k: number): string => new Date(lastMs + k * 86400000).toISOString().slice(0, 10)
  type Proj = { ctl: number; atl: number; tsb: number }
  const project = (load: number): Proj[] => {
    let c = daily[n - 1].ctl
    let a = daily[n - 1].atl
    const out: Proj[] = []
    for (let k = 0; k < H; k++) {
      c += (load - c) * K42
      a += (load - a) * K7
      out.push({ ctl: c, atl: a, tsb: c - a })
    }
    return out
  }

  let maxFitRaw = 1
  let tsbAbsRaw = 10
  let maxLoad = 1
  let loadSum = 0
  for (const d of daily) {
    maxFitRaw = Math.max(maxFitRaw, d.ctl, d.atl)
    tsbAbsRaw = Math.max(tsbAbsRaw, Math.abs(d.tsb))
    maxLoad = Math.max(maxLoad, d.load)
  }
  for (const d of daily.slice(Math.max(0, n - 14))) loadSum += d.load
  const avgRecent = Math.round(loadSum / Math.min(14, n))
  const LOAD_MAX = niceUp(Math.max(120, Math.round(avgRecent * 1.4)))
  const performanceBounds = { lastObservedIndex: n - 1, maximumIndex: N - 1, maximumLoad: LOAD_MAX }
  let model = initialPerformanceModel(avgRecent, performanceBounds)
  let futLoad = model.futureDailyLoad
  for (const p of project(LOAD_MAX)) maxFitRaw = Math.max(maxFitRaw, p.ctl, p.atl)
  for (const p of [...project(LOAD_MAX), ...project(0)])
    tsbAbsRaw = Math.max(tsbAbsRaw, Math.abs(p.tsb))
  const maxFit = niceUp(maxFitRaw)
  const tsbAbs = niceUp(tsbAbsRaw)

  const x = (i: number): number => (i / (N - 1)) * ANA_W
  const yFit = (v: number): number => PMC_BOT - (v / maxFit) * (PMC_BOT - PMC_TOP)
  const yTsb = (v: number): number => PMC_TSB_ZERO - (v / tsbAbs) * PMC_TSB_HALF
  const yBar = (v: number): number => PMC_BAR_BOT - (v / maxLoad) * (PMC_BAR_BOT - PMC_BAR_TOP)
  const nowX = x(n - 1)

  const ctlPts = daily.map((d, i) => [x(i), yFit(d.ctl)] as [number, number])
  const atlPts = daily.map((d, i) => [x(i), yFit(d.atl)] as [number, number])
  const tsbPts = daily.map((d, i) => [x(i), yTsb(d.tsb)] as [number, number])

  let projSeries = project(futLoad)
  const projPath = (
    anchor: number,
    get: (p: Proj) => number,
    yfn: (v: number) => number,
  ): string => {
    const pts: [number, number][] = [[nowX, yfn(anchor)]]
    projSeries.forEach((p, k) => pts.push([x(n + k), yfn(get(p))]))
    return polyD(pts)
  }

  const s = svg('svg', {
    class: 'tri-ana-svg tri-pmc-svg',
    viewBox: `0 0 ${ANA_W} ${PMC_H}`,
    preserveAspectRatio: 'none',
  })
  s.appendChild(
    svg('line', { x1: 0, y1: yFit(maxFit), x2: ANA_W, y2: yFit(maxFit), class: 'tri-pmc-grid' }),
  )
  const areaInner = ctlPts.map(([px, py]) => `L ${px.toFixed(2)} ${py.toFixed(2)}`).join(' ')
  s.appendChild(
    svg('path', {
      d: `M 0 ${PMC_BOT} ${areaInner} L ${nowX.toFixed(2)} ${PMC_BOT} Z`,
      class: 'tri-pmc-area',
    }),
  )
  const bw = (ANA_W / N) * 0.62
  for (let i = 0; i < n; i++) {
    const day = daily[i]
    const load = day.load
    if (load <= 0) continue
    const by = yBar(load)
    const classes = ['tri-pmc-bar']
    if (raceDates.has(day.date)) classes.push('tri-pmc-bar--race')
    if (i === n - 1) classes.push('tri-pmc-bar--now')
    s.appendChild(
      svg('rect', {
        x: (x(i) - bw / 2).toFixed(2),
        y: by.toFixed(2),
        width: bw.toFixed(2),
        height: (PMC_BAR_BOT - by).toFixed(2),
        class: classes.join(' '),
      }),
    )
  }
  s.appendChild(
    svg('line', { x1: 0, y1: PMC_BAR_BOT, x2: ANA_W, y2: PMC_BAR_BOT, class: 'tri-pmc-baseline' }),
  )
  s.appendChild(svg('line', { x1: nowX, y1: 0, x2: nowX, y2: PMC_BAR_BOT, class: 'tri-pmc-now' }))
  const hits = svg('g', { class: 'tri-pmc-hit-layer' })
  for (let i = 0; i < N; i++) {
    const left = i === 0 ? 0 : (x(i - 1) + x(i)) / 2
    const right = i === N - 1 ? ANA_W : (x(i) + x(i + 1)) / 2
    hits.appendChild(
      svg('rect', {
        x: left.toFixed(2),
        y: 0,
        width: Math.max(0.1, right - left).toFixed(2),
        height: PMC_H,
        class: 'tri-pmc-hit',
        'data-i': i,
      }),
    )
  }
  s.appendChild(svg('path', { d: polyD(tsbPts), class: 'tri-pmc-l-form' }))
  s.appendChild(svg('path', { d: polyD(atlPts), class: 'tri-pmc-l-fat' }))
  s.appendChild(svg('path', { d: polyD(ctlPts), class: 'tri-pmc-l-fit' }))
  const tsbProj = svg('path', {
    d: projPath(daily[n - 1].tsb, p => p.tsb, yTsb),
    class: 'tri-pmc-l-form tri-pmc-proj',
  })
  const atlProj = svg('path', {
    d: projPath(daily[n - 1].atl, p => p.atl, yFit),
    class: 'tri-pmc-l-fat tri-pmc-proj',
  })
  const ctlProj = svg('path', {
    d: projPath(daily[n - 1].ctl, p => p.ctl, yFit),
    class: 'tri-pmc-l-fit tri-pmc-proj',
  })
  s.append(tsbProj, atlProj, ctlProj)
  const cursor = svg('line', { x1: 0, y1: 0, x2: 0, y2: PMC_H, class: 'tri-ana-cursor' })
  s.appendChild(cursor)
  s.appendChild(hits)

  const frame = axisFrame(
    createDomFactory(context.presentation),
    s,
    [maxFit, maxFit / 2, 0].map(gv => ({ label: String(Math.round(gv)), vbY: yFit(gv) })),
    PMC_H,
    [
      ...monthTicks(
        context.formatter,
        daily.map(d => d.date),
        i => x(i),
      ),
      { label: context.formatter.text('today'), pct: nowX, cls: 'tri-pmc-xt--now' },
      { label: `+${H}d`, pct: 100, cls: 'tri-pmc-xt--end' },
    ],
    true,
    { top: PMC_TOP, bottom: PMC_BOT },
  ) as HTMLElement
  const readoutEl = el('div', 'tri-chart-readout')
  const stage = frame.querySelector<HTMLElement>('.tri-cax-stage')
  for (let i = 0; i < n; i++) {
    const day = daily[i]
    if (!raceDates.has(day.date)) continue
    stage?.appendChild(
      el('span', 'tri-pmc-race-marker', undefined, {
        style: `left:${((x(i) / ANA_W) * 100).toFixed(2)}%;top:${((yFit(day.ctl) / PMC_H) * 100).toFixed(2)}%`,
        'data-date': day.date,
        'aria-hidden': 'true',
      }),
    )
  }
  stage?.appendChild(readoutEl)
  block.appendChild(frame)

  const ctrl = el('div', 'tri-pmc-ctrl')
  const slider = el('input', 'tri-pmc-load') as HTMLInputElement
  slider.type = 'range'
  slider.min = '0'
  slider.max = String(LOAD_MAX)
  slider.step = '5'
  slider.value = String(futLoad)
  slider.setAttribute('aria-label', 'assumed future daily load')
  const ctrlLab = el('span', 'tri-pmc-ctrl-lab')
  ctrl.append(
    el('span', 'tri-pmc-ctrl-k', context.formatter.text('projected load')),
    slider,
    ctrlLab,
  )
  block.appendChild(ctrl)

  const sportSeries: { sp: Sport; get: (d: DailyPoint) => number }[] = [
    { sp: 'swim', get: d => d.swimCtl },
    { sp: 'bike', get: d => d.bikeCtl },
    { sp: 'run', get: d => d.runCtl },
  ]
  const sportCap = el('div', 'tri-elev-cap')
  sportCap.appendChild(el('span', 'tri-ana-k', context.formatter.text('fitness')))
  for (const { sp, get } of sportSeries) {
    const th = bySport(data.thresholds, sp)
    const stale = th == null ? '—' : th.staleDays === 0 ? 'today' : `${th.staleDays}d ago`
    const leg = el('span', `tri-ana-leg tri-leg-${sp}`)
    leg.append(
      buildIcon(context.presentation, sp),
      el('span', 'tri-ana-k', `${Math.round(get(daily[n - 1]))} · ${stale}`),
    )
    sportCap.appendChild(leg)
  }
  block.appendChild(sportCap)

  const legendRow = (cls: string, name: string, val: string): HTMLElement => {
    const row = el('div', `tri-pmc-leg ${cls}`)
    row.append(
      el('span', 'tri-pmc-dot'),
      el('span', 'tri-pmc-leg-v', val),
      el('span', 'tri-pmc-leg-k', context.formatter.text(name)),
    )
    return row
  }
  const entryRow = (a: ActivitySummary): HTMLElement => {
    const row = el('div', 'tri-pmc-entry')
    row.append(
      el('span', 'tri-pmc-entry-n', a.name || a.sport),
      el('span', `tri-pmc-entry-s tri-leg-${a.sport}`, context.formatter.text(a.sport)),
      el('span', 'tri-pmc-entry-d', dist(context.presentation, a.distanceKm, a.sport)),
    )
    return row
  }
  const renderLegend = (i: number): void => {
    const proj = i >= n
    const p = proj ? projSeries[Math.min(H - 1, i - n)] : daily[i]
    const date = proj ? projDate(i - n + 1) : daily[i].date
    const entries = proj
      ? []
      : [...(activitiesByDate.get(date) ?? [])].sort((a, b) => b.load - a.load)
    const entryList = el('div', 'tri-pmc-entries')
    if (!proj) {
      if (entries.length === 0)
        entryList.appendChild(
          el('div', 'tri-pmc-entry tri-pmc-entry--empty', context.formatter.text('no activity')),
        )
      else for (const activity of entries.slice(0, 3)) entryList.appendChild(entryRow(activity))
    }
    const metricGrid = el('div', 'tri-pmc-leg-grid')
    metricGrid.append(
      legendRow('tri-pmc-fit', 'fitness', String(Math.round(p.ctl))),
      legendRow('tri-pmc-fat', 'fatigue', String(Math.round(p.atl))),
      legendRow('tri-pmc-form', 'form', signed(Math.round(p.tsb))),
    )
    readoutEl.replaceChildren(
      el(
        'span',
        'tri-pmc-leg-date',
        `${context.formatter.shortDate(date)}${proj ? ' · proj' : ''}`,
      ),
      metricGrid,
    )
    if (!proj) {
      const d = daily[i]
      readoutEl.append(
        el(
          'span',
          'tri-pmc-leg-load',
          `${context.formatter.text('swim')} ${Math.round(d.swimCtl)} · ${context.formatter.text('bike')} ${Math.round(d.bikeCtl)} · ${context.formatter.text('run')} ${Math.round(d.runCtl)}`,
        ),
        el(
          'span',
          'tri-pmc-leg-load',
          `${context.formatter.text('training impulse')} ${Math.round(d.load)}`,
        ),
      )
      readoutEl.append(entryList)
    }
  }
  const setCtrlLab = (): void => {
    const lp = projSeries[H - 1]
    ctrlLab.textContent = `${futLoad}/day → ${H}d: ${context.formatter.text('fitness')} ${Math.round(lp.ctl)} · ${context.formatter.text('form')} ${signed(Math.round(lp.tsb))}`
  }
  const focusIndex = (i: number, hover: boolean): void => {
    const activeIndex = Math.round(clampN(i, 0, N - 1))
    const cx = x(activeIndex).toFixed(2)
    cursor.setAttribute('x1', cx)
    cursor.setAttribute('x2', cx)
    readoutEl.style.setProperty(
      '--tri-pmc-readout-x',
      `${clampN((x(activeIndex) / ANA_W) * 100, 14, 86).toFixed(2)}%`,
    )
    renderLegend(activeIndex)
    block.classList.toggle('tri-chart--hover', hover)
  }
  const indexAt = (event: MouseEvent): number => {
    const rect = s.getBoundingClientRect()
    return Math.round(clampN((event.clientX - rect.left) / rect.width, 0, 1) * (N - 1))
  }
  setCtrlLab()
  focusIndex(n - 1, false)
  const onSliderInput = (): void => {
    model = updatePerformance(
      model,
      { type: 'set-load', load: Number(slider.value) },
      performanceBounds,
    )
    futLoad = model.futureDailyLoad
    projSeries = project(futLoad)
    tsbProj.setAttribute(
      'd',
      projPath(daily[n - 1].tsb, p => p.tsb, yTsb),
    )
    atlProj.setAttribute(
      'd',
      projPath(daily[n - 1].atl, p => p.atl, yFit),
    )
    ctlProj.setAttribute(
      'd',
      projPath(daily[n - 1].ctl, p => p.ctl, yFit),
    )
    setCtrlLab()
    focusIndex(model.activeIndex, block.classList.contains('tri-chart--hover'))
  }

  const onMove = (event: MouseEvent): void => {
    model = updatePerformance(model, { type: 'hover', index: indexAt(event) }, performanceBounds)
    focusIndex(model.activeIndex, true)
  }
  const onLeave = (): void => {
    model = updatePerformance(model, { type: 'leave' }, performanceBounds)
    focusIndex(model.activeIndex, model.lockedIndex != null)
  }
  const onClick = (event: MouseEvent): void => {
    model = updatePerformance(
      model,
      { type: 'toggle-lock', index: indexAt(event) },
      performanceBounds,
    )
    block.classList.toggle('tri-chart--locked', model.lockedIndex != null)
    focusIndex(model.activeIndex, model.lockedIndex != null)
  }
  return {
    element: block,
    mount: () => {
      slider.addEventListener('input', onSliderInput)
      s.addEventListener('mousemove', onMove)
      s.addEventListener('mouseleave', onLeave)
      s.addEventListener('click', onClick)
      return () => {
        slider.removeEventListener('input', onSliderInput)
        s.removeEventListener('mousemove', onMove)
        s.removeEventListener('mouseleave', onLeave)
        s.removeEventListener('click', onClick)
      }
    },
  }
}

export type WkKind = 'load' | 'effort'

export const WKT_H = 34

export const WKT_TOP = 4

export const WKT_BOT = WKT_H - 4

export const WKT_ACTS = 4

export const wkVal = (w: Analytics['weekly'][number], kind: WkKind): number =>
  kind === 'load' ? w.load : w.effort

export interface WkTrendRow {
  week: Analytics['weekly'][number]
  sourceIndex: number
  value: number
  band: [number, number] | null
}

export const wkTrendRows = (data: Analytics, kind: WkKind): WkTrendRow[] =>
  data.weekly.map((week, sourceIndex) => ({
    week,
    sourceIndex,
    value: wkVal(week, kind),
    band: kind === 'load' ? week.loadRange : week.effortRange,
  }))

export const wkDates = (weekStart: string): string[] =>
  Array.from({ length: 7 }, (_, k) =>
    new Date(Date.parse(`${weekStart}T00:00:00Z`) + k * 86400000).toISOString().slice(0, 10),
  )

export const wkDayLetter = (context: TriathlonContext, iso: string): string => {
  const day = new Date(`${iso}T00:00:00Z`).getUTCDay()
  return context.formatter.weekdayNarrow(day)
}

export const renderWkDetail = (
  block: HTMLElement,
  data: Analytics,
  kind: WkKind,
  i: number,
  context: TriathlonContext,
): void => {
  const host = block.querySelector<HTMLElement>('.tri-wkdetail')
  const row = wkTrendRows(data, kind)[i]
  if (!host || !row || host.dataset.week === row.week.weekStart) return
  host.dataset.week = row.week.weekStart
  const w = row.week
  const band = row.band
  const days = wkDates(w.weekStart)
  const head = el('div', 'tri-wkdetail-head')
  head.append(
    el('span', 'tri-wkdetail-num', String(Math.round(row.value))),
    el(
      'span',
      'tri-wkdetail-range',
      `${context.formatter.shortDate(days[0])} – ${context.formatter.shortDate(days[6])}`,
    ),
  )
  if (band) {
    const state = row.value > band[1] ? 'above' : row.value < band[0] ? 'below' : 'in'
    head.appendChild(
      el(
        'span',
        `tri-wkdetail-state tri-wkdetail-state--${state}`,
        context.formatter.text(`${state} range`),
      ),
    )
  }
  const byDate = new Map<string, DailyPoint>()
  for (const d of data.daily) byDate.set(d.date, d)
  const dayVals = days.map(date => {
    const d = byDate.get(date)
    return d ? (kind === 'load' ? d.load : d.effort) : 0
  })
  let dayMax = 1
  for (const dv of dayVals) if (dv > dayMax) dayMax = dv
  const grid = el('div', 'tri-wkdetail-days')
  days.forEach((date, k) => {
    const col = el('span', 'tri-wkdetail-day')
    const track = el('span', 'tri-wkdetail-track')
    if (dayVals[k] > 0) {
      const fill = el('span', 'tri-wkdetail-fill')
      fill.style.height = `${Math.max(6, (dayVals[k] / dayMax) * 100).toFixed(1)}%`
      track.appendChild(fill)
    }
    col.append(track, el('span', 'tri-wkdetail-dl', wkDayLetter(context, date)))
    grid.appendChild(col)
  })
  const stats = mathK(
    'tri-ana-k tri-wkdetail-stats',
    `${w.sessions}$\\times$ $\\cdot$ ${fmtKm(context.presentation, w.km)} $\\cdot$ ${w.hours.toFixed(1)}h`,
  )
  const actsBox = el('div', 'tri-wkdetail-acts')
  const acts = data.activities
    .filter(a => a.date >= days[0] && a.date <= days[6])
    .map(a => ({ a, v: kind === 'load' ? a.load : a.effort }))
    .sort((p, q) => (q.v ?? -1) - (p.v ?? -1))
  for (const { a, v } of acts.slice(0, WKT_ACTS)) {
    const row = el('div', 'tri-wkdetail-act')
    row.append(
      buildIconLeg(context.formatter, a.sport),
      el('span', 'tri-wkdetail-act-name', a.name || a.sport),
      el('span', 'tri-wkdetail-act-t', hms(a.movingTimeS)),
      el('span', 'tri-wkdetail-act-v', v != null && v > 0 ? String(Math.round(v)) : '—'),
    )
    actsBox.appendChild(row)
  }
  if (acts.length > WKT_ACTS)
    actsBox.appendChild(
      el('div', 'tri-wkdetail-act tri-wkdetail-act--more', `+${acts.length - WKT_ACTS}`),
    )
  host.replaceChildren(head, grid, stats, actsBox)
}

export const buildWeekTrend = (
  data: Analytics,
  kind: WkKind,
  context: TriathlonContext,
): HTMLElement => {
  const block = el('div', kind === 'load' ? 'tri-ana-weekly' : 'tri-ana-effort')
  block.appendChild(
    kind === 'load'
      ? anaTitle(context.formatter, 'weekly load', 'load')
      : anaTitle(context.formatter, 'relative effort', 'effort'),
  )
  const rows = wkTrendRows(data, kind)
  if (kind === 'load' ? !rows.length : !rows.some(row => row.value > 0)) {
    block.appendChild(
      el(
        'div',
        'tri-ana-empty',
        context.formatter.text(kind === 'load' ? 'no weeks' : 'no effort logged'),
      ),
    )
    return block
  }
  const n = rows.length
  const vals = rows.map(row => row.value)
  const bands = rows.map(row => row.band)
  let mx = 1
  for (const v of vals) if (v > mx) mx = v
  for (const b of bands) if (b && b[1] > mx) mx = b[1]
  const yMax = Math.ceil(mx)
  const x = (i: number): number => weeklyChartX(i, n) * ANA_W
  const y = (v: number): number => WKT_BOT - (v / yMax) * (WKT_BOT - WKT_TOP)
  const s = svg('svg', {
    class: 'tri-ana-svg tri-wkt-svg',
    viewBox: `0 0 ${ANA_W} ${WKT_H}`,
    preserveAspectRatio: 'none',
  })
  let run: number[] = []
  const flushBand = (): void => {
    if (run.length >= 2) {
      const top = run.map(i => [x(i), y(bands[i]![1])] as [number, number])
      const btm = [...run].reverse().map(i => [x(i), y(bands[i]![0])] as [number, number])
      s.appendChild(svg('path', { d: `${polyD([...top, ...btm])} Z`, class: 'tri-wkt-band' }))
    }
    run = []
  }
  bands.forEach((b, i) => {
    if (b) run.push(i)
    else flushBand()
  })
  flushBand()
  const pred = bands[n - 1]
  if (pred)
    s.appendChild(
      svg('line', {
        x1: x(n - 1).toFixed(2),
        y1: 0,
        x2: x(n - 1).toFixed(2),
        y2: WKT_H,
        class: 'tri-wkt-current',
      }),
    )
  s.appendChild(svg('line', { x1: 0, y1: 0, x2: 0, y2: WKT_H, class: 'tri-ana-cursor' }))
  vals.forEach((v, i) => {
    const d = `M ${x(i).toFixed(2)} ${y(v).toFixed(2)} l 0.01 0`
    const g = svg('g', {
      class: i === n - 1 ? 'tri-wkt-pt tri-wkt-pt--now tri-wkt-pt--sel' : 'tri-wkt-pt',
      'data-week': i,
      'data-source-index': rows[i].sourceIndex,
      'data-week-start': rows[i].week.weekStart,
    })
    g.appendChild(svg('path', { d, class: 'tri-wkt-halo' }))
    g.appendChild(svg('path', { d, class: 'tri-wkt-o' }))
    if (i !== n - 1) g.appendChild(svg('path', { d, class: 'tri-wkt-i' }))
    s.appendChild(g)
  })
  const frame = axisFrame(
    createDomFactory(context.presentation),
    s,
    [yMax, yMax / 2, 0].map(v => ({ label: String(Math.round(v)), vbY: y(v) })),
    WKT_H,
    monthTicks(
      context.formatter,
      rows.map(row => row.week.weekStart),
      i => weeklyChartX(i, n) * 100,
    ),
  ) as HTMLElement
  if (pred) {
    const stage = frame.querySelector<HTMLElement>('.tri-cax-stage')
    for (const bound of [pred[1], pred[0]])
      stage?.appendChild(
        el('span', 'tri-wkt-pred', String(Math.round(bound)), {
          style: `top:${((y(bound) / WKT_H) * 100).toFixed(1)}%`,
        }),
      )
  }
  block.appendChild(frame)
  const wrap = el('div', 'tri-wkdetail-wrap tri-wkdetail-wrap--open')
  wrap.appendChild(el('div', 'tri-wkdetail'))
  block.appendChild(wrap)
  renderWkDetail(block, data, kind, n - 1, context)
  return block
}

export const buildWeekly = (data: Analytics, context: TriathlonContext): HTMLElement => {
  const block = buildWeekTrend(data, 'load', context)
  const wk = data.weekly
  if (!wk.length) return block
  let mx = 1
  for (const w of wk) if (w.load > mx) mx = w.load
  const active = wk.filter(w => w.load > 0).length
  const last = wk[wk.length - 1]
  const prev = wk.length >= 2 ? wk[wk.length - 2] : null
  const vol = data.calibration.volume
  const deltaClass =
    vol.deltaLoad > 0 ? 'tri-dir-up' : vol.deltaLoad < 0 ? 'tri-dir-down' : 'tri-dir-flat'
  const cap = el('div', 'tri-elev-cap tri-wk-cap')
  const statRow = el('div', 'tri-wk-cap-row')
  statRow.append(
    el('span', 'tri-ana-k', `${active} ${context.formatter.text('active wk')}`),
    el('span', 'tri-ana-k', `${context.formatter.text('peak')} ${Math.round(mx)}/wk`),
    mathK(
      'tri-ana-k',
      `28d ${fmtKm(context.presentation, vol.currentKm)} $\\cdot$ ${vol.currentHours.toFixed(1)}h`,
    ),
  )
  cap.appendChild(statRow)
  const deltaRow = el('div', 'tri-wk-cap-row')
  deltaRow.appendChild(
    mathK(
      `tri-ana-k ${deltaClass}`,
      `$\\Delta$ ${fmtSignedKm(context.presentation, vol.deltaKm)} $\\cdot$ ${signedFixed(vol.deltaHours, 1)}h $\\cdot$ ${signedFixed(vol.deltaLoad, 0)} load`,
    ),
  )
  if (prev)
    deltaRow.appendChild(
      mathK(
        'tri-ana-k',
        `wk $\\Delta$ ${fmtSignedKm(context.presentation, last.km - prev.km)} $\\cdot$ ${signedFixed(last.hours - prev.hours, 1)}h`,
      ),
    )
  cap.appendChild(deltaRow)
  const legRow = el('div', 'tri-wk-cap-row tri-wk-cap-legs')
  for (const sport of vol.sports) {
    if (sport.currentKm <= 0 && sport.previousKm <= 0) continue
    const leg = el('span', `tri-ana-leg tri-leg-${sport.sport}`)
    leg.append(
      buildIcon(context.presentation, sport.sport),
      el(
        'span',
        'tri-ana-k',
        `${fmtKm(context.presentation, sport.currentKm)} (${fmtSignedKm(context.presentation, sport.deltaKm)})`,
      ),
    )
    legRow.appendChild(leg)
  }
  if (legRow.childElementCount) cap.appendChild(legRow)
  block.appendChild(cap)
  return block
}
