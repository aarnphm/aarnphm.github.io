import type { Analytics } from '../../../../plugins/stores/analytics'
import type { OuraDayDetail } from '../../../../plugins/stores/oura'
import type { OuraSeries } from '../../../../plugins/stores/oura'
import type { AxisXTick } from '../../../../util/triathlon-card'
import type { TriathlonContext } from '../../runtime/context'
import type { TriathlonFormatter } from '../../runtime/formatter'
import { axisFrame } from '../../../../util/triathlon-card'
import { createDomFactory } from '../../runtime/dom'
import { el } from '../../runtime/dom'
import { mathK } from '../../runtime/dom'
import { svg } from '../../runtime/dom'
import { ANA_H } from '../shared'
import { ANA_W } from '../shared'
import { anaTitle } from '../shared'
import { clampN } from '../shared'
import { hms } from '../shared'
import { markGloss } from '../shared'
import { monthTicks } from '../shared'
import { polyD } from '../shared'
import { missingRuns } from './body'
import { segRuns } from './body'

export const buildRecoveryChart = (data: Analytics, context: TriathlonContext): HTMLElement => {
  const block = el('div', 'tri-ana-recovery')
  block.appendChild(anaTitle(context.formatter, 'recovery · hrv · rhr', 'hrv'))
  const rec = data.recovery
  if (!rec.series.length) {
    block.appendChild(el('div', 'tri-ana-empty', context.formatter.text('no recovery data')))
    return block
  }
  if (rec.flags.length) {
    const flags = el('div', 'tri-rec-flags')
    for (const f of rec.flags) {
      const row = el('div', `tri-rec-flag tri-flag--${f.severity}`)
      row.appendChild(el('span', 'tri-rec-dot'))
      row.appendChild(markGloss(el('span', 'tri-rec-flag-label', f.label), f.metric))
      row.appendChild(el('span', 'tri-rec-flag-detail', f.detail))
      flags.appendChild(row)
    }
    block.appendChild(flags)
  }
  const n = rec.series.length
  if (rec.status !== 'building' && n > 1) {
    const recLeg = (cls: string, name: string): HTMLElement => {
      const w = el('span', 'tri-rec-legitem')
      w.append(
        el('span', `tri-rec-legdot ${cls}`),
        el('span', 'tri-rec-legname', context.formatter.text(name)),
      )
      return w
    }
    const legend = el('div', 'tri-rec-legend')
    legend.append(recLeg('tri-rec-leg-hrv', 'hrv'), recLeg('tri-rec-leg-rhr', 'rhr'))
    block.appendChild(legend)
    const x = (i: number): number => (i / (n - 1)) * ANA_W
    const yZ = (z: number): number => 15 - (clampN(z, -3, 3) / 3) * 12
    const s = svg('svg', {
      class: 'tri-ana-svg tri-rec-svg',
      viewBox: `0 0 ${ANA_W} ${ANA_H}`,
      preserveAspectRatio: 'none',
    })
    s.appendChild(
      svg('rect', { x: 0, y: yZ(1), width: ANA_W, height: yZ(-1) - yZ(1), class: 'tri-rec-band' }),
    )
    s.appendChild(svg('line', { x1: 0, y1: yZ(0), x2: ANA_W, y2: yZ(0), class: 'tri-ana-zero' }))
    s.appendChild(svg('line', { x1: 0, y1: yZ(-1), x2: ANA_W, y2: yZ(-1), class: 'tri-rec-zline' }))
    s.appendChild(
      svg('line', {
        x1: 0,
        y1: yZ(-2),
        x2: ANA_W,
        y2: yZ(-2),
        class: 'tri-rec-zline tri-rec-zline--alert',
      }),
    )
    for (const seg of missingRuns(rec.series, d => d.hrvZ, x, yZ))
      s.appendChild(svg('path', { d: polyD(seg), class: 'tri-rec-hrv tri-rec-missing' }))
    for (const seg of missingRuns(rec.series, d => (d.rhrZ == null ? null : -d.rhrZ), x, yZ))
      s.appendChild(svg('path', { d: polyD(seg), class: 'tri-rec-rhr tri-rec-missing' }))
    for (const seg of segRuns(rec.series, d => d.hrvZ, x, yZ))
      s.appendChild(svg('path', { d: polyD(seg), class: 'tri-rec-hrv' }))
    for (const seg of segRuns(rec.series, d => (d.rhrZ == null ? null : -d.rhrZ), x, yZ))
      s.appendChild(svg('path', { d: polyD(seg), class: 'tri-rec-rhr' }))
    s.appendChild(svg('line', { x1: ANA_W, y1: 0, x2: ANA_W, y2: ANA_H, class: 'tri-pmc-now' }))
    s.appendChild(svg('line', { x1: 0, y1: 0, x2: 0, y2: ANA_H, class: 'tri-ana-cursor' }))
    block.appendChild(
      axisFrame(
        createDomFactory(context.presentation),
        s,
        [
          { label: '+2σ', vbY: yZ(2) },
          { label: '0', vbY: yZ(0) },
          { label: '-2σ', vbY: yZ(-2) },
        ],
        ANA_H,
        monthTicks(
          context.formatter,
          rec.series.map(d => d.date),
          i => (i / (n - 1)) * ANA_W,
        ),
        false,
      ),
    )
    block.appendChild(el('div', 'tri-chart-readout'))
  }
  const t = rec.thresholds
  const sevCls = (cond: boolean | null, alert: boolean | null): string =>
    alert ? 'tri-flag--alert' : cond ? 'tri-flag--watch' : ''
  const hrvCls = sevCls(
    rec.hrvZ != null && rec.hrvZ <= t.hrvWatchZ,
    rec.hrvZ != null && rec.hrvZ <= t.hrvAlertZ,
  )
  const rhrCls = sevCls(
    rec.rhrZ != null && rec.rhrZ >= t.rhrWatchZ,
    rec.rhrZ != null && rec.rhrZ >= t.rhrAlertZ,
  )
  const tmpCls = sevCls(
    rec.tempDevLatest != null && rec.tempDevLatest >= t.tempWatchC,
    rec.tempDevLatest != null && rec.tempDevLatest >= t.tempAlertC,
  )
  const rdyCls =
    rec.readinessLatest != null && rec.readinessLatest < t.readinessFloor ? 'tri-flag--watch' : ''
  const cap = el('div', 'tri-elev-cap')
  cap.append(
    markGloss(
      el(
        'span',
        `tri-ana-k ${hrvCls}`.trim(),
        rec.hrvLatest != null ? `HRV ${Math.round(rec.hrvLatest)} ms` : 'HRV —',
      ),
      'hrv',
    ),
    markGloss(
      el(
        'span',
        `tri-ana-k ${rhrCls}`.trim(),
        rec.rhrLatest != null ? `RHR ${Math.round(rec.rhrLatest)}` : 'RHR —',
      ),
      'rhr',
    ),
    markGloss(
      el(
        'span',
        `tri-ana-k ${rdyCls}`.trim(),
        rec.readinessLatest != null
          ? `readiness ${Math.round(rec.readinessLatest)}`
          : 'readiness —',
      ),
      'oreadiness',
    ),
    markGloss(
      mathK(
        `tri-ana-k ${tmpCls}`.trim(),
        rec.tempDevLatest != null
          ? `temp ${rec.tempDevLatest >= 0 ? '+' : ''}${rec.tempDevLatest.toFixed(1)}$^\\circ\\mathrm{C}$`
          : 'temp —',
      ),
      'tempdev',
    ),
  )
  if (rec.status !== 'firm')
    cap.appendChild(
      el('span', 'tri-ana-k', `${context.formatter.text('baseline')} ${rec.baselineDays}/14`),
    )
  block.appendChild(cap)
  return block
}

export const wallMin = (iso: string): number =>
  Number(iso.slice(11, 13)) * 60 + Number(iso.slice(14, 16))

export const wallClock = (min: number): string => {
  const m = ((Math.round(min) % 1440) + 1440) % 1440
  return `${Math.floor(m / 60)
    .toString()
    .padStart(2, '0')}:${(m % 60).toString().padStart(2, '0')}`
}

export const hourTicks = (
  startIso: string,
  intervalS: number,
  count: number,
  pctOf: (i: number) => number,
): AxisXTick[] => {
  const out: AxisXTick[] = []
  const startS = wallMin(startIso) * 60
  let bucket = Math.floor(startS / 7200)
  for (let i = 1; i < count; i++) {
    const b = Math.floor((startS + i * intervalS) / 7200)
    if (b === bucket) continue
    bucket = b
    out.push({ label: wallClock((b * 7200) / 60), pct: pctOf(i) })
  }
  return out
}

export const OURA_STAGE: Record<string, { key: string; lane: number }> = {
  '4': { key: 'awake', lane: 0 },
  '3': { key: 'rem', lane: 1 },
  '2': { key: 'light', lane: 2 },
  '1': { key: 'deep', lane: 3 },
}

export const ouraScoreCls = (v: number): string =>
  v < 70 ? 'tri-flag--alert' : v < 85 ? 'tri-flag--watch' : 'tri-flag--info'

export const ouraContribGroup = (
  formatter: TriathlonFormatter,
  title: string,
  contrib: Record<string, number | null> | null,
): HTMLElement | null => {
  if (!contrib) return null
  const rows = Object.entries(contrib).filter((e): e is [string, number] => e[1] != null)
  if (!rows.length) return null
  const g = el('div', 'tri-sleep-contrib')
  g.appendChild(el('div', 'tri-ana-block-title', formatter.text(title)))
  for (const [key, v] of rows) {
    const row = el('div', 'tri-sleep-contrib-row')
    const bar = el('div', 'tri-sleep-contrib-bar')
    const fill = el(
      'div',
      v >= 70 ? 'tri-sleep-contrib-fill' : 'tri-sleep-contrib-fill tri-sleep-contrib-fill--low',
    )
    fill.style.width = `${clampN(v, 0, 100)}%`
    bar.appendChild(fill)
    row.append(
      el('span', 'tri-sleep-contrib-label', formatter.text(key.replace(/_/g, ' '))),
      bar,
      el('span', 'tri-sleep-contrib-val', String(Math.round(v))),
    )
    g.appendChild(row)
  }
  return g
}

export const buildHypnogram = (
  formatter: TriathlonFormatter,
  d: OuraDayDetail,
): HTMLElement | null => {
  const phase = d.phase5Min
  if (!phase || !phase.length || !d.bedtimeStart) return null
  const len = phase.length
  const H = 16
  const wrap = el('div', 'tri-sleep-chart tri-sleep-hyp', undefined, {
    'data-oura-series': 'stages',
  })
  wrap.appendChild(el('div', 'tri-ana-block-title', formatter.text('sleep stages')))
  const s = svg('svg', {
    class: 'tri-ana-svg tri-hyp-svg',
    viewBox: `0 0 ${len} ${H}`,
    preserveAspectRatio: 'none',
  })
  let i = 0
  while (i < len) {
    const c = phase[i]
    let j = i + 1
    while (j < len && phase[j] === c) j++
    const st = OURA_STAGE[c]
    if (st)
      s.appendChild(
        svg('rect', {
          x: i,
          y: st.lane * 4 + 0.3,
          width: j - i,
          height: 3.4,
          class: `tri-hyp--${st.key}`,
        }),
      )
    i = j
  }
  const cursor = svg('line', { x1: 0, y1: 0, x2: 0, y2: H, class: 'tri-ana-cursor' })
  s.appendChild(cursor)
  wrap.appendChild(
    axisFrame(
      createDomFactory(formatter.presentation),
      s,
      [
        { label: formatter.text('awake'), vbY: 2 },
        { label: formatter.text('rem'), vbY: 6 },
        { label: formatter.text('light'), vbY: 10 },
        { label: formatter.text('deep'), vbY: 14 },
      ],
      H,
      hourTicks(d.bedtimeStart, 300, len, k => (k / len) * 100),
      false,
    ),
  )
  const readout = el('div', 'tri-chart-readout')
  wrap.appendChild(readout)
  const cap = el('div', 'tri-elev-cap')
  const durs: [string, number | null][] = [
    ['deep', d.deepS],
    ['light', d.lightS],
    ['rem', d.remS],
    ['awake', d.awakeS],
  ]
  for (const [name, sec] of durs)
    if (sec != null) cap.appendChild(el('span', 'tri-ana-k', `${formatter.text(name)} ${hms(sec)}`))
  wrap.appendChild(cap)
  return wrap
}

export const buildOuraSeriesChart = (
  formatter: TriathlonFormatter,
  key: 'hrv' | 'hr',
  title: string,
  series: OuraSeries | null,
  strokeCls: string,
): HTMLElement | null => {
  if (!series || series.items.length < 2) return null
  const items = series.items
  const n = items.length
  const vals = items.filter((v): v is number => v != null)
  if (vals.length < 2) return null
  let lo = Infinity
  let hi = -Infinity
  for (const v of vals) {
    if (v < lo) lo = v
    if (v > hi) hi = v
  }
  const pad = Math.max((hi - lo) * 0.1, 1)
  const mn = lo - pad
  const mx = hi + pad
  const x = (i: number): number => (i / (n - 1)) * ANA_W
  const y = (v: number): number => ANA_H - 2 - ((v - mn) / (mx - mn)) * (ANA_H - 4)
  const wrap = el('div', 'tri-sleep-chart', undefined, { 'data-oura-series': key })
  wrap.appendChild(el('div', 'tri-ana-block-title', formatter.text(title)))
  const s = svg('svg', {
    class: 'tri-ana-svg tri-sleep-line-svg',
    viewBox: `0 0 ${ANA_W} ${ANA_H}`,
    preserveAspectRatio: 'none',
  })
  const avg = vals.reduce((a, b) => a + b, 0) / vals.length
  s.appendChild(svg('line', { x1: 0, y1: y(avg), x2: ANA_W, y2: y(avg), class: 'tri-rec-target' }))
  for (const seg of segRuns(items, v => v, x, y))
    s.appendChild(svg('path', { d: polyD(seg), class: strokeCls }))
  const cursor = svg('line', { x1: 0, y1: 0, x2: 0, y2: ANA_H, class: 'tri-ana-cursor' })
  s.appendChild(cursor)
  wrap.appendChild(
    axisFrame(
      createDomFactory(formatter.presentation),
      s,
      [
        { label: String(Math.round(hi)), vbY: y(hi) },
        { label: String(Math.round(lo)), vbY: y(lo) },
      ],
      ANA_H,
      hourTicks(series.startTs, series.intervalS, n, k => (k / (n - 1)) * 100),
    ),
  )
  const readout = el('div', 'tri-chart-readout')
  wrap.appendChild(readout)
  return wrap
}

export const SLEEPLESS_ROCKY_FRAMES = [1, 2, 3, 0].map(
  c => `/static/landing/rocky-monomyth/frames/rocky-monomyth-r5-c${c}.webp`,
)

export const buildSleeplessRock = (caption: string): HTMLElement => {
  const wrap = el('div', 'tri-sleep-empty')
  const stage = el('div', 'tri-sleep-rocky', undefined, { 'aria-hidden': 'true' })
  for (const src of SLEEPLESS_ROCKY_FRAMES)
    stage.appendChild(
      el('img', 'tri-rocky-frame', undefined, {
        src,
        alt: '',
        width: '192',
        height: '208',
        decoding: 'async',
        draggable: 'false',
      }),
    )
  wrap.append(stage, el('div', 'tri-ana-empty', caption))
  return wrap
}

export const buildOuraDayDetail = (
  formatter: TriathlonFormatter,
  d: OuraDayDetail,
): HTMLElement => {
  const wrap = el('div', 'tri-sleep-day-body')
  const head = el('div', 'tri-sleep-day-head')
  const cap = el('div', 'tri-elev-cap')
  cap.appendChild(el('span', 'tri-ana-k tri-sleep-day-date', formatter.shortDate(d.date)))
  if (d.bedtimeStart)
    cap.appendChild(
      el('span', 'tri-ana-k', `${formatter.text('bedtime')} ${wallClock(wallMin(d.bedtimeStart))}`),
    )
  if (d.bedtimeEnd)
    cap.appendChild(
      el('span', 'tri-ana-k', `${formatter.text('wake-up')} ${wallClock(wallMin(d.bedtimeEnd))}`),
    )
  if (d.totalSleepS != null)
    cap.appendChild(el('span', 'tri-ana-k', `${formatter.text('sleep')} ${hms(d.totalSleepS)}`))
  if (d.efficiency != null)
    cap.appendChild(
      el('span', 'tri-ana-k', `${formatter.text('efficiency')} ${Math.round(d.efficiency)}%`),
    )
  if (d.latencyS != null)
    cap.appendChild(el('span', 'tri-ana-k', `${formatter.text('latency')} ${hms(d.latencyS)}`))
  if (d.lowestHr != null)
    cap.appendChild(
      el('span', 'tri-ana-k', `${formatter.text('lowest hr')} ${Math.round(d.lowestHr)}`),
    )
  if (d.avgBreath != null)
    cap.appendChild(
      el('span', 'tri-ana-k', `${formatter.text('breath')} ${d.avgBreath.toFixed(1)}`),
    )
  if (d.sleepScore != null)
    cap.appendChild(
      el(
        'span',
        `tri-ana-k ${ouraScoreCls(d.sleepScore)}`,
        `${formatter.text('sleep score')} ${Math.round(d.sleepScore)}`,
      ),
    )
  if (d.readinessScore != null)
    cap.appendChild(
      el(
        'span',
        `tri-ana-k ${ouraScoreCls(d.readinessScore)}`,
        `${formatter.text('readiness')} ${Math.round(d.readinessScore)}`,
      ),
    )
  head.appendChild(cap)
  const closeBtn = el('button', 'tri-sleep-day-close', undefined, {
    type: 'button',
    'aria-label': formatter.text('Close'),
    'data-site-cursor-close': '',
  })
  closeBtn.appendChild(
    el('span', undefined, '×', { 'aria-hidden': 'true', 'data-site-cursor-icon': '' }),
  )
  head.appendChild(closeBtn)
  wrap.appendChild(head)
  const hyp = buildHypnogram(formatter, d)
  const hrv = buildOuraSeriesChart(formatter, 'hrv', 'hrv', d.hrv, 'tri-rec-hrv')
  const hr = buildOuraSeriesChart(formatter, 'hr', 'resting heart rate', d.hr, 'tri-rec-rhr')
  if (!hyp && !hrv && !hr)
    wrap.appendChild(buildSleeplessRock(formatter.text('rock bottom — no sleep recorded')))
  const sleepContrib = ouraContribGroup(formatter, 'sleep score', d.sleepContrib)
  if (sleepContrib) wrap.appendChild(sleepContrib)
  const readyContrib = ouraContribGroup(formatter, 'readiness', d.readinessContrib)
  if (readyContrib) wrap.appendChild(readyContrib)
  if (hyp) wrap.appendChild(hyp)
  if (hrv) wrap.appendChild(hrv)
  if (hr) wrap.appendChild(hr)
  return wrap
}

export const buildSleep = (data: Analytics, context: TriathlonContext): HTMLElement => {
  const block = el('div', 'tri-ana-sleep')
  block.appendChild(anaTitle(context.formatter, 'sleep · debt', 'sleepdebt'))
  const rec = data.recovery
  const view = rec.series
  if (!view.some(d => d.sleepS != null)) {
    block.appendChild(el('div', 'tri-ana-empty', context.formatter.text('no sleep logged')))
    return block
  }
  const n = view.length
  const H = 32
  const bot = H - 0.5
  let maxS = rec.sleepTargetS
  for (const d of view) if (d.sleepS != null && d.sleepS > maxS) maxS = d.sleepS
  maxS = Math.ceil(maxS / 3600) * 3600
  const yBar = (sec: number): number => bot - (sec / maxS) * (H - 2)
  const s = svg('svg', {
    class: 'tri-ana-svg tri-sleep-svg',
    viewBox: `0 0 ${n} ${H}`,
    preserveAspectRatio: 'none',
  })
  s.appendChild(
    svg('line', {
      x1: 0,
      y1: yBar(rec.sleepTargetS),
      x2: n,
      y2: yBar(rec.sleepTargetS),
      class: 'tri-rec-target',
    }),
  )
  s.appendChild(
    svg('line', {
      x1: 0,
      y1: yBar(rec.thresholds.sleepFloorS),
      x2: n,
      y2: yBar(rec.thresholds.sleepFloorS),
      class: 'tri-rec-floor',
    }),
  )
  view.forEach((d, i) => {
    if (d.sleepS == null) {
      s.appendChild(
        svg('rect', { x: i + 0.35, y: bot - 0.5, width: 0.3, height: 0.5, class: 'tri-seg--rest' }),
      )
      return
    }
    const h = (d.sleepS / maxS) * (H - 2)
    const base = d.sleepS < rec.thresholds.sleepFloorS ? 'tri-seg--short' : 'tri-seg--sleep'
    const bar = svg('rect', {
      x: i + 0.2,
      y: bot - h,
      width: 0.6,
      height: h,
      class: base,
      'data-sleep-date': d.date,
    })
    s.appendChild(bar)
  })
  const ys = (sc: number): number => bot - (sc / 100) * (H - 2)
  for (const seg of segRuns(
    view,
    d => d.sleepScore,
    i => i + 0.5,
    ys,
  ))
    s.appendChild(svg('path', { d: polyD(seg), class: 'tri-rec-score' }))
  s.appendChild(svg('line', { x1: 0, y1: 0, x2: 0, y2: H, class: 'tri-ana-cursor' }))
  const yMaxHr = Math.round(maxS / 3600)
  block.appendChild(
    axisFrame(
      createDomFactory(context.presentation),
      s,
      [yMaxHr, yMaxHr / 2, 0].map(v => ({
        label: v === 0 ? '0' : `${v % 1 === 0 ? v : v.toFixed(1)}h`,
        vbY: yBar(v * 3600),
      })),
      H,
      monthTicks(
        context.formatter,
        view.map(d => d.date),
        i => ((i + 0.5) / n) * 100,
      ),
    ),
  )
  block.appendChild(el('div', 'tri-chart-readout'))
  const debtCls =
    rec.sleepDebtS >= rec.thresholds.sleepDebtAlertS
      ? 'tri-flag--alert'
      : rec.sleepDebtS >= rec.thresholds.sleepDebtWatchS
        ? 'tri-flag--watch'
        : ''
  const cap = el('div', 'tri-elev-cap')
  cap.append(
    el(
      'span',
      'tri-ana-k',
      rec.sleepLatestS != null
        ? `${context.formatter.text('sleep')} ${hms(rec.sleepLatestS)}`
        : `${context.formatter.text('sleep')} —`,
    ),
    el(
      'span',
      'tri-ana-k',
      rec.sleepBaselineS != null
        ? `${context.formatter.text('base')} ${hms(rec.sleepBaselineS)}`
        : `${context.formatter.text('base')} —`,
    ),
    markGloss(
      el(
        'span',
        `tri-ana-k ${debtCls}`.trim(),
        `${context.formatter.text('debt')} ${(rec.sleepDebtS / 3600).toFixed(1)} h`,
      ),
      'sleepdebt',
    ),
    el('span', 'tri-ana-k', `${context.formatter.text('target')} ${hms(rec.sleepTargetS)}`),
  )
  if (rec.shortSleepStreak >= 2)
    cap.appendChild(
      el(
        'span',
        `tri-ana-k tri-flag--${rec.shortSleepStreak >= 3 ? 'alert' : 'watch'}`,
        `${rec.shortSleepStreak} short`,
      ),
    )
  block.appendChild(cap)
  const dayWrap = el('div', 'tri-sleep-day')
  const dayInner = el('div', 'tri-sleep-day-inner')
  dayWrap.appendChild(dayInner)
  block.appendChild(dayWrap)
  return block
}
