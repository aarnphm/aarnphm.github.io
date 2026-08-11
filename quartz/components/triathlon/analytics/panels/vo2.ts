import type { Analytics } from '../../../../plugins/stores/analytics'
import type { Vo2LabProfile } from '../../../../plugins/stores/analytics'
import type { Vo2LabProfileSample } from '../../../../plugins/stores/analytics'
import type { Vo2LabProfileStats } from '../../../../plugins/stores/analytics'
import type { Vo2LabRecord } from '../../../../plugins/stores/analytics'
import type { Vo2LabTargetStep } from '../../../../plugins/stores/analytics'
import type { TriathlonContext } from '../../runtime/context'
import type { TriathlonFormatter } from '../../runtime/formatter'
import { axisFrame } from '../../../../util/triathlon-card'
import { vo2SourceText } from '../../../../util/triathlon-i18n'
import { createDomFactory } from '../../runtime/dom'
import { el } from '../../runtime/dom'
import { svg } from '../../runtime/dom'
import { ANA_H } from '../shared'
import { ANA_W } from '../shared'
import { anaTitle } from '../shared'
import { clampN } from '../shared'
import { fmtSpeedKmh } from '../shared'
import { markGloss } from '../shared'
import { polyD } from '../shared'
import { signed } from '../shared'
import { signedFixed } from '../shared'
import { speedFromKmh } from '../shared'
import { speedUnitLabel } from '../shared'

export const buildVo2max = (data: Analytics, context: TriathlonContext): HTMLElement => {
  const text = (key: string): string => context.formatter.text(key)
  const block = el('div', 'tri-engine-vo2')
  block.appendChild(anaTitle(context.formatter, 'vo2max · fitness age', 'vo2max'))
  const v = data.engine.vo2max
  if (v.value == null) {
    block.appendChild(el('div', 'tri-ana-empty', text('no power or hr data yet')))
    return block
  }
  const head = el('div', 'tri-engine-vo2-head')
  const num = el('div', 'tri-engine-vo2-num', v.value.toFixed(1))
  num.appendChild(el('span', 'tri-engine-vo2-unit', ' ml/kg/min'))
  head.appendChild(num)
  if (v.fitnessAge != null)
    head.appendChild(
      markGloss(
        el(
          'span',
          `tri-engine-age tri-dir-${(v.ageDeltaYears ?? 0) <= 0 ? 'up' : 'down'}`,
          `${text('fitness age')} ${v.fitnessAge} (${signed(v.ageDeltaYears ?? 0)}y)`,
        ),
        'fitage',
      ),
    )
  block.appendChild(head)
  const pos = (a: number): number => clampN(((a - 20) / 60) * 100, 1.5, 98.5)
  const bar = el('div', 'tri-engine-agebar')
  if (v.fitnessAge != null) {
    const needle = el('span', 'tri-engine-agebar-needle')
    needle.style.left = `${pos(v.fitnessAge)}%`
    needle.title = `${text('fitness age')} ${v.fitnessAge}`
    bar.appendChild(needle)
  }
  const chrono = el('span', 'tri-engine-agebar-chrono')
  chrono.style.left = `${pos(v.chronoAge)}%`
  chrono.title = `${text('age')} ${v.chronoAge}`
  bar.appendChild(chrono)
  block.appendChild(bar)
  if (v.trend.length > 1) {
    const n = v.trend.length
    let lo = Infinity
    let hi = -Infinity
    for (const p of v.trend) {
      if (p.vo2max < lo) lo = p.vo2max
      if (p.vo2max > hi) hi = p.vo2max
    }
    if (hi - lo < 2) {
      hi += 1
      lo -= 1
    }
    const x = (i: number): number => (i / (n - 1)) * ANA_W
    const y = (val: number): number => 27 - ((val - lo) / (hi - lo)) * 24
    const s = svg('svg', {
      class: 'tri-ana-svg tri-engine-vo2-spark',
      viewBox: `0 0 ${ANA_W} ${ANA_H}`,
      preserveAspectRatio: 'none',
    })
    const pts = v.trend.map((p, i) => [x(i), y(p.vo2max)] as [number, number])
    const proj = (i: number): boolean =>
      v.trend[i - 1].method === 'bike' || v.trend[i].method === 'bike'
    let from = 0
    for (let i = 1; i < n; i++) {
      if (i === n - 1 || proj(i + 1) !== proj(i)) {
        s.appendChild(
          svg('path', {
            d: polyD(pts.slice(from, i + 1)),
            class: `tri-elev-line tri-line-bike${proj(i) ? ' tri-vo2-proj' : ''}`,
          }),
        )
        from = i
      }
    }
    s.appendChild(svg('line', { x1: 0, y1: 0, x2: 0, y2: ANA_H, class: 'tri-ana-cursor' }))
    const frame = axisFrame(
      createDomFactory(context.presentation),
      s,
      [hi, (hi + lo) / 2, lo].map(val => ({ label: val.toFixed(1), vbY: y(val) })),
      ANA_H,
      [
        {
          label: context.formatter.shortDate(v.trend[0].weekStart),
          pct: 0,
          cls: 'tri-cax-xt--first',
        },
        {
          label: context.formatter.shortDate(v.trend[v.trend.length - 1].weekStart),
          pct: 100,
          cls: 'tri-cax-xt--last',
        },
      ],
    )
    frame.classList.add('tri-engine-vo2-axis')
    block.appendChild(frame)
    block.appendChild(el('div', 'tri-chart-readout'))
  }
  const cap = el('div', 'tri-elev-cap')
  cap.append(
    el('span', 'tri-ana-k', v.method),
    markGloss(el('span', `tri-ana-k tri-conf-${v.conf}`, v.conf), 'conf'),
  )
  if (v.trendSummary.change28d != null) {
    const direction = v.trendSummary.direction
    cap.append(
      markGloss(
        el(
          'span',
          `tri-ana-k tri-dir-${direction === 'improving' ? 'up' : direction === 'declining' ? 'down' : 'flat'}`,
          `${text('28d trend')} ${signedFixed(v.trendSummary.change28d, 1)}`,
        ),
        'vo2max',
      ),
      el(
        'span',
        'tri-ana-k',
        `${signedFixed(v.trendSummary.slopePerWeek ?? 0, 2)} ${text('per week')} · n ${v.trendSummary.sampleSize}`,
      ),
    )
  }
  if (v.percentileForAge != null)
    cap.appendChild(el('span', 'tri-ana-k', `p${v.percentileForAge} for age ${v.chronoAge}`))
  cap.appendChild(
    el('span', 'tri-ana-k', vo2SourceText(context.presentation.locale, v.method, v.bikeSource)),
  )
  cap.appendChild(el('span', 'tri-ana-k', `hrmax ${v.hrMax} (${v.hrMaxSource})`))
  if (v.trend.some(p => p.method === 'bike'))
    cap.appendChild(
      el('span', 'tri-ana-k tri-vo2-proj-note', text('dashed line is projected from bike power')),
    )
  block.appendChild(cap)
  const lab = data.tests.vo2max[data.tests.vo2max.length - 1]
  if (lab) {
    const labCap = el('div', 'tri-elev-cap tri-vo2-lab')
    if (lab.vt1Hr != null)
      labCap.appendChild(
        el(
          'span',
          'tri-ana-k',
          `vt1 ${lab.vt1Hr}bpm${lab.vt1Kmh != null ? ` · ${fmtSpeedKmh(context.formatter, lab.vt1Kmh, 1, '')}` : ''}`,
        ),
      )
    if (lab.maxKmh != null)
      labCap.appendChild(
        el('span', 'tri-ana-k', `vmax ${fmtSpeedKmh(context.formatter, lab.maxKmh, 1, '')}`),
      )
    if (lab.ve != null) labCap.appendChild(el('span', 'tri-ana-k', `ve ${lab.ve}l/min`))
    labCap.appendChild(
      el('span', 'tri-ana-k', `${text('lab')} ${context.formatter.longDate(lab.date)}`),
    )
    block.appendChild(labCap)
  }
  return block
}

export const VO2P_W = 800

export const VO2P_H = 430

export const VO2P_L = 82

export const VO2P_R = 690

export const VO2P_T = 48

export const VO2P_B = 360

export const VO2P_PW = VO2P_R - VO2P_L

export const VO2P_PH = VO2P_B - VO2P_T

export type Vo2ProfileMetric = 'vo2' | 'hr' | 've' | 'rf' | 'tv'

export type Vo2ProfileChartKind = 'metabolic' | 'ventilation'

export const vo2ProfileChartLabel = (
  formatter: TriathlonFormatter,
  kind: Vo2ProfileChartKind,
): string => formatter.text(kind === 'metabolic' ? 'Metabolic' : 'Ventilation')

export const vo2ProfileTargetLegend = (formatter: TriathlonFormatter): string =>
  `${formatter.text('Target')}[${speedUnitLabel(formatter)}]`

export const vo2ProfileTargetTick = (formatter: TriathlonFormatter, kmh: number): string =>
  formatter.presentation.distance === 'imperial'
    ? speedFromKmh(formatter, kmh).toFixed(kmh === 0 ? 0 : 1)
    : kmh.toFixed(0)

export const vo2ProfileText = (
  cls: string,
  text: string,
  attrs: Record<string, string | number>,
): SVGElement => {
  const t = svg('text', { ...attrs, class: cls })
  t.textContent = text
  return t
}

export const vo2ProfileTime = (seconds: number): string => {
  const sec = Math.max(0, Math.round(seconds))
  const min = Math.floor(sec / 60)
  return `${min}:${String(sec - min * 60).padStart(2, '0')}`
}

export const vo2ProfileWithTip = <T extends SVGElement>(
  node: T,
  heading: string,
  detail: string,
): T => {
  node.setAttribute('data-tip-h', heading)
  node.setAttribute('data-tip-d', detail)
  node.setAttribute('aria-label', `${heading} ${detail}`)
  return node
}

export const vo2ProfileSampleValue = (
  sample: Vo2LabProfileSample,
  metric: Vo2ProfileMetric,
): number | null => {
  if (metric === 'vo2') return sample.vo2
  if (metric === 'hr') return sample.hr
  if (metric === 've') return sample.ve
  if (metric === 'rf') return sample.rf
  return sample.tv
}

export const vo2ProfileX = (profile: Vo2LabProfile, t: number): number =>
  VO2P_L + clampN(t / profile.durationSec, 0, 1) * VO2P_PW

export const vo2ProfileY = (value: number, lo: number, hi: number): number =>
  VO2P_B - clampN((value - lo) / (hi - lo), 0, 1) * VO2P_PH

export const vo2ProfilePath = (
  profile: Vo2LabProfile,
  metric: Vo2ProfileMetric,
  lo: number,
  hi: number,
): string => {
  const pts: [number, number][] = []
  for (const sample of profile.samples) {
    const v = vo2ProfileSampleValue(sample, metric)
    if (v != null) pts.push([vo2ProfileX(profile, sample.t), vo2ProfileY(v, lo, hi)])
  }
  return polyD(pts)
}

export const vo2ProfileTargetPath = (
  profile: Vo2LabProfile,
  steps: Vo2LabTargetStep[],
  area: boolean,
): string => {
  if (!steps.length) return ''
  let d = `M ${vo2ProfileX(profile, 0).toFixed(2)} ${vo2ProfileY(steps[0].kmh, 0, 20).toFixed(2)}`
  for (let i = 1; i < steps.length; i++) {
    const prev = steps[i - 1]
    const curr = steps[i]
    const x = vo2ProfileX(profile, curr.t).toFixed(2)
    d += ` L ${x} ${vo2ProfileY(prev.kmh, 0, 20).toFixed(2)}`
    d += ` L ${x} ${vo2ProfileY(curr.kmh, 0, 20).toFixed(2)}`
  }
  const last = steps[steps.length - 1]
  d += ` L ${vo2ProfileX(profile, profile.durationSec).toFixed(2)} ${vo2ProfileY(last.kmh, 0, 20).toFixed(2)}`
  if (area)
    d += ` L ${vo2ProfileX(profile, profile.durationSec).toFixed(2)} ${VO2P_B} L ${vo2ProfileX(profile, 0).toFixed(2)} ${VO2P_B} Z`
  return d
}

export const vo2ProfileStat = (
  formatter: TriathlonFormatter,
  label: string,
  stats: Vo2LabProfileStats | null,
  cls: string,
  dp: number,
): HTMLElement | null => {
  if (!stats) return null
  const item = el('span', 'tri-vo2p-stat')
  item.append(
    el('span', `tri-vo2p-stat-name ${cls}`, label),
    el('span', 'tri-vo2p-stat-k', `${formatter.text('Min')}:`),
    el('span', `tri-vo2p-stat-v ${cls}`, formatter.number(stats.min, dp, dp)),
    el('span', 'tri-vo2p-stat-k', `${formatter.text('Max')}:`),
    el('span', `tri-vo2p-stat-v ${cls}`, formatter.number(stats.max, dp, dp)),
    el('span', 'tri-vo2p-stat-k', `${formatter.text('Avg')}:`),
    el('span', `tri-vo2p-stat-v ${cls}`, formatter.number(stats.avg, dp, dp)),
  )
  return item
}

export const vo2ProfileLegendItem = (label: string, cls: string, area = false): HTMLElement => {
  const item = el('span', 'tri-vo2p-leg')
  item.append(el('span', `tri-vo2p-leg-mark ${cls}${area ? ' tri-vo2p-leg-mark--area' : ''}`))
  item.appendChild(el('span', 'tri-vo2p-leg-text', label))
  return item
}

export const vo2ProfileTicks = (
  s: SVGElement,
  values: number[],
  lo: number,
  hi: number,
  xText: number,
  xTick0: number,
  xTick1: number,
  cls: string,
  anchor: 'start' | 'end',
  dp = 0,
): void => {
  for (const v of values) {
    const y = vo2ProfileY(v, lo, hi)
    s.appendChild(svg('line', { x1: xTick0, y1: y, x2: xTick1, y2: y, class: 'tri-vo2p-tick' }))
    s.appendChild(
      vo2ProfileText('tri-vo2p-ytext ' + cls, v.toFixed(dp), {
        x: xText,
        y: y + 4,
        'text-anchor': anchor,
      }),
    )
  }
}

export const vo2ProfileTargetTicks = (formatter: TriathlonFormatter, s: SVGElement): void => {
  for (const kmh of [0, 5, 10, 15, 20]) {
    const y = vo2ProfileY(kmh, 0, 20)
    s.appendChild(
      svg('line', { x1: VO2P_R, y1: y, x2: VO2P_R + 10, y2: y, class: 'tri-vo2p-tick' }),
    )
    s.appendChild(
      vo2ProfileText('tri-vo2p-ytext tri-vo2p-target', vo2ProfileTargetTick(formatter, kmh), {
        x: VO2P_R + 52,
        y: y + 4,
        'text-anchor': 'start',
      }),
    )
  }
}

export const vo2ProfilePhase = (s: SVGElement, profile: Vo2LabProfile): void => {
  if (profile.warmupEndSec != null)
    s.appendChild(
      svg('rect', {
        x: vo2ProfileX(profile, 0),
        y: VO2P_T,
        width: vo2ProfileX(profile, profile.warmupEndSec) - VO2P_L,
        height: VO2P_PH,
        class: 'tri-vo2p-phase',
      }),
    )
  if (profile.cooldownStartSec != null)
    s.appendChild(
      svg('rect', {
        x: vo2ProfileX(profile, profile.cooldownStartSec),
        y: VO2P_T,
        width:
          vo2ProfileX(profile, profile.durationSec) -
          vo2ProfileX(profile, profile.cooldownStartSec),
        height: VO2P_PH,
        class: 'tri-vo2p-phase',
      }),
    )
}

export const vo2ProfileZoneHit = (
  s: SVGElement,
  profile: Vo2LabProfile,
  start: number,
  end: number,
  label: string,
): void => {
  if (end <= start) return
  s.appendChild(
    vo2ProfileWithTip(
      svg('rect', {
        x: vo2ProfileX(profile, start),
        y: VO2P_T,
        width: vo2ProfileX(profile, end) - vo2ProfileX(profile, start),
        height: VO2P_PH,
        class: 'tri-vo2p-zone-hit',
      }),
      label,
      `${vo2ProfileTime(start)}-${vo2ProfileTime(end)}`,
    ),
  )
}

export const vo2ProfileZoneHits = (
  formatter: TriathlonFormatter,
  s: SVGElement,
  profile: Vo2LabProfile,
): void => {
  const warmupEnd = profile.warmupEndSec ?? 0
  const cooldownStart = profile.cooldownStartSec ?? profile.durationSec
  if (profile.warmupEndSec != null)
    vo2ProfileZoneHit(s, profile, 0, warmupEnd, formatter.text('Warm-Up'))
  vo2ProfileZoneHit(s, profile, warmupEnd, cooldownStart, formatter.text('Test'))
  if (profile.cooldownStartSec != null)
    vo2ProfileZoneHit(s, profile, cooldownStart, profile.durationSec, formatter.text('Cool-Down'))
}

export const vo2ProfileMarker = (
  s: SVGElement,
  profile: Vo2LabProfile,
  t: number | null,
  label: string,
): void => {
  if (t == null) return
  const x = vo2ProfileX(profile, t)
  s.appendChild(svg('line', { x1: x, y1: VO2P_T, x2: x, y2: VO2P_B, class: 'tri-vo2p-marker' }))
  s.appendChild(
    vo2ProfileWithTip(
      svg('rect', {
        x: x - 5,
        y: VO2P_T,
        width: 10,
        height: VO2P_PH,
        class: 'tri-vo2p-marker-hit',
      }),
      label,
      vo2ProfileTime(t),
    ),
  )
}

export const vo2ProfileBaseSvg = (
  formatter: TriathlonFormatter,
  profile: Vo2LabProfile,
  kind: Vo2ProfileChartKind,
): SVGElement => {
  const s = svg('svg', {
    class: 'tri-vo2p-svg',
    viewBox: `0 0 ${VO2P_W} ${VO2P_H}`,
    preserveAspectRatio: 'xMidYMid meet',
  })
  s.appendChild(svg('rect', { x: 0, y: 0, width: VO2P_W, height: VO2P_H, class: 'tri-vo2p-bg' }))
  vo2ProfilePhase(s, profile)
  const targetD = vo2ProfileTargetPath(profile, profile.targetKmh, false)
  s.appendChild(
    svg('path', {
      d: vo2ProfileTargetPath(profile, profile.targetKmh, true),
      class: 'tri-vo2p-target-area',
    }),
  )
  for (let t = 0; t <= 720; t += 30) {
    const x = vo2ProfileX(profile, t)
    const major = t % 120 === 0
    s.appendChild(
      svg('line', {
        x1: x,
        y1: VO2P_B,
        x2: x,
        y2: VO2P_B + (major ? 14 : 8),
        class: 'tri-vo2p-xtick',
      }),
    )
    if (major)
      s.appendChild(
        vo2ProfileText('tri-vo2p-xtext', `${Math.floor(t / 60)}:00`, {
          x,
          y: VO2P_B + 34,
          'text-anchor': 'middle',
        }),
      )
  }
  for (const gy of [0, 0.25, 0.5, 0.75, 1]) {
    const y = VO2P_T + gy * VO2P_PH
    s.appendChild(svg('line', { x1: VO2P_L, y1: y, x2: VO2P_R, y2: y, class: 'tri-vo2p-grid' }))
  }
  s.appendChild(svg('path', { d: targetD, class: 'tri-vo2p-target-line' }))
  vo2ProfileZoneHits(formatter, s, profile)
  vo2ProfileMarker(s, profile, profile.vt1Sec, 'VT 1')
  vo2ProfileMarker(s, profile, profile.vo2maxSec, 'VO2 max')
  s.appendChild(
    svg('rect', { x: VO2P_L, y: VO2P_T, width: VO2P_PW, height: VO2P_PH, class: 'tri-vo2p-frame' }),
  )
  if (kind === 'metabolic') {
    s.appendChild(
      svg('path', {
        d: vo2ProfilePath(profile, 'vo2', 0, 60),
        class: 'tri-vo2p-line tri-vo2p-line--vo2',
      }),
    )
    s.appendChild(
      svg('path', {
        d: vo2ProfilePath(profile, 'hr', 60, 200),
        class: 'tri-vo2p-line tri-vo2p-line--hr',
      }),
    )
    vo2ProfileTicks(
      s,
      [60, 80, 100, 120, 140, 160, 180, 200],
      60,
      200,
      VO2P_L - 17,
      VO2P_L - 10,
      VO2P_L,
      'tri-vo2p-red',
      'end',
    )
    vo2ProfileTicks(
      s,
      [0, 10, 20, 30, 40, 50, 60],
      0,
      60,
      VO2P_R + 16,
      VO2P_R,
      VO2P_R + 10,
      'tri-vo2p-blue',
      'start',
    )
  } else {
    s.appendChild(
      svg('path', {
        d: vo2ProfilePath(profile, 've', 0, 160),
        class: 'tri-vo2p-line tri-vo2p-line--ve',
      }),
    )
    s.appendChild(
      svg('path', {
        d: vo2ProfilePath(profile, 'rf', 0, 80),
        class: 'tri-vo2p-line tri-vo2p-line--rf',
      }),
    )
    s.appendChild(
      svg('path', {
        d: vo2ProfilePath(profile, 'tv', 0, 4),
        class: 'tri-vo2p-line tri-vo2p-line--tv',
      }),
    )
    vo2ProfileTicks(
      s,
      [0, 20, 40, 60, 80],
      0,
      80,
      VO2P_L - 16,
      VO2P_L - 10,
      VO2P_L,
      'tri-vo2p-cyan',
      'end',
    )
    vo2ProfileTicks(
      s,
      [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4],
      0,
      4,
      VO2P_L - 50,
      VO2P_L - 10,
      VO2P_L,
      'tri-vo2p-orange',
      'end',
      1,
    )
    vo2ProfileTicks(
      s,
      [0, 20, 40, 60, 80, 100, 120, 140, 160],
      0,
      160,
      VO2P_R + 16,
      VO2P_R,
      VO2P_R + 10,
      'tri-vo2p-green',
      'start',
    )
  }
  vo2ProfileTargetTicks(formatter, s)
  return s
}

export const buildVo2ProfileChart = (
  formatter: TriathlonFormatter,
  profile: Vo2LabProfile,
  kind: Vo2ProfileChartKind,
): HTMLElement => {
  const panel = el('div', 'tri-vo2p-panel')
  const head = el('div', 'tri-vo2p-panel-head')
  head.appendChild(el('span', 'tri-vo2p-panel-heading', vo2ProfileChartLabel(formatter, kind)))
  const stats = el('div', 'tri-vo2p-stats')
  const profileStats =
    kind === 'metabolic'
      ? [
          vo2ProfileStat(formatter, 'VO2', profile.stats.vo2, 'tri-vo2p-blue', 1),
          vo2ProfileStat(formatter, formatter.text('HR'), profile.stats.hr, 'tri-vo2p-red', 0),
        ]
      : [
          vo2ProfileStat(formatter, 'Tv', profile.stats.tv, 'tri-vo2p-orange', 1),
          vo2ProfileStat(formatter, 'Rf', profile.stats.rf, 'tri-vo2p-cyan', 1),
          vo2ProfileStat(formatter, 'Ve', profile.stats.ve, 'tri-vo2p-green', 1),
        ]
  for (const stat of profileStats) if (stat) stats.appendChild(stat)
  const legend = el('div', 'tri-vo2p-legend')
  legend.appendChild(
    vo2ProfileLegendItem(vo2ProfileTargetLegend(formatter), 'tri-vo2p-target', true),
  )
  if (kind === 'metabolic') {
    legend.appendChild(vo2ProfileLegendItem('VO2[mL/kg/min]', 'tri-vo2p-blue'))
    legend.appendChild(vo2ProfileLegendItem(`${formatter.text('HR')}[bpm]`, 'tri-vo2p-red'))
  } else {
    legend.appendChild(vo2ProfileLegendItem('Ve[L/min]', 'tri-vo2p-green'))
    legend.appendChild(vo2ProfileLegendItem('Rf[bpm]', 'tri-vo2p-cyan'))
    legend.appendChild(vo2ProfileLegendItem('Tv[L]', 'tri-vo2p-orange'))
  }
  const fig = el('div', 'tri-vo2p-fig')
  fig.append(legend, vo2ProfileBaseSvg(formatter, profile, kind))
  panel.append(head, stats, fig)
  return panel
}

export const buildVo2Profile = (
  formatter: TriathlonFormatter,
  profile: Vo2LabProfile,
): HTMLElement => {
  const wrap = el('div', 'tri-vo2p')
  wrap.append(
    buildVo2ProfileChart(formatter, profile, 'metabolic'),
    buildVo2ProfileChart(formatter, profile, 'ventilation'),
  )
  return wrap
}

export const appendVo2TestCap = (
  formatter: TriathlonFormatter,
  block: HTMLElement,
  r: Vo2LabRecord,
): void => {
  const cap = el('div', 'tri-elev-cap')
  cap.appendChild(el('span', 'tri-ana-k', `vo2max ${formatter.number(r.value, 1, 1)} ml/kg/min`))
  if (r.percentile != null) cap.appendChild(el('span', 'tri-ana-k', `p${r.percentile}`))
  if (r.vt1Hr != null)
    cap.appendChild(
      el(
        'span',
        'tri-ana-k',
        `vt1 ${r.vt1Hr}bpm${r.vt1Kmh != null ? ` · ${fmtSpeedKmh(formatter, r.vt1Kmh, 1, '')}` : ''}${r.caloriesAtVt1 != null ? ` · ${r.caloriesAtVt1}kcal/h` : ''}`,
      ),
    )
  if (r.ve != null) cap.appendChild(el('span', 'tri-ana-k', `ve ${r.ve}l/min`))
  if (r.hrMax != null) cap.appendChild(el('span', 'tri-ana-k', `hrmax ${r.hrMax}`))
  block.appendChild(cap)
}
