import type { Analytics } from '../../../../plugins/stores/analytics'
import type { Vo2LabRecord } from '../../../../plugins/stores/analytics'
import type { TriathlonContext } from '../../runtime/context'
import type { TriathlonFormatter } from '../../runtime/formatter'
import { el } from '../../runtime/dom'
import { svg } from '../../runtime/dom'
import { anaTitle } from '../shared'
import { clampN } from '../shared'
import { fmtSpeedKmh } from '../shared'
import { markGloss } from '../shared'
import { polyD } from '../shared'
import { weightUnitLabel } from '../shared'
import { wNum } from '../shared'
import { appendVo2TestCap } from './vo2'
import { buildVo2Profile } from './vo2'

export function buildVo2TestRecord(formatter: TriathlonFormatter, r?: Vo2LabRecord): HTMLElement {
  const text = (key: string): string => formatter.text(key)
  const block = el('div', 'tri-vo2t')
  const titleRow = el('div', 'tri-vo2t-titlerow')
  titleRow.appendChild(anaTitle(formatter, 'vo2 test profile', 'vo2test'))
  if (r?.profile) {
    block.appendChild(titleRow)
    block.appendChild(buildVo2Profile(formatter, r.profile))
    appendVo2TestCap(formatter, block, r)
    return block
  }
  const anchors = r
    ? r.zonesKmh.map((s, i) => ({ s, hr: r.zonesHr[i] })).filter(p => p.hr != null)
    : []
  if (!r || anchors.length < 2 || r.maxKmh == null) {
    block.appendChild(titleRow)
    block.appendChild(el('div', 'tri-ana-empty', text('no vo2 test logged')))
    return block
  }
  block.appendChild(titleRow)

  const maxHr = r.hrAtVo2max ?? r.hrMax ?? anchors[anchors.length - 1].hr
  const bandTop = r.hrMax ?? maxHr
  const curve = [...anchors, { s: r.maxKmh, hr: maxHr }].sort((a, b) => a.s - b.s)
  const speeds = curve.map(p => p.s)
  const hrs = [...curve.map(p => p.hr), bandTop]
  if (r.vt1Hr != null) hrs.push(r.vt1Hr)
  const x0 = Math.min(...speeds) - 0.6
  const x1 = Math.max(...speeds) + 0.6
  const y0 = Math.min(...hrs) - 6
  const y1 = Math.max(...hrs) + 6
  const xP = (s: number): number => ((s - x0) / (x1 - x0)) * 100
  const yP = (hr: number): number => (1 - (hr - y0) / (y1 - y0)) * 100

  const chart = el('div', 'tri-vo2t-chart')
  const yax = el('div', 'tri-vo2t-yax')
  yax.append(el('span', '', String(Math.round(y1))), el('span', '', String(Math.round(y0))))
  const plot = el('div', 'tri-vo2t-plot')
  const s = svg('svg', {
    class: 'tri-vo2t-svg',
    viewBox: '0 0 100 100',
    preserveAspectRatio: 'none',
  })

  const zoneNames = ['warm up', 'fat burning', 'endurance', 'vigorous', 'maximal']
  const lows = [...r.zonesHr, bandTop]
  lows.forEach((lo, i) => {
    const hi = i + 1 < lows.length ? lows[i + 1] : Math.ceil(y1)
    const visLo = i === 0 ? y0 : lo
    const last = i + 1 >= lows.length
    const spd = r.zonesKmh[i] != null ? ` · ${fmtSpeedKmh(formatter, r.zonesKmh[i])}` : ''
    const kcal = r.zonesKcal[i] != null ? ` · ${r.zonesKcal[i]} kcal/h` : ''
    s.appendChild(
      svg('rect', {
        x: 0,
        y: yP(hi),
        width: 100,
        height: yP(visLo) - yP(hi),
        class: `tri-vo2t-zone tri-vo2t-zone--${i + 1}`,
        'data-tip-h': text(zoneNames[i] ?? `zone ${i + 1}`),
        'data-tip-d': `${last ? `${lo}+ bpm` : `${lo}–${hi - 1} bpm`}${spd}${kcal}`,
      }),
    )
  })

  for (const gy of [0, 50, 100])
    s.appendChild(svg('line', { x1: 0, y1: gy, x2: 100, y2: gy, class: 'tri-vo2t-grid' }))
  if (r.vt1Kmh != null)
    s.appendChild(
      svg('line', { x1: xP(r.vt1Kmh), y1: 0, x2: xP(r.vt1Kmh), y2: 100, class: 'tri-vo2t-vt' }),
    )
  s.appendChild(
    svg('line', { x1: xP(r.maxKmh), y1: 0, x2: xP(r.maxKmh), y2: 100, class: 'tri-vo2t-vt' }),
  )
  s.appendChild(
    svg('path', { d: polyD(curve.map(p => [xP(p.s), yP(p.hr)])), class: 'tri-vo2t-line' }),
  )
  plot.appendChild(s)
  for (const p of curve) {
    const m = el('span', 'tri-vo2t-pt')
    m.style.left = `${xP(p.s).toFixed(1)}%`
    m.style.top = `${yP(p.hr).toFixed(1)}%`
    plot.appendChild(m)
  }
  const marker = (
    spd: number,
    hr: number,
    mod: string,
    label: string,
    tipH: string,
    tipD: string,
  ): void => {
    const pt = el('span', `tri-vo2t-pt tri-vo2t-pt--${mod}`)
    pt.style.left = `${xP(spd).toFixed(1)}%`
    pt.style.top = `${yP(hr).toFixed(1)}%`
    pt.dataset.tipH = tipH
    pt.dataset.tipD = tipD
    plot.appendChild(pt)
    if (label) {
      const lbl = el('span', `tri-vo2t-lbl tri-vo2t-lbl--${mod}`, label)
      lbl.style.left = `${xP(spd).toFixed(1)}%`
      lbl.style.top = `${yP(hr).toFixed(1)}%`
      plot.appendChild(lbl)
    }
  }
  if (r.vt1Kmh != null && r.vt1Hr != null)
    marker(
      r.vt1Kmh,
      r.vt1Hr,
      'vt',
      'vt1',
      'vt1 · aerobic threshold',
      `${r.vt1Hr} bpm · ${fmtSpeedKmh(formatter, r.vt1Kmh)}${r.caloriesAtVt1 != null ? ` · ${r.caloriesAtVt1} kcal/h` : ''}`,
    )
  marker(
    r.maxKmh,
    maxHr,
    'max',
    '',
    'vo2max',
    `${r.value.toFixed(1)} ml/kg/min · ${maxHr} bpm · ${fmtSpeedKmh(formatter, r.maxKmh)}`,
  )
  chart.append(yax, plot)
  block.appendChild(chart)
  const xax = el('div', 'tri-vo2t-xax')
  xax.append(
    el('span', '', fmtSpeedKmh(formatter, speeds[0])),
    el('span', '', fmtSpeedKmh(formatter, r.maxKmh)),
  )
  block.appendChild(xax)

  appendVo2TestCap(formatter, block, r)
  return block
}

export const buildFtpHypothesis = (data: Analytics, context: TriathlonContext): HTMLElement => {
  const text = (key: string): string => context.formatter.text(key)
  const block = el('div', 'tri-ftp')
  block.appendChild(anaTitle(context.formatter, 'ftp hypothesis', 'ftp'))
  const h = data.engine.ftpHypothesis
  if (!h) {
    block.appendChild(el('div', 'tri-ana-empty', text('no vo2-derived ftp estimate')))
    return block
  }
  const massNote =
    h.massSource === 'daily'
      ? `${text('latest daily weight')} · ${context.formatter.longDate(h.massDate)}`
      : `${text('value from vo2 report')} · ${context.formatter.longDate(h.massDate)}`
  const vo2Fallback =
    h.vo2maxSource === 'garmin' && h.defaultVo2max != null
      ? ` · ${text('fallback')} ${h.defaultVo2max.toFixed(1)}`
      : ''
  const vo2Note =
    h.vo2maxSource === 'garmin'
      ? `Garmin · ${context.formatter.longDate(h.vo2maxDate)} · ${text(
          h.vo2maxSport === 'cycling' ? 'cycling-specific source' : 'unknown sport provenance',
        )}${vo2Fallback}`
      : h.vo2maxSource === 'lab'
        ? `${text('measured during treadmill test')} · ${context.formatter.longDate(h.vo2maxDate)}`
        : `${text('athlete default')} · ${h.vo2max.toFixed(1)}`
  const vo2Label =
    h.vo2maxSport === 'cycling'
      ? 'cycling vo2max'
      : h.vo2maxSport === 'running'
        ? 'running vo2max'
        : 'vo2max'

  const head = el('div', 'tri-ftp-head')
  const headline = el('div', 'tri-ftp-main')
  headline.append(
    el('span', 'tri-ftp-num', String(h.ftp), { 'data-ftp-out': 'headline' }),
    el('span', 'tri-ftp-unit', ' W'),
  )
  const meta = el('div', 'tri-ftp-meta')
  meta.append(
    el('span', 'tri-ftp-pill', `${h.low}-${h.high} W`, { 'data-ftp-out': 'band' }),
    el('span', 'tri-ftp-pill', `${h.wattsPerKg.toFixed(2)} W/kg`, { 'data-ftp-out': 'wkg' }),
    markGloss(el('span', `tri-ftp-pill tri-conf-${h.conf}`, h.conf), 'conf'),
  )
  head.append(headline, meta)
  block.appendChild(head)

  const methods = el('div', 'tri-ftp-methods')
  const methodRow = (label: string, key: string, value: number, cls: string): HTMLElement => {
    const row = el('div', 'tri-ftp-method')
    row.appendChild(el('span', 'tri-ftp-method-k', text(label)))
    const track = el('span', 'tri-ftp-method-track')
    const fill = el('span', `tri-ftp-method-fill ${cls}`, undefined, { 'data-ftp-bar': key })
    fill.style.width = `${clampN((value / 350) * 100, 4, 100)}%`
    track.appendChild(fill)
    row.append(
      track,
      el('span', 'tri-ftp-method-v', `${Math.round(value)} W`, { 'data-ftp-out': key }),
    )
    return row
  }
  methods.append(
    methodRow('efficiency estimate', 'efficiencyFtp', h.efficiencyFtp, 'tri-ftp-method-fill--eff'),
    methodRow('ACSM estimate', 'acsmFtp', h.acsmFtp, 'tri-ftp-method-fill--acsm'),
  )
  block.appendChild(methods)

  const chain = el('div', 'tri-ftp-chain')
  const chainRow = (label: string, key: string, value: string): HTMLElement => {
    const row = el('div', 'tri-ftp-chain-row')
    row.append(
      el('span', 'tri-ftp-chain-k', text(label)),
      el('span', 'tri-ftp-chain-v', value, { 'data-ftp-out': key }),
    )
    return row
  }
  chain.append(
    chainRow('total vo2max', 'absoluteVo2', `${h.absoluteVo2.toFixed(2)} L/min`),
    chainRow('estimated cycling vo2max', 'cyclingVo2max', `${h.cyclingVo2max.toFixed(2)} L/min`),
    chainRow('vo2 used at threshold', 'thresholdVo2', `${h.thresholdVo2.toFixed(2)} L/min`),
    chainRow('energy used per second', 'metabolicWatts', `${Math.round(h.metabolicWatts)} W`),
    chainRow('maximum aerobic power', 'acsmMapWatts', `${Math.round(h.acsmMapWatts)} W`),
  )
  block.appendChild(chain)

  const evidence = el('div', 'tri-ftp-chain')
  const evidenceValue = (value: number | null): string =>
    value == null ? '—' : `${Math.round(value)} W`
  evidence.append(
    chainRow(
      'estimated critical power',
      'criticalPower',
      evidenceValue(h.power.criticalPowerWatts),
    ),
    chainRow(
      'modeled 60-minute power',
      'modeled60MinutePower',
      evidenceValue(h.power.modeled60MinuteWatts),
    ),
    chainRow('declared ftp', 'declaredFtp', evidenceValue(h.power.declaredFtpWatts)),
    chainRow(
      'independent efforts',
      'independentEfforts',
      `${h.power.independentEffortCount} · ${text(h.power.confidence ?? 'no model')}`,
    ),
  )
  if (h.pedaling) {
    evidence.append(
      chainRow(
        'pedal smoothness',
        'pedalSmoothness',
        `${text('left')} ${h.pedaling.leftPedalSmoothnessPct.toFixed(1)}% · ${text('right')} ${h.pedaling.rightPedalSmoothnessPct.toFixed(1)}%`,
      ),
      chainRow(
        'torque effectiveness',
        'torqueEffectiveness',
        `${text('left')} ${h.pedaling.leftTorqueEffectivenessPct.toFixed(1)}% · ${text('right')} ${h.pedaling.rightTorqueEffectivenessPct.toFixed(1)}%`,
      ),
      chainRow('activities', 'pedalingActivities', String(h.pedaling.activityCount)),
      chainRow('samples', 'pedalingSamples', String(h.pedaling.sampleCount)),
      chainRow('coverage', 'pedalingCoverage', `${h.pedaling.coveragePct.toFixed(1)}%`),
    )
  }
  block.appendChild(evidence)

  const controls = el('div', 'tri-ftp-controls')
  const control = (
    key: string,
    label: string,
    min: number,
    max: number,
    step: number,
    value: number,
    unit: string,
    note: string,
    editable = false,
    display: string = String(value),
  ): HTMLElement => {
    const wrap = el(editable ? 'div' : 'label', 'tri-ftp-ctrl')
    const row = el('span', 'tri-ftp-ctrl-row')
    let valEl: HTMLElement
    if (editable) {
      valEl = el('span', 'tri-ftp-ctrl-val tri-ftp-ctrl-val--edit', undefined, {
        'data-ftp-val': key,
      })
      const numIn = document.createElement('input')
      numIn.className = 'tri-ftp-ctrl-num'
      numIn.type = 'text'
      numIn.inputMode = 'decimal'
      numIn.dataset.ftpNum = key
      numIn.value = display
      numIn.setAttribute('aria-label', text(label))
      valEl.append(numIn, el('span', 'tri-ftp-ctrl-unit', unit, { 'data-ftp-unit': key }))
    } else {
      valEl = el('span', 'tri-ftp-ctrl-val', `${display}${unit}`, { 'data-ftp-val': key })
    }
    row.append(el('span', 'tri-ftp-ctrl-label', text(label)), valEl)
    const input = document.createElement('input')
    input.className = 'tri-ftp-range'
    input.type = 'range'
    input.dataset.ftpParam = key
    input.dataset.ftpDefault = String(value)
    input.min = String(min)
    input.max = String(max)
    input.step = String(step)
    input.value = String(value)
    input.setAttribute('aria-label', text(label))
    wrap.append(row, input, el('span', 'tri-ftp-note', text(note)))
    return wrap
  }
  controls.append(
    control(
      'mass',
      'body weight',
      60,
      110,
      0.1,
      h.massKg,
      ` ${weightUnitLabel(context.formatter)}`,
      massNote,
      true,
      wNum(context.formatter, h.massKg, 1, 0),
    ),
    control('vo2', vo2Label, 30, 70, 0.1, h.vo2max, '', vo2Note, true, h.vo2max.toFixed(1)),
    control(
      'discount',
      'cross-modal adjustment',
      0,
      15,
      0.5,
      h.crossModalDiscountPct,
      '%',
      h.vo2maxSport === 'cycling'
        ? 'cycling-specific vo2max needs no adjustment'
        : h.vo2maxSport === 'running'
          ? 'reduces running vo2max for cycling'
          : 'conservative adjustment for unknown sport provenance',
    ),
    control(
      'threshold',
      'vo2max used at threshold',
      70,
      92,
      0.5,
      h.thresholdPct,
      '%',
      'estimated because the treadmill test did not find the second threshold',
    ),
    control(
      'efficiency',
      'gross metabolic efficiency',
      18,
      25,
      0.5,
      h.grossEfficiencyPct,
      '%',
      `${h.efficiency.valuePct}% · ${text(
        h.efficiency.source === 'literature-prior'
          ? 'literature prior'
          : 'measured metabolic efficiency',
      )}`,
    ),
  )
  block.appendChild(controls)
  const foot = el('div', 'tri-ftp-foot')
  foot.append(
    el('span', 'tri-ftp-source', context.formatter.longDate(h.date)),
    el('span', 'tri-ftp-source', text(h.note)),
    el('button', 'tri-ftp-reset', text('reset'), { type: 'button' }),
  )
  block.appendChild(foot)
  return block
}
