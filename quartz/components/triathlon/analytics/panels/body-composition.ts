import type { Analytics } from '../../../../plugins/stores/analytics'
import type { DexaRecord } from '../../../../plugins/stores/analytics'
import type { TriathlonContext } from '../../runtime/context'
import type { TriathlonFormatter } from '../../runtime/formatter'
import { buildSelect } from '../../../controls/select'
import { el } from '../../runtime/dom'
import { anaTitle } from '../shared'
import { KG_PER_LB } from '../shared'
import { pctFmt } from '../shared'
import { wConv } from '../shared'
import { weightUnitLabel } from '../shared'
import { wFmt } from '../shared'
import { buildVo2TestRecord } from './ftp'

export const aceBand = (pct: number): string =>
  pct < 6
    ? 'essential'
    : pct < 14
      ? 'athlete'
      : pct < 18
        ? 'fitness'
        : pct < 25
          ? 'average'
          : 'obese'

export const TRI_LAB_DATE_KEY = 'tri-lab-date'

export type DexaChartColumn = {
  label: string
  totalLbs: number
  leanLbs: number
  fatLbs: number
  boneLbs: number
}

export const dexaVerticalSegment = (cls: string, lbs: number, totalLbs: number): HTMLElement => {
  const segment = el('span', `tri-dexa-seg ${cls}`)
  segment.style.height = `${totalLbs > 0 ? (lbs / totalLbs) * 100 : 0}%`
  return segment
}

export const buildDexaChart = (
  formatter: TriathlonFormatter,
  d: DexaRecord,
): HTMLElement | undefined => {
  const text = (key: string): string => formatter.text(key)
  const columns: DexaChartColumn[] = []
  for (const [label, region] of [
    ['arms', d.arms],
    ['legs', d.legs],
    ['trunk', d.trunk],
  ] as const) {
    if (!region) continue
    columns.push({
      label,
      totalLbs: region.fat + region.lean + region.bmc,
      leanLbs: region.lean,
      fatLbs: region.fat,
      boneLbs: region.bmc,
    })
  }
  if (columns.length === 0) return

  const displayMass = (lbs: number): number => wConv(formatter, lbs * KG_PER_LB)
  const axisStep = formatter.presentation.distance === 'imperial' ? 20 : 10
  const axisMax = Math.max(
    axisStep,
    Math.ceil(Math.max(...columns.map(column => displayMass(column.totalLbs))) / axisStep) *
      axisStep,
  )
  const chart = el('div', 'tri-dexa-chart', undefined, {
    role: 'group',
    'aria-label': text('body composition by region'),
  })
  const axis = el('div', 'tri-dexa-axis', undefined, { 'aria-hidden': 'true' })
  axis.append(
    el('span', 'tri-dexa-axis-value is-top', `${axisMax} ${weightUnitLabel(formatter)}`),
    el('span', 'tri-dexa-axis-value is-mid', String(axisMax / 2)),
    el('span', 'tri-dexa-axis-value is-bottom', '0'),
  )
  const plot = el('div', 'tri-dexa-plot')
  plot.style.gridTemplateColumns = `repeat(${columns.length}, minmax(0, 1fr))`
  plot.append(
    el('span', 'tri-dexa-gridline is-top'),
    el('span', 'tri-dexa-gridline is-mid'),
    el('span', 'tri-dexa-gridline is-bottom'),
  )
  for (const column of columns) {
    const compositionLbs = column.leanLbs + column.fatLbs + column.boneLbs
    const leanPct = compositionLbs > 0 ? (column.leanLbs / compositionLbs) * 100 : 0
    const fatPct = compositionLbs > 0 ? (column.fatLbs / compositionLbs) * 100 : 0
    const bonePct = compositionLbs > 0 ? (column.boneLbs / compositionLbs) * 100 : 0
    const item = el('div', 'tri-dexa-column', undefined, {
      role: 'img',
      tabindex: '0',
      'aria-label': `${text(column.label)}, ${wFmt(formatter, column.totalLbs * KG_PER_LB, 1, 1)}, ${text('lean')} ${wFmt(formatter, column.leanLbs * KG_PER_LB, 1, 1)}, ${text('fat')} ${wFmt(formatter, column.fatLbs * KG_PER_LB, 1, 1)}, ${text('bone')} ${wFmt(formatter, column.boneLbs * KG_PER_LB, 1, 1)}`,
      'data-dexa-region': text(column.label),
      'data-dexa-total': wFmt(formatter, column.totalLbs * KG_PER_LB, 1, 1),
      'data-dexa-lean': wFmt(formatter, column.leanLbs * KG_PER_LB, 1, 1),
      'data-dexa-lean-pct': pctFmt(formatter, leanPct, 0),
      'data-dexa-fat': wFmt(formatter, column.fatLbs * KG_PER_LB, 1, 1),
      'data-dexa-fat-pct': pctFmt(formatter, fatPct, 0),
      'data-dexa-bone': wFmt(formatter, column.boneLbs * KG_PER_LB, 1, 1),
      'data-dexa-bone-pct': pctFmt(formatter, bonePct, 0),
    })
    const track = el('div', 'tri-dexa-column-track')
    const bar = el('div', 'tri-dexa-column-bar')
    bar.style.height = `${(displayMass(column.totalLbs) / axisMax) * 100}%`
    bar.append(
      dexaVerticalSegment('is-lean', column.leanLbs, compositionLbs),
      dexaVerticalSegment('is-fat', column.fatLbs, compositionLbs),
      dexaVerticalSegment('is-bone', column.boneLbs, compositionLbs),
    )
    track.appendChild(bar)
    item.append(
      track,
      el('span', 'tri-dexa-column-label', text(column.label)),
      el('span', 'tri-dexa-column-value', wFmt(formatter, column.totalLbs * KG_PER_LB, 1, 1)),
      el('span', 'tri-dexa-column-fat', `${pctFmt(formatter, fatPct, 0)} ${text('fat')}`),
    )
    plot.appendChild(item)
  }
  chart.append(axis, plot)
  return chart
}

export const buildDexaDetail = (formatter: TriathlonFormatter, d: DexaRecord): HTMLElement => {
  const text = (key: string): string => formatter.text(key)
  const detail = el('div', 'tri-dexa-detail-inner')
  const head = el('div', 'tri-dexa-head')
  const bf = el('div', 'tri-dexa-bf')
  bf.append(
    el('span', 'tri-dexa-number', formatter.number(d.bodyFat, 1, 1)),
    el('span', 'tri-dexa-unit', text('% fat')),
  )
  head.append(bf, el('span', 'tri-dexa-cat', `ACE ${text(aceBand(d.bodyFat))}`))
  detail.appendChild(head)

  const legend = el('div', 'tri-dexa-legend')
  const leg = (cls: string, name: string, lbs: number): HTMLElement => {
    const item = el('span', 'tri-dexa-legitem')
    item.append(
      el('span', `tri-dexa-dot ${cls}`),
      el('span', 'tri-dexa-legname', text(name)),
      el('span', 'tri-dexa-legval', wFmt(formatter, lbs * KG_PER_LB, 1, 1)),
    )
    return item
  }
  legend.append(
    leg('is-lean', 'lean', d.leanLbs),
    leg('is-fat', 'fat', d.fatLbs),
    leg('is-bone', 'bone', d.bmcLbs),
  )
  detail.appendChild(legend)
  const chart = buildDexaChart(formatter, d)
  if (chart) detail.appendChild(chart)

  const stats = el('div', 'tri-dexa-stats')
  const stat = (label: string, val: string): void => {
    const item = el('div', 'tri-dexa-stat')
    item.append(el('span', 'tri-dexa-statv', val), el('span', 'tri-dexa-statk', text(label)))
    stats.appendChild(item)
  }
  stat('lean', wFmt(formatter, d.leanLbs * KG_PER_LB, 1, 1))
  stat('FFMI', formatter.number(d.ffmi, 1, 1))
  if (d.rmr != null) stat('rmr', `${d.rmr} kcal`)
  if (d.bmd != null) {
    const tScore =
      d.bmdT != null ? ` · T${d.bmdT > 0 ? '+' : ''}${formatter.number(d.bmdT, 1, 1)}` : ''
    stat('bmd', `${formatter.number(d.bmd, 2, 2)}${tScore}`)
  }
  if (d.vatLbs != null) stat('vat', wFmt(formatter, d.vatLbs * KG_PER_LB, 2, 2))
  if (d.rsmi != null) stat('rsmi', formatter.number(d.rsmi, 1, 1))
  if (d.ag != null) stat('a/g', formatter.number(d.ag, 2, 2))
  detail.appendChild(stats)
  return detail
}

export const buildDexa = (
  data: Analytics,
  context: TriathlonContext,
): { element: HTMLElement; mount?: () => () => void } => {
  const text = (key: string): string => context.formatter.text(key)
  const block = el('div', 'tri-dexa')
  const titleRow = el('div', 'tri-dexa-titlerow')
  titleRow.appendChild(anaTitle(context.formatter, 'body composition', 'dexa'))
  const dexaByDate = new Map(data.tests.dexa.map(scan => [scan.date, scan] as const))
  const vo2ByDate = new Map(data.tests.vo2max.map(test => [test.date, test] as const))
  const labDates = [...new Set([...dexaByDate.keys(), ...vo2ByDate.keys()])].sort((a, b) =>
    a.localeCompare(b),
  )
  if (labDates.length === 0) {
    block.appendChild(titleRow)
    block.appendChild(el('div', 'tri-ana-empty', text('no dexa scan logged')))
    block.appendChild(buildVo2TestRecord(context.formatter))
    return { element: block }
  }

  const datePicker = buildSelect<number>({
    id: 'tri-lab-date-menu',
    label: text('lab test date'),
    groups: [
      {
        options: labDates.map((date, index) => ({
          value: index,
          label: context.formatter.longDate(date),
        })),
      },
    ],
    selected: labDates.length - 1,
    onSelect: index => selectLabDate(index, true),
    className: 'tri-dexa-date',
  })
  titleRow.appendChild(datePicker.element)
  block.appendChild(titleRow)

  const sessions = labDates.map(date => {
    const session = el('div', 'tri-lab-session', undefined, { 'data-lab-date': date })
    const dexa = dexaByDate.get(date)
    session.appendChild(
      dexa
        ? buildDexaDetail(context.formatter, dexa)
        : el('div', 'tri-ana-empty', text('no dexa scan logged')),
    )
    session.appendChild(buildVo2TestRecord(context.formatter, vo2ByDate.get(date)))
    session.hidden = true
    block.appendChild(session)
    return session
  })
  const selectLabDate = (index: number, persist: boolean): void => {
    const date = labDates[index]
    if (!date) return
    datePicker.setSelected(index)
    for (const [sessionIndex, session] of sessions.entries())
      session.hidden = sessionIndex !== index
    if (persist) {
      try {
        localStorage.setItem(TRI_LAB_DATE_KEY, date)
      } catch {}
    }
  }
  selectLabDate(labDates.length - 1, false)
  return {
    element: block,
    mount: () => {
      try {
        const storedDate = localStorage.getItem(TRI_LAB_DATE_KEY)
        const storedIndex = labDates.findIndex(date => date === storedDate)
        if (storedIndex >= 0) selectLabDate(storedIndex, false)
      } catch {}
      return datePicker.mount()
    },
  }
}
