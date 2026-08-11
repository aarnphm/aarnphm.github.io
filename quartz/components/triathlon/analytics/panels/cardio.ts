import type { Analytics } from '../../../../plugins/stores/analytics'
import type { CardioKey } from '../../../../plugins/stores/analytics'
import type { TriathlonContext } from '../../runtime/context'
import { axisNumber } from '../../../../util/triathlon-card'
import { niceStep } from '../../../../util/triathlon-card'
import { el } from '../../runtime/dom'
import { svg } from '../../runtime/dom'
import { anaTitle } from '../shared'
import { clampN } from '../shared'
import { markGloss } from '../shared'
import { polyD } from '../shared'

export type CardioSeriesPoint = { date: string; value: number }

export const cardioSeriesOf = (
  cardio: Analytics['engine']['cardio'],
  key: CardioKey,
): CardioSeriesPoint[] => {
  if (key === 'rhr') return cardio.rhrSeries.map(point => ({ date: point.date, value: point.rhr }))
  if (key === 'hrv') return cardio.hrvSeries.map(point => ({ date: point.date, value: point.hrv }))
  if (key === 'ef') return cardio.efSeries.map(point => ({ date: point.date, value: point.ef }))
  return cardio.decouplingSeries.map(point => ({ date: point.date, value: point.pct }))
}

export const cardioValueText = (value: number, unit: string): string => {
  const decimals = unit === 'bpm' || unit === 'ms' ? 0 : unit === '%' ? 1 : 2
  const text = value.toFixed(decimals)
  return unit === '%' ? `${text}%` : unit ? `${text} ${unit}` : text
}

export const buildCardio = (data: Analytics, context: TriathlonContext): HTMLElement => {
  const text = (key: string): string => context.formatter.text(key)
  const block = el('div', 'tri-engine-cardio')
  const c = data.engine.cardio
  const head = el('div', 'tri-engine-cardio-head')
  head.appendChild(anaTitle(context.formatter, 'cardiovascular health', 'ef'))
  const readout = el('div', 'tri-chart-readout tri-cardio-readout', undefined, {
    'aria-hidden': 'true',
  })
  readout.appendChild(el('span', 'tri-cardio-readout-date'))
  const readoutTable = el('table', 'tri-cardio-readout-table')
  const readoutBody = el('tbody')
  for (const metric of c.metrics) {
    const row = el('tr')
    row.append(
      el('th', 'tri-cardio-readout-metric', text(metric.label), { scope: 'row' }),
      el('td', 'tri-cardio-readout-value', '—', { 'data-cardio-metric': metric.key }),
    )
    readoutBody.appendChild(row)
  }
  readoutTable.appendChild(readoutBody)
  readout.appendChild(readoutTable)
  block.append(head, readout)
  if (!c.metrics.length || c.metrics.every(m => m.value == null)) {
    block.appendChild(el('div', 'tri-ana-empty', text('no heart data yet')))
    return block
  }
  const dates = [
    ...new Set(c.metrics.flatMap(metric => cardioSeriesOf(c, metric.key).map(p => p.date))),
  ].sort()
  const domainStartMs = Date.parse(`${dates[0] ?? data.meta.windowFrom}T00:00:00Z`)
  const domainEndMs = Date.parse(`${dates[dates.length - 1] ?? data.meta.windowTo}T00:00:00Z`)
  const domainSpanMs = Math.max(1, domainEndMs - domainStartMs)
  const xAt = (date: string): number =>
    clampN(((Date.parse(`${date}T00:00:00Z`) - domainStartMs) / domainSpanMs) * 100, 0, 100)
  const glossOf: Record<string, string> = {
    rhr: 'rhr',
    hrv: 'hrv',
    ef: 'ef',
    decoupling: 'decouple',
  }
  for (const m of c.metrics) {
    const row = el('div', 'tri-engine-row')
    row.dataset.metric = m.key
    const meta = el('div', 'tri-engine-row-meta')
    meta.appendChild(
      markGloss(el('span', 'tri-engine-row-k', text(m.label)), glossOf[m.key] ?? 'ef'),
    )
    const status = el('div', 'tri-engine-row-status')
    const val = el(
      'span',
      'tri-engine-row-v',
      m.value != null ? cardioValueText(m.value, m.unit) : '—',
    )
    val.title = text(m.note)
    status.append(
      val,
      el(
        'span',
        `tri-engine-row-dir ${m.dir === 'improving' ? 'tri-dir-up' : m.dir === 'declining' ? 'tri-dir-down' : 'tri-dir-flat'}`,
        m.dir === 'improving' ? '▲' : m.dir === 'declining' ? '▼' : m.dir === 'stable' ? '■' : '',
      ),
    )
    meta.appendChild(status)
    row.appendChild(meta)
    const points = cardioSeriesOf(c, m.key)
    const ys = points.map(point => point.value)
    if (ys.length > 1) {
      const rawMin = Math.min(...ys)
      const rawMax = Math.max(...ys)
      const minimumPadding = m.key === 'ef' ? 0.02 : m.key === 'decoupling' ? 0.5 : 2
      const padding = Math.max((rawMax - rawMin) * 0.1, minimumPadding)
      const step = niceStep(rawMax - rawMin + padding * 2, 3)
      const domainMin = Math.floor((rawMin - padding) / step) * step
      const domainMax = Math.ceil((rawMax + padding) / step) * step
      const chartHeight = 42
      const chartTop = 3
      const chartBottom = 39
      const yAt = (value: number): number =>
        chartBottom -
        ((value - domainMin) / Math.max(step, domainMax - domainMin)) * (chartBottom - chartTop)
      const ticks: number[] = []
      for (let tick = domainMin; tick <= domainMax + step * 1e-6; tick += step)
        ticks.push(Math.round(tick * 1e6) / 1e6)
      const s = svg('svg', {
        class: 'tri-engine-spark',
        viewBox: `0 0 100 ${chartHeight}`,
        preserveAspectRatio: 'none',
        role: 'slider',
        tabindex: 0,
        'aria-label': text(m.label),
        'aria-orientation': 'horizontal',
        'aria-valuemin': 1,
        'aria-valuemax': Math.max(1, dates.length),
        'aria-valuenow': Math.max(1, dates.length),
      })
      for (const tick of ticks)
        s.appendChild(
          svg('line', {
            class: 'tri-engine-grid',
            x1: 0,
            y1: yAt(tick),
            x2: 100,
            y2: yAt(tick),
            'aria-hidden': 'true',
          }),
        )
      const pathPoints: [number, number][] = points.map(point => [
        xAt(point.date),
        yAt(point.value),
      ])
      s.appendChild(
        svg('path', {
          d: polyD(pathPoints),
          class: `tri-elev-line ${m.key === 'hrv' ? 'tri-line-bike' : m.key === 'ef' ? 'tri-line-swim' : m.key === 'rhr' ? 'tri-line-run' : ''}`,
        }),
      )
      s.appendChild(
        svg('line', { x1: 0, y1: chartTop, x2: 0, y2: chartBottom, class: 'tri-ana-cursor' }),
      )
      row.appendChild(s)
      const axis = el('div', 'tri-engine-domain', undefined, { 'aria-hidden': 'true' })
      for (const tick of ticks) {
        const label = el('span', 'tri-engine-domain-tick', axisNumber(tick, step))
        label.style.top = `${((yAt(tick) / chartHeight) * 100).toFixed(2)}%`
        axis.appendChild(label)
      }
      row.appendChild(axis)
    } else row.append(el('span', 'tri-engine-spark'), el('span', 'tri-engine-domain'))
    block.appendChild(row)
  }
  return block
}
