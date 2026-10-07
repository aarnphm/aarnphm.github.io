import type { TriNodeFactory } from './triathlon-card'
import { triText } from './triathlon-i18n'

export interface HealthStatus {
  tone: 'good' | 'watch' | 'alert' | 'neutral' | 'info'
  label: string
}

// Oura rates both scores and contributors on this scale.
// https://support.ouraring.com/hc/en-us/articles/360025577993-Activity-Score
export function ouraScoreStatus(value: number | null | undefined): HealthStatus | null {
  if (value == null || !Number.isFinite(value) || value < 0 || value > 100) return null
  if (value >= 85) return { tone: 'good', label: 'optimal' }
  if (value >= 70) return { tone: 'good', label: 'good' }
  if (value >= 60) return { tone: 'watch', label: 'fair' }
  return { tone: 'alert', label: 'pay attention' }
}

export function healthRangeStatus(
  value: number | null | undefined,
  low: number | null | undefined,
  high: number | null | undefined,
): HealthStatus | null {
  if (
    value == null ||
    low == null ||
    high == null ||
    !Number.isFinite(value) ||
    !Number.isFinite(low) ||
    !Number.isFinite(high) ||
    value < 0 ||
    low < 0 ||
    high < low
  )
    return null
  return value < low
    ? { tone: 'watch', label: 'below target' }
    : value > high
      ? { tone: 'alert', label: 'above target' }
      : { tone: 'good', label: 'within target' }
}

export interface HealthRow {
  label: string
  value: string
  detail: string
  score?: number | null
  status?: HealthStatus | null
}

export function healthTooltip<N>(f: TriNodeFactory<N>, id: string, text: string): N {
  return f.el('span', 'tri-day-analytics-detail tri-health-tooltip', text, { id, role: 'tooltip' })
}

export function healthScoreTable<N>(
  f: TriNodeFactory<N>,
  date: string,
  key: string,
  rows: HealthRow[],
): N {
  const t = (key: string): string => triText(f.presentation.locale, key)
  const table = f.el('table', 'tri-health-score-table', undefined, { 'aria-label': t(key) })
  const columns = f.el('colgroup')
  f.add(
    columns,
    f.el('col', 'tri-health-label-column'),
    f.el('col'),
    f.el('col', 'tri-health-value-column'),
  )
  f.add(table, columns)
  const body = f.el('tbody')
  for (const [index, metric] of rows.entries()) {
    const id = `tri-health-${date}-${key.replaceAll(' ', '-')}-${index}`
    const status = metric.status
    const valueText = `${metric.value}${status ? ` · ${t(status.label)}` : ''}`
    const row = f.el('tr', 'tri-health-score-row', undefined, {
      'data-health-tone': status?.tone ?? 'neutral',
    })
    const label = f.el('th', 'tri-health-tooltip-trigger', t(metric.label), {
      scope: 'row',
      tabindex: '0',
      'aria-describedby': id,
    })
    f.add(label, healthTooltip(f, id, `${metric.detail}${status ? `\n${t(status.label)}` : ''}`))
    f.add(row, label)
    if ('score' in metric) {
      const cell = f.el('td', 'tri-health-score-bar')
      const score = metric.score
      const bar = f.el(
        'div',
        'tri-sleep-contrib-bar',
        undefined,
        score == null
          ? { 'aria-hidden': 'true' }
          : {
              role: 'meter',
              'aria-label': t(metric.label),
              'aria-valuemin': '0',
              'aria-valuemax': '100',
              'aria-valuenow': String(score),
              'aria-valuetext': valueText,
            },
      )
      if (score != null)
        f.add(
          bar,
          f.el('span', 'tri-sleep-contrib-fill', undefined, {
            style: `width:${Math.max(0, Math.min(100, score))}%`,
          }),
        )
      f.add(cell, bar)
      f.add(row, cell, f.el('td', 'tri-sleep-contrib-val', metric.value, { title: valueText }))
    } else {
      f.add(
        row,
        f.el('td', 'tri-sleep-contrib-val tri-health-score-text', metric.value, {
          colspan: '2',
          title: valueText,
        }),
      )
    }
    f.add(body, row)
  }
  f.add(table, body)
  return table
}
