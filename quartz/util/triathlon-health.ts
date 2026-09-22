import type { TriNodeFactory } from './triathlon-card'
import { triText } from './triathlon-i18n'

export interface HealthRow {
  label: string
  value: string
  detail: string
  score?: number | null
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
    const row = f.el('tr', 'tri-health-score-row')
    const label = f.el('th', 'tri-health-tooltip-trigger', t(metric.label), {
      scope: 'row',
      tabindex: '0',
      'aria-describedby': id,
    })
    f.add(label, healthTooltip(f, id, metric.detail))
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
              'aria-valuetext': metric.value,
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
      f.add(row, cell, f.el('td', 'tri-sleep-contrib-val', metric.value))
    } else {
      f.add(
        row,
        f.el('td', 'tri-sleep-contrib-val tri-health-score-text', metric.value, { colspan: '2' }),
      )
    }
    f.add(body, row)
  }
  f.add(table, body)
  return table
}
