import type { ActivityHealth, StravaActivityDetail } from '../../plugins/stores/strava'
import type { TriNodeFactory } from './construction-contracts'
import { dur, formatMl } from './metric-text'

export type ActivityFueling = NonNullable<StravaActivityDetail['fueling']>

export const formatFuelingSource = (fueling: ActivityFueling): string => {
  if (fueling.source === 'manual') return 'manual'
  if (fueling.source === 'garmin+wahoo') return 'Garmin + Wahoo'
  const value = fueling.sourceDevice
  const clean = value?.trim()
  if (fueling.source === 'wahoo') {
    if (!clean) return 'Wahoo'
    return clean.toLowerCase().includes('wahoo') ? clean : `Wahoo ${clean}`
  }
  if (!clean) return 'Garmin'
  return clean.toLowerCase().includes('garmin') ? clean : `Garmin ${clean}`
}

export const recoveryRows = (h: ActivityHealth): [string, string][] => {
  const rows: [string, string][] = []
  if (h.readiness != null) rows.push(['readiness', `${h.readiness}`])
  if (h.sleepScore != null) rows.push(['sleep', `${h.sleepScore}`])
  if (h.sleepDurationS != null) rows.push(['slept', dur(h.sleepDurationS)])
  if (h.hrv != null) rows.push(['hrv', `${h.hrv} ms`])
  if (h.rhr != null) rows.push(['resting hr', `${h.rhr} bpm`])
  if (h.tempDeviationC != null)
    rows.push(['temp', `${h.tempDeviationC > 0 ? '+' : ''}${h.tempDeviationC.toFixed(1)}°C`])
  if (h.windKph != null)
    rows.push([
      'wind',
      `${h.windKph} km/h${h.windDir ? ` ${h.windDir}` : ''}${h.windGustKph != null ? ` / gust ${h.windGustKph}` : ''}`,
    ])
  if (h.totalCalories != null)
    rows.push(['total burn', `${Math.round(h.totalCalories).toLocaleString('en-US')} kcal`])
  if (h.activeCalories != null)
    rows.push(['active burn', `${Math.round(h.activeCalories).toLocaleString('en-US')} kcal`])
  return rows
}

export const fuelingRows = (f: ActivityFueling): [string, string][] => {
  const rows: [string, string][] = []
  const consumed: string[] = []
  if (f.caloriesConsumed != null) consumed.push(`${Math.round(f.caloriesConsumed)} kcal`)
  if (f.carbsConsumedG != null) consumed.push(`${Math.round(f.carbsConsumedG)} g carb`)
  if (consumed.length > 0) rows.push(['consumed', consumed.join(' / ')])
  if (f.fluidMl != null) rows.push(['fluid', formatMl(f.fluidMl)])

  const target: string[] = []
  if (f.carbsRecommendedG != null) target.push(`${Math.round(f.carbsRecommendedG)} g carb`)
  if (f.fluidRecommendedMl != null) target.push(formatMl(f.fluidRecommendedMl))
  if (target.length > 0) rows.push(['target', target.join(' / ')])

  if (f.sweatLossMl != null) rows.push(['sweat', formatMl(f.sweatLossMl)])
  if (f.sodiumLossMg != null)
    rows.push(['sodium loss', `${Math.round(f.sodiumLossMg).toLocaleString('en-US')} mg`])
  if (rows.length > 0) rows.push(['source', formatFuelingSource(f)])
  return rows
}

export const statRow = <N>(
  f: TriNodeFactory<N>,
  label: string,
  value: string,
  attrs?: Record<string, string>,
): N => {
  const tr = f.el('tr', undefined, undefined, { ...attrs, 'data-stat-key': label })
  const description = attrs?.['data-source-description']
  const cell = f.el('td', 'tri-act-stat-v', description ? undefined : value)
  if (description)
    f.add(
      cell,
      f.el('span', 'tri-act-source', value, {
        'data-gloss': '',
        'data-gloss-def': description,
        tabindex: '0',
        'aria-label': `${value}: ${description}`,
      }),
    )
  f.add(tr, f.el('th', 'tri-act-stat-k', label, { 'data-i18n': label }), cell)
  return tr
}

export const statsTable = <N>(
  f: TriNodeFactory<N>,
  rows: [string, string][],
  rowAttrs?: (label: string, index: number) => Record<string, string> | undefined,
): N => {
  const table = f.el('table', 'tri-act-stats')
  const tbody = f.el('tbody')
  rows.forEach(([k, v], index) => f.add(tbody, statRow(f, k, v, rowAttrs?.(k, index))))
  f.add(table, tbody)
  return table
}

export const buildFueling = <N>(f: TriNodeFactory<N>, fueling: ActivityFueling): N | null => {
  const rows = fuelingRows(fueling)
  if (rows.length === 0) return null
  const wrap = f.el('div', 'tri-act-health tri-act-fueling')
  f.add(wrap, f.el('span', 'tri-act-health-h', 'fueling'), statsTable(f, rows))
  return wrap
}

export const buildReservedFueling = <N>(f: TriNodeFactory<N>): N => {
  const wrap = f.el('div', 'tri-act-health tri-act-fueling tri-act-fueling--empty', undefined, {
    'aria-hidden': 'true',
  })
  f.add(wrap, f.el('span', 'tri-act-health-h', 'fueling'), statsTable(f, []))
  return wrap
}

export const buildRecovery = <N>(f: TriNodeFactory<N>, h: ActivityHealth): N | null => {
  const rows = recoveryRows(h)
  if (rows.length === 0) return null
  const wrap = f.el('div', 'tri-act-health')
  f.add(wrap, f.el('span', 'tri-act-health-h', 'recovery'), statsTable(f, rows))
  return wrap
}
