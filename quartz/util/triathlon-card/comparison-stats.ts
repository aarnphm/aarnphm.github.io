import type { StravaActivityDetail } from '../../plugins/stores/strava'
import type { DetailCtx, TriNodeFactory } from './construction-contracts'
import { activityComparisonStats } from '../activity-comparison-metrics'
import { triText } from '../triathlon-i18n'
import { activityCompareColor, comparisonChartHead } from './comparison-plot'
import {
  COMPARISON_CONTEXT_STATS,
  COMPARISON_STAT_FULL_COVERAGE,
  COMPARISON_STAT_GROUPS,
  COMPARISON_STAT_SOURCE,
  comparisonMathNode,
  comparisonRelativeChange,
  comparisonStatTrend,
  type ActivityComparisonStatsView,
} from './comparison-stat-format'
import { activityComparisonStatSpecs } from './comparison-stat-specs'
import {
  activityAverageCadence,
  activityCadenceScale,
  excludesZeroPower,
  scrubDist,
} from './metric-text'

export const buildActivityComparisonStats = <N>(
  f: TriNodeFactory<N>,
  activities: readonly StravaActivityDetail[],
  ctx?: DetailCtx,
  view: ActivityComparisonStatsView = {},
): N => {
  const text = (key: string): string => triText(f.presentation.locale, key)
  const sport = activities[0]?.sport ?? 'bike'
  const window = view.window && view.window.endKm > view.window.startKm ? view.window : null
  const selection = view.scope === 'selection' && window != null
  const selected = activities.map((activity, index) => ({ activity, index, id: `${activity.id}` }))
  const columns = selected.filter(({ id }) => !view.hiddenIds?.has(id))
  const baseline = selected[0]
  const statsById = new Map(
    selected.map(({ activity, id }) => {
      const options = {
        health: view.health?.[activity.date] ?? null,
        athleteFtpWatts: ctx?.ftp ?? null,
        averageCadence: activityAverageCadence(activity),
        cadenceScale: activityCadenceScale(activity.sport),
        excludeZeroPower: excludesZeroPower(f.presentation),
      }
      const whole = activityComparisonStats(activity, options)
      if (!selection) return [id, whole]
      const scoped = activityComparisonStats(activity, { ...options, window })
      for (const key of COMPARISON_CONTEXT_STATS) scoped[key] ??= whole[key]
      return [id, scoped]
    }),
  )
  const windowKm = window ? window.endKm - window.startKm : 0
  const coverage = (id: string): number =>
    selection ? Math.min(1, (statsById.get(id)?.distance?.value ?? 0) / windowKm) : 1
  const partial = (id: string): boolean => coverage(id) < COMPARISON_STAT_FULL_COVERAGE

  const panel = f.el('section', 'tri-compare-stats-panel', undefined, {
    'data-compare-stats': '',
    'data-scope': selection ? 'selection' : 'whole',
    'data-baseline': baseline?.id ?? '',
    'aria-label': 'metrics',
    'data-i18n-aria-label': 'metrics',
  })
  const controls = f.el('div', 'tri-compare-stats-controls')
  const scopes = f.el('div', 'tri-compare-stats-scopes', undefined, {
    role: 'group',
    'aria-label': 'metrics scope',
    'data-i18n-aria-label': 'metrics scope',
  })
  for (const scope of ['whole', 'selection'] as const) {
    const attrs: Record<string, string> = {
      type: 'button',
      'data-compare-scope': scope,
      'aria-pressed': String((scope === 'selection') === selection),
      'data-i18n': scope,
    }
    if (scope === 'selection' && !window) attrs.disabled = ''
    f.add(scopes, f.el('button', 'tri-compare-stats-scope', text(scope), attrs))
  }
  f.add(controls, scopes)
  if (window)
    f.add(
      controls,
      f.el(
        'span',
        'tri-compare-stats-window',
        `${scrubDist(f.presentation, window.startKm, sport)} – ${scrubDist(f.presentation, window.endKm, sport)}`,
        { 'data-compare-stats-window': '' },
      ),
    )
  f.add(panel, comparisonChartHead(f, 'metrics', controls))

  const specs = activityComparisonStatSpecs(f.presentation)
  const table = f.el('table', 'tri-compare-stats', undefined, {
    'aria-label': baseline
      ? `${text('metrics')} (${text('baseline')}: ${baseline.activity.name})`
      : text('metrics'),
  })
  const head = f.el('thead')
  const headRow = f.el('tr')
  f.add(
    headRow,
    f.el('th', 'tri-compare-stats-corner', undefined, {
      scope: 'col',
      'aria-label': text('metric'),
    }),
  )
  for (const { activity, index, id } of columns) {
    const isBaseline = id === baseline?.id
    const attrs: Record<string, string> = {
      scope: 'col',
      'data-activity-id': id,
      'data-activity-index': `${index}`,
      style: `--tri-compare-color:${activityCompareColor(index)}`,
    }
    if (partial(id)) attrs['data-coverage'] = 'partial'
    const cell = f.el(
      'th',
      `tri-compare-stats-activity${isBaseline ? ' tri-compare-stats-activity--baseline' : ''}`,
      undefined,
      attrs,
    )
    const title = activity.name || activity.sport
    const heading = f.el('div', 'tri-compare-stats-heading', undefined, { title })
    f.add(
      heading,
      f.el('span', 'tri-compare-stats-swatch', undefined, { 'aria-hidden': 'true' }),
      f.el('span', 'tri-compare-stats-name', title),
    )
    f.add(cell, heading)
    if (partial(id))
      f.add(
        cell,
        f.el('span', 'tri-compare-stats-tag', `${Math.round(coverage(id) * 100)}%`, {
          title: text('partial coverage'),
        }),
      )
    f.add(headRow, cell)
  }
  f.add(head, headRow)
  f.add(table, head)

  let rowCount = 0
  for (const group of COMPARISON_STAT_GROUPS) {
    const rows = specs.filter(
      spec => spec.group === group && columns.some(({ id }) => statsById.get(id)?.[spec.key]),
    )
    if (rows.length === 0) continue
    const body = f.el('tbody', 'tri-compare-stats-group', undefined, { 'data-stat-group': group })
    const groupRow = f.el('tr')
    f.add(
      groupRow,
      f.el('th', 'tri-compare-stats-group-head', text(group), {
        scope: 'colgroup',
        colspan: `${columns.length + 1}`,
        'data-i18n': group,
      }),
    )
    f.add(body, groupRow)
    for (const spec of rows) {
      rowCount++
      const cells = columns.map(column => ({
        ...column,
        stat: statsById.get(column.id)?.[spec.key],
      }))
      const baseStat = baseline ? statsById.get(baseline.id)?.[spec.key] : undefined
      const sources = [...new Set(cells.flatMap(({ stat }) => (stat ? [stat.source] : [])))]
      const direction = typeof spec.better === 'function' ? spec.better(selection) : spec.better
      const label = spec.label(sport)
      const row = f.el('tr', 'tri-compare-stats-row', undefined, { 'data-stat': spec.key })
      const labelCell = f.el('th', 'tri-compare-stats-label', undefined, { scope: 'row' })
      const labelAttrs: Record<string, string> = {}
      if (spec.gloss) {
        labelAttrs['data-gloss'] = ''
        labelAttrs['data-gloss-def'] = spec.gloss
        labelAttrs.tabindex = '0'
      }
      f.add(
        labelCell,
        comparisonMathNode(f, 'tri-compare-stats-label-text', text(label), labelAttrs),
      )
      if (sources.length > 1) {
        const mixed = `${text('mixed sources')}: ${sources.map(source => text(COMPARISON_STAT_SOURCE[source])).join(', ')}`
        f.add(
          labelCell,
          comparisonMathNode(f, 'tri-compare-stats-mixed', '≠', {
            role: 'img',
            'aria-label': mixed,
            title: mixed,
          }),
        )
      }
      f.add(row, labelCell)
      for (const { id, stat } of cells) {
        const isBaseline = id === baseline?.id
        const attrs: Record<string, string> = { 'data-activity-id': id }
        if (stat) {
          attrs['data-source'] = stat.source
          attrs.title = [
            text(COMPARISON_STAT_SOURCE[stat.source]),
            ...(stat.estimate ? [text('estimate')] : []),
          ].join(' · ')
          if (stat.estimate) attrs['data-estimate'] = ''
        }
        const cell = f.el(
          'td',
          `tri-compare-stats-cell${isBaseline ? ' tri-compare-stats-cell--baseline' : ''}`,
          undefined,
          attrs,
        )
        if (!stat) {
          f.add(cell, f.el('span', 'tri-compare-stats-value tri-compare-stats-value--missing', '–'))
          f.add(row, cell)
          continue
        }
        const content = f.el('span', 'tri-compare-stats-content')
        f.add(
          content,
          comparisonMathNode(
            f,
            'tri-compare-stats-value',
            `${stat.estimate ? '≈' : ''}${spec.format(stat, sport)}`,
          ),
        )
        if (
          !isBaseline &&
          baseStat &&
          baseline &&
          (!spec.sameSourceDelta || stat.source === baseStat.source)
        ) {
          const delta = spec.delta(stat, baseStat, sport)
          const same = delta.startsWith('±')
          const relativeScalar = spec.relativeScalar ?? spec.scalar
          const relative =
            spec.relative && !same
              ? comparisonRelativeChange(
                  relativeScalar(stat, sport),
                  relativeScalar(baseStat, sport),
                )
              : null
          const trend =
            direction && !same && !partial(id) && !partial(baseline.id)
              ? comparisonStatTrend(direction, spec.scalar(stat), spec.scalar(baseStat))
              : undefined
          const deltaNode = f.el(
            'span',
            'tri-compare-stats-delta',
            undefined,
            trend ? { 'data-trend': trend } : undefined,
          )
          f.add(deltaNode, comparisonMathNode(f, 'tri-compare-stats-delta-abs', delta))
          if (relative)
            f.add(deltaNode, comparisonMathNode(f, 'tri-compare-stats-delta-rel', relative))
          f.add(content, deltaNode)
        }
        f.add(cell, content)
        f.add(row, cell)
      }
      f.add(body, row)
    }
    f.add(table, body)
  }

  if (rowCount === 0) {
    f.add(panel, f.el('p', 'tri-compare-empty', text('no data'), { 'data-i18n': 'no data' }))
    return panel
  }
  const scroll = f.el('div', 'tri-compare-stats-scroll', undefined, {
    tabindex: '0',
    'aria-label': 'metrics',
    'data-i18n-aria-label': 'metrics',
  })
  f.add(scroll, table)
  f.add(panel, scroll)
  return panel
}
