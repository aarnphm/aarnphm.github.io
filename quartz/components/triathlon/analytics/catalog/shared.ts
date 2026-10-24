import type { Analytics, PowerToWeightDurationS } from '../../../../plugins/stores/analytics'
import type { BestEffortCategory } from '../../../../util/best-efforts'
import type { TriathlonContext } from '../../runtime/context'
import type { TriathlonFormatter } from '../../runtime/formatter'
import { bestEffortCategoryLabel, bestEffortDisplay } from '../panels/best-efforts-format'

export const ANALYTICS_PANEL_ORDER = [
  'body',
  'dexa',
  'gauge',
  'recovery',
  'sleep',
  'vo2max',
  'lactate',
  'best-efforts',
  'power',
  'swim-power',
  'abilities',
  'distributions',
  'cardio',
  'pmc',
  'weekly',
  'effort',
  'heat',
  'readiness',
  'trend',
  'actions',
  'ftp',
] as const

export type AnalyticsPanelKey = (typeof ANALYTICS_PANEL_ORDER)[number]

export interface AnalyticsSummaryValue {
  label: string
  value: string
  detail?: string
}

export interface AnalyticsPanelSeries {
  label: string
  values: readonly number[]
  dates?: readonly string[]
}

export interface AnalyticsPanelContent {
  title: string
  values: readonly AnalyticsSummaryValue[]
  series?: readonly AnalyticsPanelSeries[]
  seriesDomain?: 'independent' | 'shared-zero'
}

export interface AnalyticsPanelView {
  element: HTMLElement
  mount?: () => () => void
}

export type AnalyticsPanelRender = HTMLElement | AnalyticsPanelView

export type AnalyticsPanelMount = (root: HTMLElement) => () => void

export const withPanelMount = (
  rendered: AnalyticsPanelRender,
  mounts: readonly AnalyticsPanelMount[],
): AnalyticsPanelView => {
  const view = rendered instanceof HTMLElement ? { element: rendered } : rendered
  return {
    element: view.element,
    mount: () => {
      const root = view.element.parentElement ?? view.element
      const cleanups = [view.mount?.(), ...mounts.map(mount => mount(root))].filter(
        (cleanup): cleanup is () => void => cleanup != null,
      )
      return () => {
        for (const cleanup of cleanups) cleanup()
      }
    },
  }
}

export interface AnalyticsPanelDefinition {
  key: AnalyticsPanelKey
  label: string
  search: string
  render: (data: Analytics, context: TriathlonContext) => AnalyticsPanelRender
  server: (data: Analytics, formatter: TriathlonFormatter) => AnalyticsPanelContent
}

export const value = (input: number | null | undefined, suffix = '', digits = 0): string =>
  input == null || !Number.isFinite(input) ? '—' : `${input.toFixed(digits)}${suffix}`

export const duration = (seconds: number | null | undefined): string => {
  if (seconds == null || !Number.isFinite(seconds)) return '—'
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.round((seconds % 3600) / 60)
  return hours > 0 ? `${hours}h ${minutes.toString().padStart(2, '0')}m` : `${minutes}m`
}

export const finite = (values: readonly (number | null | undefined)[]): number[] => {
  const result: number[] = []
  for (const item of values) if (item != null && Number.isFinite(item)) result.push(item)
  return result
}

export const powerToWeightValues = (data: Analytics, durationS: PowerToWeightDurationS): number[] =>
  finite(data.powerCurve.powerToWeight.points.map(point => point.efforts[durationS]?.wattsPerKg))

export const latestPowerToWeight = (
  data: Analytics,
  durationS: PowerToWeightDurationS,
): number | null =>
  data.powerCurve.powerToWeight.points.at(-1)?.efforts[durationS]?.wattsPerKg ?? null

type BestEffortMatch = (category: BestEffortCategory) => boolean

const runDistance =
  (label: string): BestEffortMatch =>
  category =>
    category.sport === 'run' && category.group === 'distance' && category.label === label

const bikePower =
  (durationS: number): BestEffortMatch =>
  category =>
    category.sport === 'bike' && category.group === 'power' && category.durationS === durationS

const BEST_EFFORT_SUMMARY: readonly BestEffortMatch[] = [
  runDistance('1K'),
  runDistance('5K'),
  runDistance('10K'),
  runDistance('Half marathon'),
  category => category.sport === 'bike' && category.group === 'longest',
  category => category.sport === 'bike' && category.group === 'elevation',
  bikePower(300),
  bikePower(1200),
]

const BEST_EFFORT_SERIES: readonly BestEffortMatch[] = [runDistance('5K'), bikePower(1200)]

const bestEffortSummaryLabel = (
  category: BestEffortCategory,
  formatter: TriathlonFormatter,
): string =>
  `${formatter.text(category.group === 'power' ? 'power' : category.sport)} · ${bestEffortCategoryLabel(category, formatter)}`

export const bestEffortsServer = (
  data: Analytics,
  formatter: TriathlonFormatter,
): AnalyticsPanelContent => {
  const { categories } = data.bestEfforts
  const values: AnalyticsSummaryValue[] = []
  for (const match of BEST_EFFORT_SUMMARY) {
    const category = categories.find(match)
    const best = category?.efforts.find(entry => entry.rank === 1)
    if (!category || !best) continue
    values.push({
      label: bestEffortSummaryLabel(category, formatter),
      value: bestEffortDisplay(category, best, formatter).primary,
      detail: formatter.longDate(best.date),
    })
  }
  const series: AnalyticsPanelSeries[] = []
  for (const match of BEST_EFFORT_SERIES) {
    const category = categories.find(match)
    const records = category?.efforts.filter(entry => entry.pr) ?? []
    if (!category || records.length === 0) continue
    series.push({
      label: bestEffortSummaryLabel(category, formatter),
      values: records.map(entry => entry.value),
      dates: records.map(entry => entry.date),
    })
  }
  return {
    title: formatter.text('best efforts'),
    values:
      values.length > 0
        ? values
        : [{ label: formatter.text('status'), value: formatter.text('no best efforts yet') }],
    series,
  }
}
