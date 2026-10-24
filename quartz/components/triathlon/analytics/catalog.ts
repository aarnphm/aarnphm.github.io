import { physiologyPanelDefinitions } from './catalog/panels-physiology'
import { trainingPanelDefinitions } from './catalog/panels-training'
import {
  ANALYTICS_PANEL_ORDER,
  type AnalyticsPanelContent,
  type AnalyticsPanelDefinition,
  type AnalyticsPanelKey,
  type AnalyticsPanelRender,
  type AnalyticsPanelSeries,
  type AnalyticsPanelView,
  type AnalyticsSummaryValue,
} from './catalog/shared'

export {
  ANALYTICS_PANEL_ORDER,
  type AnalyticsPanelContent,
  type AnalyticsPanelDefinition,
  type AnalyticsPanelKey,
  type AnalyticsPanelRender,
  type AnalyticsPanelSeries,
  type AnalyticsPanelView,
  type AnalyticsSummaryValue,
}

const definitions: Record<AnalyticsPanelKey, AnalyticsPanelDefinition> = {
  ...physiologyPanelDefinitions,
  ...trainingPanelDefinitions,
}

export const ANALYTICS_CATALOG: readonly AnalyticsPanelDefinition[] = ANALYTICS_PANEL_ORDER.map(
  key => definitions[key],
)

export const analyticsPanelDefinition = (key: string): AnalyticsPanelDefinition | undefined =>
  ANALYTICS_CATALOG.find(panel => panel.key === key)

// Each chapter is one tab of the analytics page; every panel sits in exactly one.
export const ANALYTICS_CHAPTERS = [
  {
    key: 'load',
    shortcut: 'l',
    panels: ['pmc', 'weekly', 'effort', 'gauge', 'readiness', 'actions'],
  },
  {
    key: 'performance',
    shortcut: 'p',
    panels: [
      'best-efforts',
      'power',
      'swim-power',
      'trend',
      'ftp',
      'lactate',
      'vo2max',
      'abilities',
    ],
  },
  { key: 'zones', shortcut: 'z', panels: ['distributions', 'heat', 'cardio'] },
  { key: 'body', shortcut: 'b', panels: ['body', 'dexa', 'recovery', 'sleep'] },
] as const satisfies readonly {
  key: string
  shortcut: string
  panels: readonly AnalyticsPanelKey[]
}[]

export type AnalyticsChapterKey = (typeof ANALYTICS_CHAPTERS)[number]['key']

export const DEFAULT_ANALYTICS_CHAPTER: AnalyticsChapterKey = 'load'

export const analyticsChapterFromHash = (hash: string): AnalyticsChapterKey =>
  ANALYTICS_CHAPTERS.find(chapter => `#${chapter.key}` === hash)?.key ?? DEFAULT_ANALYTICS_CHAPTER
