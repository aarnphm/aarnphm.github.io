import { setupPerformanceDebug } from './performance-debug/panel'
import { sitePerformanceProfile } from './performance-debug/profile'

export { setupPerformanceDebug }
export {
  DEFAULT_DEBUG_QUERIES,
  matchesPerformanceDebugQuery,
  sitePerformanceProfile,
  type PerformanceDebugReading,
  type PerformanceDebugProbe,
  type PerformanceDebugProfile,
} from './performance-debug/profile'
export {
  estimateFrameBudget,
  frameChartRatio,
  monotoneSlopes,
  smoothWindow,
  summarizeFrameDurations,
  type ChartSeries,
  type FrameSummary,
} from './performance-debug/sampling'

export const setupSitePerformanceDebug = (doc: Document = document): (() => void) =>
  setupPerformanceDebug(sitePerformanceProfile(doc))
