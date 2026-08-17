export type SitePerformanceSource = 'cursor' | 'popover' | 'timeline'

export interface SitePerformanceSample {
  duration: number
  source: SitePerformanceSource
}

export const SITE_PERFORMANCE_SAMPLE_EVENT = 'site:performance-sample'

declare global {
  interface WindowEventMap {
    [SITE_PERFORMANCE_SAMPLE_EVENT]: CustomEvent<SitePerformanceSample>
  }
}

export const beginSitePerformanceSample = (): number | null =>
  document.documentElement.dataset.sitePerformanceDebug === 'true' ? performance.now() : null

export const endSitePerformanceSample = (
  source: SitePerformanceSource,
  startedAt: number | null,
): void => {
  if (startedAt === null) return
  window.dispatchEvent(
    new CustomEvent(SITE_PERFORMANCE_SAMPLE_EVENT, {
      detail: { duration: performance.now() - startedAt, source },
    }),
  )
}
