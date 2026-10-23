import type { SitePerformanceSource } from '../performance-sample'
import { sceneProbe } from '../../visitor-trail/performance-probe'

const DEBUG_VALUE = 'performance'

export const DEFAULT_DEBUG_QUERIES = ['debug', 'trail-debug', 'tri-debug'] as const

export interface PerformanceDebugReading {
  label: string
  value: string
}

export interface PerformanceDebugProbe {
  read(): readonly PerformanceDebugReading[]
  stop(): void
}

export interface PerformanceDebugProfile {
  label: string
  /** `?<query>=performance` opens the panel on load. */
  query: string | readonly string[]
  sources: readonly SitePerformanceSource[]
  /** Text beside the frame rate; the cursor mode shows when this returns nothing. */
  state?: () => string | null | undefined
  /** Surface readings, created when the panel opens and stopped when it closes. */
  probe?: () => PerformanceDebugProbe
}

export const matchesPerformanceDebugQuery = (
  search: string,
  query: string | readonly string[] = DEFAULT_DEBUG_QUERIES,
): boolean => {
  if (!search) return false
  const params = new URLSearchParams(search)
  const keys =
    typeof query === 'string'
      ? new Set<string>([query, ...DEFAULT_DEBUG_QUERIES])
      : new Set<string>([...query, ...DEFAULT_DEBUG_QUERIES])
  for (const key of keys) {
    if (params.get(key) === DEBUG_VALUE) return true
  }
  return false
}

export const sitePerformanceProfile = (doc: Document = document): PerformanceDebugProfile => {
  const trailHost = doc.querySelector<HTMLElement>('.homepage-trail-scene')
  const triathlonRoot = doc.querySelector<HTMLElement>('.triathlon')
  const hasTimeline = Boolean(triathlonRoot || doc.querySelector('.tri-bars'))
  const label = trailHost
    ? 'trail performance debug'
    : triathlonRoot
      ? 'triathlon performance debug'
      : 'performance debug'
  const sources: readonly SitePerformanceSource[] = hasTimeline
    ? ['cursor', 'timeline', 'popover']
    : ['cursor', 'popover']
  return {
    label,
    query: DEFAULT_DEBUG_QUERIES,
    sources,
    state: () => {
      const currentTrail = doc.querySelector<HTMLElement>('.homepage-trail-scene') ?? trailHost
      if (currentTrail && !currentTrail.hidden) return 'trail'
      const currentTri = doc.querySelector<HTMLElement>('.triathlon') ?? triathlonRoot
      if (currentTri) {
        const openModal = currentTri.className.match(
          /tri-(analytics|calc|calendar|map|training)-open/,
        )?.[1]
        if (openModal) return openModal
        if (currentTri.dataset?.triView) return currentTri.dataset.triView
      }
      return null
    },
    ...(trailHost ? { probe: () => sceneProbe(doc) } : {}),
  }
}
