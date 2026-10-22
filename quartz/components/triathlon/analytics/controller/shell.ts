import type { Dispatch } from '../../../../functional'
import type { Analytics } from '../../../../plugins/stores/analytics'
import type { ActivityKind, StravaActivityDetail } from '../../../../plugins/stores/strava'
import type { DetailPayload } from '../../activity/data'
import type { TriathlonContext } from '../../runtime/context'
import type { AnalyticsMessage, AnalyticsModel } from '../model'
import { activityComparisonEligible } from '../../../../util/triathlon-card'
import { TRI_ANALYTICS_BOOT_CLASS } from '../boot'
import {
  analyticsChapterFromHash,
  DEFAULT_ANALYTICS_CHAPTER,
  type AnalyticsChapterKey,
  type AnalyticsPanelRender,
} from '../catalog'

export interface AnalyticsSession {
  root: HTMLElement
  context: TriathlonContext
  btn: HTMLElement | null
  panel: HTMLElement
  scrim: HTMLElement | null
  closeBtn: HTMLElement | null
  results: HTMLElement | null
  compareToggle: HTMLButtonElement | null
  compareSubmit: HTMLButtonElement | null
  pageMode: boolean
  body: HTMLElement | null
  detail: HTMLElement | null
  live: boolean
  sourceData: Analytics | null
  data: Analytics | null
  detailData: DetailPayload | null
  detailPromise: Promise<boolean> | null
  panelCleanups: Map<HTMLElement, () => void>
  compareCleanup: (() => void) | null
  activityCleanup: (() => void) | null
  detailGeneration: number
  comparisonScrollTop: number
  renderFrame: number
  chapterTabs: HTMLButtonElement[]
  chapterPanels: HTMLElement[]
  chapter: AnalyticsChapterKey
  renderedVersion: WeakMap<HTMLElement, number>
  renderVersion: number
  finishPageBoot: () => void
  inCompareMode: () => boolean
  comparisonIds: () => readonly string[]
  panelView: (rendered: AnalyticsPanelRender) => { element: HTMLElement; mount?: () => () => void }
  renderBlock: (block: HTMLElement, d: Analytics) => void
  renderChapter: () => void
  render: (source: Analytics) => void
  load: () => void
  setBarBack: (back: boolean) => void
  closeDetail: () => void
  setCompareMode: (enabled: boolean) => void
  toMain: () => void
  close: () => void
  loadDetails: () => Promise<boolean>
  compareActivityEligible: (activity: StravaActivityDetail | undefined) => boolean
  selectedCompareActivities: () => StravaActivityDetail[]
  compareSport: () => ActivityKind | null
  showDetail: () => void
  showActivity: (id: string) => void
  comparisonScroller: (comparison: Element) => HTMLElement | undefined
  showComparison: (chartScrollTop?: number) => void
  resultItems: () => HTMLElement[]
  setSel: (i: number) => void
  toggleCompareActivity: (id: string) => void
  activate: (it: HTMLElement | undefined) => void
  renderSearch: () => void
  refreshPicker: () => void
  onResultsClick: (event: MouseEvent) => void
  onPickerKey: (event: KeyboardEvent) => void
  enterCompare: () => void
  onCompareSubmit: () => void
  inComparison: () => boolean
  onCompareToggle: () => void
  onClose: () => void
  open: () => void
  selectChapter: (next: AnalyticsChapterKey, updateHash: boolean, focus: boolean) => void
  onChapterClick: (event: MouseEvent) => void
  onChapterKey: (event: KeyboardEvent) => void
  onHashChange: () => void
  onMethodToggle: (event: MouseEvent) => void
  onComparisonFill: (event: Event) => void
  onKey: (event: KeyboardEvent) => void
  program: {
    dispatch: Dispatch<AnalyticsMessage>
    retrieve: () => AnalyticsModel
    stop: () => void
  }
  rangeCleanup: () => void
  powerActivityCleanup: () => void
  onUnitChange: () => void
}

export function initSession(s: AnalyticsSession) {
  s.body = s.root.querySelector<HTMLElement>('.tri-ana-body')

  s.detail = s.root.querySelector<HTMLElement>('.tri-ana-detail')

  s.live = true

  s.sourceData = null

  s.data = null

  s.detailData = null

  s.detailPromise = null

  s.panelCleanups = new Map<HTMLElement, () => void>()

  s.compareCleanup = null

  s.activityCleanup = null

  s.detailGeneration = 0

  s.comparisonScrollTop = 0

  s.renderFrame = 0

  s.chapterTabs = Array.from(
    s.panel.querySelectorAll<HTMLButtonElement>('[data-analytics-chapter-tab]'),
  )

  s.chapterPanels = Array.from(s.panel.querySelectorAll<HTMLElement>('[data-analytics-chapter]'))

  s.chapter = s.pageMode
    ? analyticsChapterFromHash(window.location.hash)
    : DEFAULT_ANALYTICS_CHAPTER

  // Only the open chapter renders; the others catch up to the current version when opened.
  s.renderedVersion = new WeakMap<HTMLElement, number>()

  s.renderVersion = 0

  s.finishPageBoot = (): void => {
    if (s.pageMode) document.documentElement.classList.remove(TRI_ANALYTICS_BOOT_CLASS)
  }

  s.inCompareMode = (): boolean => s.program.retrieve().mode === 'compare'

  s.comparisonIds = (): readonly string[] => s.program.retrieve().comparisonActivityIds
}

export function mountShellChrome(s: AnalyticsSession) {
  // The comparison's back lives in the bar: the compare toggle on the page, the close in the dialog.
  s.setBarBack = (back: boolean) => {
    const target = s.pageMode ? s.compareToggle : s.closeBtn
    if (!target) return
    const key = back ? 'go back' : target === s.compareToggle ? 'compare activities' : 'Close'
    target.setAttribute('aria-label', s.context.formatter.text(key))
    if (target === s.compareToggle) {
      target.dataset.i18nAriaLabel = key
      return
    }
    target.toggleAttribute('data-site-cursor-close', !back)
    target.toggleAttribute('data-site-cursor-action', back)
  }

  s.closeDetail = () => {
    s.detailGeneration += 1
    s.compareCleanup?.()
    s.compareCleanup = null
    s.activityCleanup?.()
    s.activityCleanup = null
    s.panel.classList.remove('tri-analytics--comparison')
    s.panel.classList.remove('tri-analytics--detail')
    s.setBarBack(false)
    if (s.detail) {
      s.detail.replaceChildren()
      s.detail.setAttribute('aria-hidden', 'true')
    }
    if (s.results)
      s.results.setAttribute(
        'aria-hidden',
        String(!s.inCompareMode() && !s.panel.classList.contains('tri-analytics--searching')),
      )
  }

  s.setCompareMode = (enabled: boolean) => {
    if (s.inCompareMode() !== enabled) s.program.dispatch({ type: 'toggle-compare' })
    s.panel.classList.toggle('tri-analytics--compare', enabled)
    s.compareToggle?.setAttribute('aria-pressed', String(enabled))
  }

  s.toMain = () => {
    s.closeDetail()
    s.setCompareMode(false)
    s.panel.classList.remove('tri-analytics--searching')
    if (s.results) {
      s.results.replaceChildren()
      s.results.setAttribute('aria-hidden', 'true')
    }
    s.program.dispatch({ type: 'reset' })
  }

  s.close = () => {
    const wasOpen = s.root.classList.contains('tri-analytics-open')
    s.detailGeneration += 1
    s.root.classList.remove('tri-analytics-open')
    s.panel.setAttribute('aria-hidden', 'true')
    s.program.dispatch({ type: 'close' })
    if (wasOpen && !s.pageMode) s.btn?.focus({ preventScroll: true })
  }

  s.loadDetails = (): Promise<boolean> => {
    if (s.detailData) return Promise.resolve(true)
    if (s.detailPromise) return s.detailPromise
    const p = s.root.dataset.detailPath
    if (!p) return Promise.resolve(false)
    s.detailPromise = s.context.resources.detail.load(p).then(result => {
      if (!s.live || result.status !== 'ready') {
        if (result.status === 'error') s.detailPromise = null
        return false
      }
      s.detailData = result.value
      return true
    })
    return s.detailPromise
  }

  s.compareActivityEligible = (activity: StravaActivityDetail | undefined): boolean =>
    activity != null && activityComparisonEligible(activity)

  s.selectedCompareActivities = (): StravaActivityDetail[] =>
    s.comparisonIds().flatMap(id => {
      const activity = s.detailData?.details[id]
      return s.compareActivityEligible(activity) && activity ? [activity] : []
    })

  s.compareSport = (): ActivityKind | null => s.selectedCompareActivities()[0]?.sport ?? null

  s.showDetail = () => {
    s.panel.classList.add('tri-analytics--detail')
    s.detail?.setAttribute('aria-hidden', 'false')
    s.results?.setAttribute('aria-hidden', 'true')
    s.body?.scrollTo({ top: 0 })
  }
}
