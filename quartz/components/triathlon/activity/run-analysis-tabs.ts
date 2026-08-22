export type RunAnalysisView = 'workout' | 'laps' | 'pace'

const RUN_ANALYSIS_VIEWS: readonly RunAnalysisView[] = ['workout', 'laps', 'pace']

const isRunAnalysisView = (value: string | undefined): value is RunAnalysisView =>
  RUN_ANALYSIS_VIEWS.some(view => view === value)

export const runAnalysisViewFromKey = (
  selected: RunAnalysisView,
  key: string,
  views: readonly RunAnalysisView[] = RUN_ANALYSIS_VIEWS,
): RunAnalysisView | null => {
  const current = views.indexOf(selected)
  if (current < 0 || views.length === 0) return null
  const next =
    key === 'Home'
      ? 0
      : key === 'End'
        ? views.length - 1
        : key === 'ArrowLeft'
          ? (current - 1 + views.length) % views.length
          : key === 'ArrowRight'
            ? (current + 1) % views.length
            : -1
  return views[next] ?? null
}

const selectRunAnalysisView = (
  analysis: HTMLElement,
  selected: RunAnalysisView,
  focus: boolean,
): void => {
  const tabs = Array.from(
    analysis.querySelectorAll<HTMLButtonElement>(
      ':scope > .tri-run-analysis-tabs [data-run-analysis-tab]',
    ),
  )
  const panels = Array.from(
    analysis.querySelectorAll<HTMLElement>(
      ':scope > .tri-run-analysis-stage > [data-run-analysis-panel]',
    ),
  )
  const views = tabs.flatMap(tab => {
    const view = tab.dataset.runAnalysisTab
    return isRunAnalysisView(view) ? [view] : []
  })
  if (
    tabs.length === 0 ||
    tabs.length !== panels.length ||
    views.length !== tabs.length ||
    !views.includes(selected) ||
    panels.some(panel => !isRunAnalysisView(panel.dataset.runAnalysisPanel))
  )
    return
  analysis.dataset.runAnalysisView = selected
  for (const tab of tabs) {
    const active = tab.dataset.runAnalysisTab === selected
    tab.setAttribute('aria-selected', String(active))
    tab.tabIndex = active ? 0 : -1
    if (active && focus) tab.focus({ preventScroll: true })
  }
  for (const panel of panels) {
    const active = panel.dataset.runAnalysisPanel === selected
    panel.hidden = !active
    panel.inert = !active
    panel.setAttribute('aria-hidden', String(!active))
  }
}

export const setupRunAnalysisTabs = (root: HTMLElement): (() => void) => {
  const analysisFromTab = (tab: HTMLElement): HTMLElement | null =>
    tab.closest<HTMLElement>('[data-run-analysis]')

  const onClick = (event: MouseEvent): void => {
    const tab =
      event.target instanceof Element
        ? event.target.closest<HTMLButtonElement>('[data-run-analysis-tab]')
        : null
    const selected = tab?.dataset.runAnalysisTab
    const analysis = tab ? analysisFromTab(tab) : null
    if (!tab || !analysis || !root.contains(analysis) || !isRunAnalysisView(selected)) return
    selectRunAnalysisView(analysis, selected, false)
  }

  const onKeyDown = (event: KeyboardEvent): void => {
    if (
      event.ctrlKey ||
      event.metaKey ||
      event.altKey ||
      event.isComposing ||
      event.repeat ||
      !(event.target instanceof HTMLButtonElement)
    )
      return
    const analysis = analysisFromTab(event.target)
    const selected = event.target.dataset.runAnalysisTab
    if (!analysis || !root.contains(analysis) || !isRunAnalysisView(selected)) return
    const views = Array.from(
      analysis.querySelectorAll<HTMLElement>(
        ':scope > .tri-run-analysis-tabs [data-run-analysis-tab]',
      ),
    ).flatMap(tab => {
      const view = tab.dataset.runAnalysisTab
      return isRunAnalysisView(view) ? [view] : []
    })
    const next = runAnalysisViewFromKey(selected, event.key, views)
    if (!next) return
    event.preventDefault()
    event.stopPropagation()
    selectRunAnalysisView(analysis, next, true)
  }

  for (const analysis of root.querySelectorAll<HTMLElement>('[data-run-analysis]')) {
    const selected = analysis.dataset.runAnalysisView
    selectRunAnalysisView(analysis, isRunAnalysisView(selected) ? selected : 'workout', false)
  }
  root.addEventListener('click', onClick)
  root.addEventListener('keydown', onKeyDown)
  return () => {
    root.removeEventListener('click', onClick)
    root.removeEventListener('keydown', onKeyDown)
  }
}
