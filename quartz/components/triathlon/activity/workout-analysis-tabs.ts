export type WorkoutAnalysisView = 'workout' | 'laps' | 'pace'

const WORKOUT_ANALYSIS_VIEWS: readonly WorkoutAnalysisView[] = ['workout', 'laps', 'pace']

const isWorkoutAnalysisView = (value: string | undefined): value is WorkoutAnalysisView =>
  WORKOUT_ANALYSIS_VIEWS.some(view => view === value)

export const workoutAnalysisViewFromKey = (
  selected: WorkoutAnalysisView,
  key: string,
  views: readonly WorkoutAnalysisView[] = WORKOUT_ANALYSIS_VIEWS,
): WorkoutAnalysisView | null => {
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

const selectWorkoutAnalysisView = (
  analysis: HTMLElement,
  selected: WorkoutAnalysisView,
  focus: boolean,
): void => {
  const tabs = Array.from(
    analysis.querySelectorAll<HTMLButtonElement>(
      ':scope > .tri-workout-analysis-tabs [data-workout-analysis-tab]',
    ),
  )
  const panels = Array.from(
    analysis.querySelectorAll<HTMLElement>(
      ':scope > .tri-workout-analysis-stage > [data-workout-analysis-panel]',
    ),
  )
  const views = tabs.flatMap(tab => {
    const view = tab.dataset.workoutAnalysisTab
    return isWorkoutAnalysisView(view) ? [view] : []
  })
  if (
    tabs.length === 0 ||
    tabs.length !== panels.length ||
    views.length !== tabs.length ||
    !views.includes(selected) ||
    panels.some(panel => !isWorkoutAnalysisView(panel.dataset.workoutAnalysisPanel))
  )
    return
  analysis.dataset.workoutAnalysisView = selected
  for (const tab of tabs) {
    const active = tab.dataset.workoutAnalysisTab === selected
    tab.setAttribute('aria-selected', String(active))
    tab.tabIndex = active ? 0 : -1
    if (active && focus) tab.focus({ preventScroll: true })
  }
  for (const panel of panels) {
    const active = panel.dataset.workoutAnalysisPanel === selected
    panel.hidden = !active
    panel.inert = !active
    panel.setAttribute('aria-hidden', String(!active))
  }
}

export const setupWorkoutAnalysisTabs = (root: HTMLElement): (() => void) => {
  const analysisFromTab = (tab: HTMLElement): HTMLElement | null =>
    tab.closest<HTMLElement>('[data-workout-analysis]')

  const onClick = (event: MouseEvent): void => {
    const tab =
      event.target instanceof Element
        ? event.target.closest<HTMLButtonElement>('[data-workout-analysis-tab]')
        : null
    const selected = tab?.dataset.workoutAnalysisTab
    const analysis = tab ? analysisFromTab(tab) : null
    if (!tab || !analysis || !root.contains(analysis) || !isWorkoutAnalysisView(selected)) return
    selectWorkoutAnalysisView(analysis, selected, false)
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
    const selected = event.target.dataset.workoutAnalysisTab
    if (!analysis || !root.contains(analysis) || !isWorkoutAnalysisView(selected)) return
    const views = Array.from(
      analysis.querySelectorAll<HTMLElement>(
        ':scope > .tri-workout-analysis-tabs [data-workout-analysis-tab]',
      ),
    ).flatMap(tab => {
      const view = tab.dataset.workoutAnalysisTab
      return isWorkoutAnalysisView(view) ? [view] : []
    })
    const next = workoutAnalysisViewFromKey(selected, event.key, views)
    if (!next) return
    event.preventDefault()
    event.stopPropagation()
    selectWorkoutAnalysisView(analysis, next, true)
  }

  for (const analysis of root.querySelectorAll<HTMLElement>('[data-workout-analysis]')) {
    const selected = analysis.dataset.workoutAnalysisView
    selectWorkoutAnalysisView(
      analysis,
      isWorkoutAnalysisView(selected) ? selected : 'workout',
      false,
    )
  }
  root.addEventListener('click', onClick)
  root.addEventListener('keydown', onKeyDown)
  return () => {
    root.removeEventListener('click', onClick)
    root.removeEventListener('keydown', onKeyDown)
  }
}
