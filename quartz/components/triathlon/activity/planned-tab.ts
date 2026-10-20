// The planned tab of a workout analysis section. The server renders it for every activity linked
// to a TrainingPeaks workout, and the client rebuilds it with the same attributes when a card is
// rebuilt, so the tab row does not move after load. The panel holds no plan data until the
// training calendar session is unlocked.

export const PLANNED_LABEL = 'planned'
export const PLANNED_SHORT_LABEL = 'PL'
export const PLANNED_TAB_CLASS = 'tri-map-tab tri-workout-analysis-tab'
export const PLANNED_PANEL_CLASS = 'tri-workout-analysis-panel tri-planned-panel'

/** How the section's tabs show their labels: shortcut and label spans, short text or full text. */
export type PlannedTabForm = 'expanded' | 'short' | 'full'

export const plannedTabForm = (hasShortcut: boolean, text: string): PlannedTabForm =>
  hasShortcut ? 'expanded' : text.trim().length <= 2 ? 'short' : 'full'

export const plannedTabStyle = (label: string): string =>
  `--tri-map-tab-shortcut-width: ${PLANNED_SHORT_LABEL.length}ch; --tri-map-tab-label-width: ${label.length}ch`

/** The section builder names its tabs `tri-workout-analysis-<activity id>-<view>-tab`. */
export const workoutAnalysisOwner = (
  tabId: string | undefined,
): { base: string; activityId: number } | null => {
  const match = /^(tri-workout-analysis-(\d+))-[a-z-]+-tab$/.exec(tabId ?? '')
  return match ? { base: match[1], activityId: Number(match[2]) } : null
}

export const plannedTabAttributes = (base: string): Record<string, string> => ({
  id: `${base}-planned-tab`,
  type: 'button',
  role: 'tab',
  tabindex: '-1',
  'aria-label': PLANNED_LABEL,
  'aria-controls': `${base}-planned-panel`,
  'aria-selected': 'false',
  'data-workout-analysis-tab': 'planned',
})

export const plannedPanelAttributes = (base: string): Record<string, string> => ({
  id: `${base}-planned-panel`,
  role: 'tabpanel',
  tabindex: '0',
  hidden: '',
  inert: '',
  'aria-hidden': 'true',
  'aria-labelledby': `${base}-planned-tab`,
  'data-workout-analysis-panel': 'planned',
})
