import { render } from 'preact'
import type {
  TrainingPeaksCalendar,
  TrainingPeaksCalendarWorkout,
} from '../../../util/trainingpeaks-calendar'
import type { TriathlonPresentation } from '../../../util/triathlon-presentation'
import type { TriathlonFormatter } from '../runtime/formatter'
import {
  TRAINING_CALENDAR_API,
  TRAINING_CALENDAR_SESSION_API,
} from '../../../util/training-calendar-access'
import { parseTrainingPeaksCalendar } from '../../../util/trainingpeaks-calendar'
import { trainingPeaksExecution } from '../../../util/trainingpeaks-execution'
import { calendarToday } from '../calendar/display'
import { parseSwimDescription } from '../calendar/swim-structure'
import { TrainingStructure } from '../calendar/TrainingStructure'
import { TrainingStructureHover } from '../calendar/TrainingStructureHover'
import { TrainingInstructions, TrainingWorkoutTotals } from '../calendar/TrainingWorkoutPlan'
import { createTriathlonFormatter } from '../runtime/formatter'
import {
  PLANNED_LABEL,
  PLANNED_PANEL_CLASS,
  PLANNED_SHORT_LABEL,
  PLANNED_TAB_CLASS,
  plannedPanelAttributes,
  plannedTabAttributes,
  plannedTabForm,
  plannedTabStyle,
  workoutAnalysisOwner,
} from './planned-tab'
import { selectWorkoutAnalysisView } from './workout-analysis-tabs'

interface PlannedIndex {
  calendar: TrainingPeaksCalendar
  byActivity: Map<number, TrainingPeaksCalendarWorkout>
}

interface PlannedMount {
  base: string
  activityId: number
  tab: HTMLButtonElement
  panel: HTMLElement
  // Added on the client from the unlocked calendar, without a server-rendered tab.
  injected: boolean
}

// The calendar is private: it loads only through the password session that the training calendar
// opens, so planned workouts never enter the public pages or activity payloads.
const loadPlanned = async (signal: AbortSignal): Promise<PlannedIndex | null> => {
  const init: RequestInit = { signal, cache: 'no-store', credentials: 'same-origin' }
  const session = await fetch(TRAINING_CALENDAR_SESSION_API, init)
  if (!session.ok) return null
  const response = await fetch(TRAINING_CALENDAR_API, init)
  if (!response.ok) return null
  const calendar = parseTrainingPeaksCalendar(await response.json())
  if (!calendar) return null
  const byActivity = new Map<number, TrainingPeaksCalendarWorkout>()
  for (const workout of calendar.workouts)
    if (workout.activity) byActivity.set(workout.activity.id, workout)
  return { calendar, byActivity }
}

// A swim set list read from free text is drawn only when it adds up to the coach's planned volume;
// otherwise the parser missed part of the coach's format and the chart would misstate the plan.
const PLAN_TOLERANCE = 0.05
const showsStructure = (workout: TrainingPeaksCalendarWorkout): boolean => {
  if (workout.structure) return true
  if (workout.sport !== 'swim') return false
  const swim = parseSwimDescription(workout.description)
  const [total, planned] =
    swim?.unit === 'meters'
      ? [swim.meters, workout.planned.distanceMeters]
      : swim?.unit === 'seconds'
        ? [swim.seconds, workout.planned.durationSeconds]
        : [0, null]
  return planned !== null && planned > 0 && Math.abs(total - planned) <= planned * PLAN_TOLERANCE
}

const PlannedWorkout = ({
  id,
  workout,
  calendar,
  formatter,
}: {
  id: string
  workout: TrainingPeaksCalendarWorkout
  calendar: TrainingPeaksCalendar
  formatter: TriathlonFormatter
}) => {
  const today = calendarToday()
  const time = workout.startTimePlanned?.slice(0, 5)
  return (
    <TrainingStructureHover id={`${id}-tip`} resetKey={workout.id}>
      <article
        class="tri-planned tri-training-workout-detail"
        data-sport={workout.sport}
        data-execution={trainingPeaksExecution(workout, today).status}
      >
        <div class="tri-planned-head">
          <strong class="tri-planned-title">{workout.title}</strong>
          <span class="tri-planned-source">
            TrainingPeaks · {formatter.shortDate(workout.date)}
            {time && time !== '00:00' && ` · ${time}`}
          </span>
        </div>
        <div class="tri-planned-body">
          {showsStructure(workout) && (
            <TrainingStructure workout={workout} zones={calendar} formatter={formatter} />
          )}
          <div class="tri-planned-summary">
            <TrainingWorkoutTotals workout={workout} formatter={formatter} today={today} />
            <TrainingInstructions workout={workout} formatter={formatter} />
          </div>
        </div>
      </article>
    </TrainingStructureHover>
  )
}

const PlannedLocked = ({ formatter }: { formatter: TriathlonFormatter }) => (
  <p class="tri-planned tri-planned-locked">
    {formatter.text('Enter the password to view the TrainingPeaks calendar.')}{' '}
    <a href="/triathlon/calendar">{formatter.text('unlock calendar')}</a>
  </p>
)

const TABS = ':scope > .tri-workout-analysis-tabs'
const STAGE = ':scope > .tri-workout-analysis-stage'

const sectionOwner = (section: HTMLElement) =>
  workoutAnalysisOwner(section.querySelector(`${TABS} [data-workout-analysis-tab]`)?.id)

const setAttributes = (element: Element, attributes: Record<string, string>): void => {
  for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, value)
}

// The same tab and panel the server renders, for a card the client has rebuilt.
const createShell = (
  section: HTMLElement,
  base: string,
): { tab: HTMLButtonElement; panel: HTMLElement } | null => {
  const tabs = section.querySelector<HTMLElement>(TABS)
  const stage = section.querySelector<HTMLElement>(STAGE)
  const model = tabs?.querySelector<HTMLElement>('[data-workout-analysis-tab]')
  if (!tabs || !stage || !model) return null
  const tab = document.createElement('button')
  setAttributes(tab, { ...plannedTabAttributes(base), class: PLANNED_TAB_CLASS })
  const form = plannedTabForm(
    model.querySelector('.tri-map-tab-shortcut') !== null,
    model.textContent ?? '',
  )
  if (form === 'expanded') {
    tab.setAttribute('style', plannedTabStyle(PLANNED_LABEL))
    for (const [name, text] of [
      ['tri-map-tab-shortcut', PLANNED_SHORT_LABEL],
      ['tri-map-tab-label', PLANNED_LABEL],
    ]) {
      const span = document.createElement('span')
      span.className = name
      span.textContent = text
      span.setAttribute('aria-hidden', 'true')
      tab.append(span)
    }
  } else tab.textContent = form === 'short' ? PLANNED_SHORT_LABEL : PLANNED_LABEL
  const panel = document.createElement('div')
  setAttributes(panel, { ...plannedPanelAttributes(base), class: PLANNED_PANEL_CLASS })
  tabs.prepend(tab)
  stage.prepend(panel)
  return { tab, panel }
}

export const setupPlannedWorkouts = (
  root: HTMLElement,
  signal: AbortSignal,
  presentation: () => TriathlonPresentation,
): (() => void) => {
  let request = new AbortController()
  // `done` with no data means the session is locked or the calendar is unavailable.
  let status: 'idle' | 'loading' | 'done' = 'idle'
  let data: PlannedIndex | null = null
  // Activities whose planned tab the server rendered. Card rebuilds drop the tab, and the
  // mutation callback adds it back before the next paint.
  const known = new Set<number>()
  const mounted = new Map<HTMLElement, PlannedMount>()
  const pending = new Set<HTMLElement>()
  const channel = new BroadcastChannel('training-calendar-session')

  const paint = (entry: PlannedMount): void => {
    const formatter = createTriathlonFormatter(presentation())
    const label = formatter.text(PLANNED_LABEL)
    entry.tab.setAttribute('aria-label', label)
    const full = entry.tab.querySelector<HTMLElement>('.tri-map-tab-label')
    if (full) {
      full.textContent = label
      entry.tab.style.setProperty('--tri-map-tab-label-width', `${label.length}ch`)
    } else if (entry.tab.textContent !== PLANNED_SHORT_LABEL) entry.tab.textContent = label
    const workout = data?.byActivity.get(entry.activityId)
    entry.panel.dataset.plannedState = workout ? 'ready' : status === 'done' ? 'locked' : status
    render(
      workout && data ? (
        <PlannedWorkout
          id={entry.base}
          workout={workout}
          calendar={data.calendar}
          formatter={formatter}
        />
      ) : status === 'done' ? (
        <PlannedLocked formatter={formatter} />
      ) : null,
      entry.panel,
    )
  }

  const unmount = (section: HTMLElement): void => {
    const entry = mounted.get(section)
    if (!entry) return
    mounted.delete(section)
    if (entry.injected && section.dataset.workoutAnalysisView === 'planned')
      selectWorkoutAnalysisView(section, 'workout', false)
    render(null, entry.panel)
    if (!entry.injected) return
    entry.panel.remove()
    entry.tab.remove()
  }

  const register = (section: HTMLElement, entry: PlannedMount): void => {
    mounted.set(section, entry)
    paint(entry)
  }

  const load = (): void => {
    if (status !== 'idle') return
    status = 'loading'
    const { signal: aborted } = request
    void loadPlanned(aborted)
      .catch(() => null)
      .then(result => {
        if (signal.aborted || aborted.aborted) return
        status = 'done'
        data = result
        for (const entry of mounted.values()) paint(entry)
        for (const section of pending) attachLoaded(section)
      })
  }

  // A card the server rendered without a planned tab, such as a note embed: the tab appears
  // only once the unlocked calendar names its activity.
  const attachLoaded = (section: HTMLElement): void => {
    const owner = sectionOwner(section)
    if (!data || !owner || !section.isConnected || mounted.has(section)) return
    if (!data.byActivity.has(owner.activityId)) return
    pending.delete(section)
    const shell = createShell(section, owner.base)
    if (shell) register(section, { ...owner, ...shell, injected: true })
  }

  const visibility = new IntersectionObserver(
    entries => {
      for (const entry of entries)
        if (entry.isIntersecting && entry.target instanceof HTMLElement) {
          visibility.unobserve(entry.target)
          load()
          if (data) attachLoaded(entry.target)
        }
    },
    { rootMargin: '200px 0px' },
  )

  const adopt = (section: HTMLElement): void => {
    if (mounted.has(section)) return
    const owner = sectionOwner(section)
    if (!owner) return
    const tab = section.querySelector<HTMLButtonElement>(
      `${TABS} [data-workout-analysis-tab="planned"]`,
    )
    const panel = section.querySelector<HTMLElement>(
      `${STAGE} > [data-workout-analysis-panel="planned"]`,
    )
    if (tab && panel) {
      known.add(owner.activityId)
      register(section, { ...owner, tab, panel, injected: false })
    } else if (known.has(owner.activityId)) {
      const shell = createShell(section, owner.base)
      if (shell) register(section, { ...owner, ...shell, injected: false })
    } else pending.add(section)
    visibility.observe(section)
  }

  const scan = (element: Element): void => {
    if (element instanceof HTMLElement && element.matches('[data-workout-analysis]')) adopt(element)
    for (const section of element.querySelectorAll<HTMLElement>('[data-workout-analysis]'))
      adopt(section)
  }
  const additions = new MutationObserver(records => {
    for (const section of mounted.keys()) if (!section.isConnected) unmount(section)
    for (const section of pending) if (!section.isConnected) pending.delete(section)
    for (const record of records)
      for (const node of record.addedNodes) if (node instanceof Element) scan(node)
  })

  // Locking the calendar anywhere, or leaving the page, drops the private plan from this page.
  // A lock is a known answer, so the panels show the locked prompt without a new request.
  const purge = (next: 'idle' | 'done'): void => {
    request.abort()
    request = new AbortController()
    status = next
    data = null
    for (const [section, entry] of mounted)
      if (entry.injected) {
        unmount(section)
        pending.add(section)
      } else paint(entry)
  }
  channel.onmessage = event => {
    if (event.data === 'locked') purge('done')
    else if (event.data === 'unlocked') {
      purge('idle')
      load()
    }
  }
  const onPageHide = (): void => purge('idle')
  const onPageShow = (event: PageTransitionEvent): void => {
    if (event.persisted) load()
  }
  const repaint = (): void => {
    for (const entry of mounted.values()) paint(entry)
  }
  window.addEventListener('pagehide', onPageHide)
  window.addEventListener('pageshow', onPageShow)
  window.addEventListener('tri:locale', repaint)
  window.addEventListener('tri:unit', repaint)
  additions.observe(root, { childList: true, subtree: true })
  scan(root)
  return () => {
    channel.close()
    window.removeEventListener('pagehide', onPageHide)
    window.removeEventListener('pageshow', onPageShow)
    window.removeEventListener('tri:locale', repaint)
    window.removeEventListener('tri:unit', repaint)
    visibility.disconnect()
    additions.disconnect()
    request.abort()
    for (const section of mounted.keys()) unmount(section)
  }
}
