import { createElement, render } from 'preact'
import type { TrainingPeaksCalendar } from '../../../util/trainingpeaks-calendar'
import type { DatePicker } from '../../controls/date-picker'
import type { TriathlonContext } from '../runtime/context'
import {
  TRAINING_CALENDAR_API,
  TRAINING_CALENDAR_SESSION_API,
} from '../../../util/training-calendar-access'
import { parseTrainingPeaksCalendar } from '../../../util/trainingpeaks-calendar'
import { isRecord } from '../../../util/type-guards'
import { buildDatePicker } from '../../controls/date-picker'
import { calendarToday } from './display'
import {
  trainingAddDays,
  trainingAddMonths,
  trainingDate,
  trainingMonthDates,
  trainingMonthStart,
  trainingWeekStart,
} from './training-display'
import { TrainingCalendarView, type TrainingView } from './TrainingCalendar'
import { TrainingCalendarGate } from './TrainingCalendarGate'

// Seven day columns need about 52rem of views, plus the calendar's padding.
const WEEK_MIN_REM = 54

export interface TrainingCalendarController {
  activate(date?: string, view?: TrainingView): void
  deactivate(): void
  /** The selected local date and chosen view, retained when switching between periods. */
  location(): string
  dispose(): void
}

export const mountTrainingCalendar = (
  root: HTMLElement,
  context: TriathlonContext,
  onNavigate: (location: string) => void,
): TrainingCalendarController => {
  const content = root.querySelector<HTMLElement>('[data-training-content]')
  let calendar: TrainingPeaksCalendar | null = null
  const id = `${root.closest('.sidepanel-container') ? 'sidepanel-' : ''}${root.dataset.trainingId ?? 'tri-training-calendar'}`
  let anchorDate = calendarToday()
  // Without a chosen view, the calendar's width picks one and the hash leaves it out.
  let chosenView: TrainingView | null = null
  let shownView: TrainingView | null = null
  let selectedWorkoutId: string | null = null
  let mounted = false
  let loading = false
  let failed = false
  let loaded = false
  let authenticated = false
  let authenticating = false
  let authMessage = ''
  let active = false
  let revision = 0
  let expiryTimer: ReturnType<typeof setTimeout> | undefined
  const channel = new BroadcastChannel('training-calendar-session')
  const controller = new AbortController()
  let datePicker: DatePicker | undefined
  let datePickerCleanup: (() => void) | undefined
  let pickerLocale = context.presentation.locale
  let pickerMonth = false

  const removeDatePicker = (): void => {
    datePicker?.close()
    datePickerCleanup?.()
    datePicker?.wrap.remove()
    datePicker = undefined
    datePickerCleanup = undefined
  }
  const syncDatePicker = (): void => {
    const host = content?.querySelector<HTMLElement>('[data-training-date-picker]')
    if (!host) return
    if (
      datePicker &&
      (!host.contains(datePicker.wrap) ||
        pickerLocale !== context.presentation.locale ||
        pickerMonth !== (shownView === 'month'))
    )
      removeDatePicker()
    if (!datePicker) {
      pickerLocale = context.presentation.locale
      pickerMonth = shownView === 'month'
      const choose = (date: string): void => {
        selectDate(date, true)
        datePicker?.trigger.focus({ preventScroll: true })
      }
      datePicker = buildDatePicker({
        id: `${id}-date-picker`,
        label: context.formatter.text(
          pickerMonth ? 'choose training month' : 'choose training week',
        ),
        labels: context.formatter,
        selected: () =>
          pickerMonth ? trainingMonthStart(displayDate()) : trainingWeekStart(displayDate()),
        onSelect: choose,
        onClear: () => choose(calendarToday()),
      })
      datePicker.trigger.dataset.trainingDate = ''
      host.append(datePicker.wrap)
      datePickerCleanup = datePicker.mount()
    }
    const start = pickerMonth ? trainingMonthStart(displayDate()) : trainingWeekStart(displayDate())
    datePicker.text.textContent = pickerMonth
      ? `${context.formatter.month(start)} ${start.slice(0, 4)}`
      : context.formatter.shortDate(start)
    datePicker.trigger.dataset.value = start
  }

  const currentView = (): TrainingView => {
    // Details use the list temporarily; closing them restores the chosen overview.
    if (selectedWorkoutId) return 'list'
    if (chosenView) return chosenView
    const width = root.clientWidth
    const rem = Number.parseFloat(window.getComputedStyle(document.documentElement).fontSize) || 16
    return width > 0 && width < WEEK_MIN_REM * rem ? 'list' : 'week'
  }
  const displayDate = (): string =>
    calendar?.workouts.find(workout => workout.id === selectedWorkoutId)?.date ?? anchorDate
  const location = (): string => `training-${anchorDate}${chosenView ? `-${chosenView}` : ''}`

  const update = (): void => {
    if (!content || !active || controller.signal.aborted) return
    if (!selectedWorkoutId)
      content
        .querySelector<HTMLElement>('.tri-training-week')
        ?.style.removeProperty('--training-scroll-tail')
    root.dataset.trainingDate = displayDate()
    shownView = authenticated ? currentView() : null
    const view = authenticated
      ? createElement(TrainingCalendarView, {
          calendar,
          date: displayDate(),
          id,
          view: shownView ?? undefined,
          embedded: root.dataset.trainingEmbedded === 'true',
          panel: root.dataset.trainingPanel === 'true',
          presentation: context.presentation,
          loading,
          failed,
          selectedWorkoutId,
        })
      : createElement(TrainingCalendarGate, {
          id,
          busy: authenticating,
          message: authMessage,
          formatter: context.formatter,
        })
    // Saved locale and units can differ from the generated HTML, including accessible labels.
    if (!mounted) content.replaceChildren()
    render(view, content)
    mounted = true
    if (authenticated) syncDatePicker()
    else removeDatePicker()
  }
  const clearPrivateData = (message = ''): void => {
    revision++
    clearTimeout(expiryTimer)
    calendar = null
    selectedWorkoutId = null
    authenticated = false
    authenticating = false
    loading = false
    loaded = false
    failed = false
    authMessage = message
    update()
  }
  const load = async (): Promise<void> => {
    if (!authenticated || loaded || loading || controller.signal.aborted) return
    const current = revision
    loading = true
    failed = false
    update()
    try {
      const response = await fetch(TRAINING_CALENDAR_API, {
        signal: controller.signal,
        cache: 'no-store',
        credentials: 'same-origin',
      })
      if (current !== revision) return
      if (response.status === 401) {
        clearPrivateData('Your session expired. Enter the password again.')
        return
      }
      if (!response.ok) throw new Error(`Training calendar returned HTTP ${response.status}`)
      const data: unknown = await response.json()
      if (current !== revision || controller.signal.aborted) return
      calendar = parseTrainingPeaksCalendar(data)
      if (data !== null && !calendar) throw new Error('Training calendar payload is invalid')
      loaded = true
    } catch {
      if (current === revision && !controller.signal.aborted) failed = true
    } finally {
      if (current === revision) {
        loading = false
        update()
      }
    }
  }
  const authenticate = async (password?: string): Promise<void> => {
    if (authenticating || controller.signal.aborted) return
    const current = ++revision
    authenticating = true
    authMessage = ''
    update()
    try {
      const response = await fetch(TRAINING_CALENDAR_SESSION_API, {
        method: password === undefined ? 'GET' : 'POST',
        ...(password === undefined
          ? {}
          : {
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ password }),
            }),
        signal: controller.signal,
        credentials: 'same-origin',
        cache: 'no-store',
      })
      if (current !== revision || controller.signal.aborted) return
      if (!response.ok) {
        clearPrivateData(
          response.status === 401
            ? password === undefined
              ? ''
              : 'Incorrect password.'
            : response.status === 429
              ? 'Too many attempts. Try again in 15 minutes.'
              : 'Calendar access is unavailable. Try again later.',
        )
        return
      }
      const session: unknown = await response.json()
      if (current !== revision || controller.signal.aborted) return
      if (
        !isRecord(session) ||
        typeof session.expiresAt !== 'number' ||
        session.expiresAt <= Date.now()
      )
        throw new Error('Invalid calendar session')
      authenticated = true
      authenticating = false
      clearTimeout(expiryTimer)
      expiryTimer = setTimeout(
        () => clearPrivateData('Your session expired. Enter the password again.'),
        session.expiresAt - Date.now(),
      )
      update()
      if (password !== undefined) channel.postMessage('unlocked')
      await load()
    } catch {
      if (current === revision && !controller.signal.aborted)
        clearPrivateData('Calendar access is unavailable. Try again later.')
    }
  }
  const lock = async (): Promise<void> => {
    clearPrivateData()
    authenticating = true
    update()
    channel.postMessage('locked')
    try {
      const response = await fetch(TRAINING_CALENDAR_SESSION_API, {
        method: 'DELETE',
        credentials: 'same-origin',
        cache: 'no-store',
        signal: controller.signal,
      })
      if (!response.ok) throw new Error('Session could not be revoked')
    } catch {
      if (!controller.signal.aborted) {
        authMessage = 'Calendar hidden. Could not end the session; retry locking.'
      }
    } finally {
      authenticating = false
      update()
    }
  }
  const onSubmit = (event: SubmitEvent): void => {
    if (
      !(event.target instanceof HTMLFormElement) ||
      !event.target.matches('[data-training-login]')
    )
      return
    event.preventDefault()
    const input = event.target.querySelector<HTMLInputElement>('input[name="password"]')
    if (!input) return
    const password = input.value
    input.value = ''
    void authenticate(password)
  }
  channel.onmessage = event => {
    if (event.data !== 'locked' && event.data !== 'unlocked') return
    clearPrivateData()
    if (event.data === 'unlocked' && active) void authenticate()
  }
  const onPageHide = (): void => clearPrivateData()
  const onPageShow = (event: PageTransitionEvent): void => {
    if (event.persisted && active) void authenticate()
  }
  const selectDate = (date: string, navigate: boolean): void => {
    if (!trainingDate(date)) return
    datePicker?.close()
    selectedWorkoutId = null
    anchorDate = date
    update()
    if (navigate) onNavigate(location())
  }
  const selectView = (next: TrainingView): void => {
    const views = content?.querySelector<HTMLElement>('.tri-calendar-views')
    anchorDate = displayDate()
    if (next !== 'list' || (views && window.getComputedStyle(views).visibility === 'hidden'))
      selectedWorkoutId = null
    chosenView = next
    update()
    if (selectedWorkoutId) scrollWorkoutToTop(selectedWorkoutId)
    onNavigate(location())
  }
  const scrollWorkoutToTop = (workoutId: string): void => {
    if (shownView === 'month') {
      const views = content?.querySelector<HTMLElement>('.tri-calendar-views')
      if (views && window.getComputedStyle(views).visibility !== 'hidden')
        views
          .querySelector<HTMLElement>(`[data-training-workout-open="${CSS.escape(workoutId)}"]`)
          ?.scrollIntoView({ block: 'nearest', inline: 'nearest', behavior: 'instant' })
      return
    }
    const card = content?.querySelector<HTMLElement>(
      `[data-training-workout="${CSS.escape(workoutId)}"]`,
    )
    const sessions = card?.closest<HTMLElement>('.tri-training-day-sessions')
    const day = sessions?.closest<HTMLElement>('.tri-training-day')
    const header = day?.querySelector<HTMLElement>('.tri-training-day-header')
    const list = day?.closest<HTMLElement>('.tri-training-week')
    if (!card || !sessions || !day || !header || !list) return
    const isList = shownView === 'list'
    const scroller = isList ? list : sessions
    if (isList) list.style.removeProperty('--training-scroll-tail')
    else {
      // A narrow week scrolls sideways; bring the workout's day column into view.
      const left =
        day.getBoundingClientRect().left - list.getBoundingClientRect().left - list.clientLeft
      if (left < 0 || left + day.offsetWidth > list.clientWidth)
        list.scrollTo({ left: list.scrollLeft + left, behavior: 'instant' })
    }
    const top = Math.max(
      0,
      scroller.scrollTop +
        card.getBoundingClientRect().top -
        scroller.getBoundingClientRect().top -
        scroller.clientTop -
        (isList ? header.offsetHeight : 0),
    )
    if (isList) {
      // Add only the trailing space needed to align a workout near the end of the list.
      const tail = Math.max(0, top - (list.scrollHeight - list.clientHeight))
      list.style.setProperty('--training-scroll-tail', `${tail}px`)
    }
    scroller.scrollTo({ top, behavior: 'instant' })
  }
  const closeDetail = (): boolean => {
    if (!selectedWorkoutId) return false
    const previous = selectedWorkoutId
    selectedWorkoutId = null
    update()
    scrollWorkoutToTop(previous)
    content
      ?.querySelector<HTMLButtonElement>(
        `[data-training-workout-open="${CSS.escape(previous)}"][aria-controls]`,
      )
      ?.focus({ preventScroll: true })
    return true
  }
  const openDetail = (workoutId: string): void => {
    if (workoutId === selectedWorkoutId) {
      closeDetail()
      return
    }
    if (
      !calendar?.workouts.some(
        workout =>
          workout.id === workoutId &&
          (shownView === 'month'
            ? trainingMonthDates(anchorDate).includes(workout.date)
            : trainingWeekStart(workout.date) === trainingWeekStart(displayDate())),
      )
    )
      return
    selectedWorkoutId = workoutId
    update()
    const detail = content?.querySelector<HTMLElement>('.tri-training-detail')
    if (detail) detail.scrollTop = 0
    scrollWorkoutToTop(workoutId)
    const views = content?.querySelector<HTMLElement>('.tri-calendar-views')
    if (views && window.getComputedStyle(views).visibility === 'hidden' && detail) {
      const top = detail.getBoundingClientRect().top
      if (top < 0 || top > window.innerHeight - 80)
        detail.scrollIntoView({ block: 'start', behavior: 'instant' })
    }
    detail
      ?.querySelector<HTMLButtonElement>('[data-training-detail-close]')
      ?.focus({ preventScroll: true })
  }
  const onClick = (event: MouseEvent): void => {
    if (!(event.target instanceof Element)) return
    if (event.target.closest('[data-training-lock]')) {
      void lock()
      return
    }
    if (event.target.closest('[data-training-detail-close]')) {
      closeDetail()
      return
    }
    const viewSelect = event.target.closest<HTMLButtonElement>('[data-training-view-select]')
    const nextView = viewSelect?.dataset.trainingViewSelect
    if (nextView === 'list' || nextView === 'week' || nextView === 'month') {
      selectView(nextView)
      return
    }
    const shift = event.target.closest<HTMLButtonElement>('[data-training-shift]')
    if (shift) {
      const direction = shift.dataset.trainingShift === '-1' ? -1 : 1
      selectDate(
        shownView === 'month'
          ? trainingAddMonths(displayDate(), direction)
          : trainingAddDays(displayDate(), direction * 7),
        true,
      )
      return
    }
    if (event.target.closest('[data-training-today]')) selectDate(calendarToday(), true)
    else if (event.target.closest('[data-training-retry]')) void load()
    if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey || event.button !== 0)
      return
    const opener =
      event.target.closest<HTMLButtonElement>('[data-training-workout-open]') ??
      (event.target.closest('a, button')
        ? null
        : event.target
            .closest('[data-training-workout]')
            ?.querySelector<HTMLButtonElement>('[data-training-workout-open]'))
    if (opener?.dataset.trainingWorkoutOpen) openDetail(opener.dataset.trainingWorkoutOpen)
  }
  const onKeydown = (event: KeyboardEvent): void => {
    if (event.key !== 'Escape' || !(event.target instanceof Element)) return
    if (datePicker?.panel.contains(event.target)) {
      event.preventDefault()
      event.stopPropagation()
      datePicker.close()
      datePicker.trigger.focus({ preventScroll: true })
      return
    }
    if (!closeDetail()) return
    event.preventDefault()
    event.stopPropagation()
  }
  root.addEventListener('click', onClick)
  root.addEventListener('submit', onSubmit)
  root.addEventListener('keydown', onKeydown)
  window.addEventListener('pagehide', onPageHide)
  window.addEventListener('pageshow', onPageShow)
  const unsubscribe = context.events.subscribe('presentation', update)
  // Re-render only when the width-picked view changes; a chosen view stays put.
  const resize = new ResizeObserver(() => {
    if (shownView && shownView !== currentView()) update()
  })
  resize.observe(root)
  return {
    activate: (date, view) => {
      active = true
      if (date && trainingDate(date)) {
        anchorDate = date
        chosenView = view ?? null
      } else if (view) chosenView = view
      selectedWorkoutId = null
      update()
      if (!authenticated) void authenticate()
      else if (!loaded && !failed) void load()
    },
    deactivate: () => {
      active = false
      selectedWorkoutId = null
      shownView = null
      removeDatePicker()
      if (content && mounted) render(null, content)
      mounted = false
    },
    location,
    dispose: () => {
      controller.abort()
      revision++
      calendar = null
      clearTimeout(expiryTimer)
      channel.close()
      unsubscribe()
      resize.disconnect()
      removeDatePicker()
      root.removeEventListener('click', onClick)
      root.removeEventListener('submit', onSubmit)
      root.removeEventListener('keydown', onKeydown)
      window.removeEventListener('pagehide', onPageHide)
      window.removeEventListener('pageshow', onPageShow)
      if (content && mounted) render(null, content)
    },
  }
}
