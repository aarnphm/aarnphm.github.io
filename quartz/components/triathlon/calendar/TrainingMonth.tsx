import { autoUpdate, computePosition, flip, offset, shift, size } from '@floating-ui/dom'
import { useLayoutEffect, useRef, useState } from 'preact/hooks'
import type {
  TrainingPeaksCalendar,
  TrainingPeaksCalendarWorkout,
} from '../../../util/trainingpeaks-calendar'
import type { TriathlonFormatter } from '../runtime/formatter'
import { trainingPeaksExecution } from '../../../util/trainingpeaks-execution'
import { calendarDateLabel, calendarWeekdayLabel } from './display'
import { trainingDayCovered, trainingDuration } from './training-display'
import { TrainingDayNotes } from './TrainingDayNotes'
import { TrainingStructureHover } from './TrainingStructureHover'
import { TrainingWorkoutCard } from './TrainingWorkout'

interface Preview {
  workout: TrainingPeaksCalendarWorkout
  anchor: HTMLButtonElement
}

export const TrainingMonth = ({
  calendar,
  dates,
  month,
  workouts,
  formatter,
  today,
  id,
  detailId,
  selectedWorkoutId,
}: {
  calendar: TrainingPeaksCalendar | null
  dates: string[]
  month: string
  workouts: TrainingPeaksCalendarWorkout[]
  formatter: TriathlonFormatter
  today: string
  id: string
  detailId: string
  selectedWorkoutId: string | null
}) => {
  const { text, presentation } = formatter
  const [preview, setPreview] = useState<Preview | null>(null)
  const popover = useRef<HTMLDivElement>(null)
  const hideTimer = useRef<ReturnType<typeof setTimeout>>()
  const previewDismissed = useRef(false)
  const dismissedAnchor = useRef<HTMLButtonElement | null>(null)
  const cancelHide = () => clearTimeout(hideTimer.current)
  const dismiss = () => {
    cancelHide()
    if (popover.current?.matches(':popover-open')) popover.current.hidePopover()
    setPreview(null)
  }
  const scheduleHide = () => {
    cancelHide()
    hideTimer.current = setTimeout(() => setPreview(null), 180)
  }
  const show = (workout: TrainingPeaksCalendarWorkout, anchor: HTMLButtonElement) => {
    cancelHide()
    if (workout.id !== selectedWorkoutId && preview?.anchor !== anchor)
      setPreview({ workout, anchor })
  }

  useLayoutEffect(() => {
    previewDismissed.current = true
    if (selectedWorkoutId)
      dismissedAnchor.current =
        popover.current
          ?.closest('.tri-training-month')
          ?.querySelector<HTMLButtonElement>(
            `[data-training-workout-open="${CSS.escape(selectedWorkoutId)}"]`,
          ) ?? null
    setPreview(null)
  }, [month, selectedWorkoutId, presentation.locale, presentation.distance])
  useLayoutEffect(() => () => clearTimeout(hideTimer.current), [])
  useLayoutEffect(() => {
    const panel = popover.current
    if (!preview || !panel) return
    let disposed = false
    panel.showPopover()
    const update = async () => {
      if (!preview.anchor.getClientRects().length) return setPreview(null)
      const { x, y } = await computePosition(preview.anchor, panel, {
        strategy: 'fixed',
        placement: 'bottom-start',
        middleware: [
          offset(6),
          flip({ padding: 8 }),
          shift({ padding: 8 }),
          size({
            padding: 8,
            apply: ({ availableHeight }) => {
              panel.style.maxHeight = `${Math.max(0, availableHeight)}px`
            },
          }),
        ],
      })
      if (disposed) return
      panel.style.left = `${x}px`
      panel.style.top = `${y}px`
    }
    const cleanup = autoUpdate(preview.anchor, panel, update)
    return () => {
      disposed = true
      cleanup()
      if (panel.matches(':popover-open')) panel.hidePopover()
    }
  }, [preview])

  return (
    <div
      class="tri-training-month"
      onKeyDown={event => {
        if (event.key !== 'Escape' || !preview) return
        event.preventDefault()
        event.stopPropagation()
        previewDismissed.current = true
        dismissedAnchor.current = preview.anchor
        dismiss()
        preview.anchor.focus({ preventScroll: true })
      }}
      onClickCapture={event => {
        if (!(event.target instanceof Element)) return
        const target = event.target.closest<HTMLButtonElement>('[data-training-workout-open]')
        if (!target) return
        previewDismissed.current = true
        dismissedAnchor.current = preview?.anchor ?? target
        dismiss()
      }}
    >
      <div class="tri-training-month-weekdays" aria-hidden="true">
        {dates.slice(0, 7).map(date => (
          <span key={date}>{calendarDateLabel(date, null, 'weekday', presentation.locale)}</span>
        ))}
      </div>
      <div
        id={id}
        class="tri-training-month-grid"
        style={{ gridTemplateRows: `repeat(${dates.length / 7}, minmax(5.5rem, 1fr))` }}
      >
        {dates.map((date, index) => {
          const sessions = workouts.filter(workout => workout.date === date)
          const covered = calendar !== null && trainingDayCovered(calendar, date)
          return (
            <section
              key={date}
              class="tri-training-month-day"
              data-date={date}
              data-covered={covered ? 'true' : 'false'}
              data-outside={date.slice(0, 7) !== month.slice(0, 7) ? 'true' : undefined}
              data-today={date === today ? 'true' : undefined}
              aria-labelledby={`${id}-day-${date}`}
            >
              <header class="tri-training-month-day-header">
                <h3 id={`${id}-day-${date}`}>
                  <time
                    dateTime={date}
                    aria-label={`${calendarWeekdayLabel(index % 7, presentation.locale)} ${calendarDateLabel(date, null, 'full', presentation.locale)}`}
                    aria-current={date === today ? 'date' : undefined}
                  >
                    {Number(date.slice(8))}
                  </time>
                </h3>
                <TrainingDayNotes
                  id={`${id}-notes-${date}`}
                  notes={calendar?.notes?.filter(note => note.date === date) ?? []}
                  formatter={formatter}
                />
              </header>
              <div class="tri-training-month-sessions">
                {sessions.map(workout => {
                  const metrics = workout.status === 'completed' ? workout.actual : workout.planned
                  return (
                    <button
                      key={workout.id}
                      type="button"
                      class="tri-training-session tri-training-month-workout"
                      data-training-workout-open={workout.id}
                      data-sport={workout.sport}
                      data-execution={trainingPeaksExecution(workout, today).status}
                      data-selected={selectedWorkoutId === workout.id ? 'true' : undefined}
                      aria-label={`${text('workout details')}: ${workout.title}`}
                      aria-controls={detailId}
                      aria-expanded={selectedWorkoutId === workout.id}
                      aria-describedby={
                        preview?.anchor.dataset.trainingWorkoutOpen === workout.id
                          ? `${id}-preview`
                          : undefined
                      }
                      onPointerEnter={event => {
                        if (event.pointerType === 'mouse' && !previewDismissed.current)
                          show(workout, event.currentTarget)
                      }}
                      onPointerMove={event => {
                        // Closing a preview can expose another workout under the stationary pointer.
                        if (event.pointerType !== 'mouse' || !previewDismissed.current) return
                        previewDismissed.current = false
                        show(workout, event.currentTarget)
                      }}
                      onPointerLeave={scheduleHide}
                      onFocus={event => {
                        if (
                          previewDismissed.current &&
                          dismissedAnchor.current === event.currentTarget
                        )
                          return
                        previewDismissed.current = false
                        show(workout, event.currentTarget)
                      }}
                      onBlur={event => {
                        if (
                          !(
                            event.relatedTarget instanceof Node &&
                            popover.current?.contains(event.relatedTarget)
                          )
                        )
                          dismiss()
                      }}
                    >
                      <span class="tri-training-month-mark" aria-hidden="true" />
                      <span class="tri-training-month-workout-title">{workout.title}</span>
                      <span class="tri-training-month-sport">{text(workout.sport)}</span>
                      {metrics.durationSeconds !== null && (
                        <span class="tri-training-month-duration">
                          {trainingDuration(metrics.durationSeconds)}
                        </span>
                      )}
                    </button>
                  )
                })}
                {sessions.length === 0 && (
                  <span
                    class="tri-training-month-empty"
                    title={text(covered ? 'No sessions.' : 'Not synced.')}
                    aria-label={text(covered ? 'No sessions.' : 'Not synced.')}
                  >
                    {covered ? '—' : '·'}
                  </span>
                )}
              </div>
            </section>
          )
        })}
      </div>
      <div
        ref={popover}
        id={`${id}-preview`}
        class="tri-calendar-pop tri-training-month-pop"
        popover="manual"
        role="region"
        aria-label={preview ? `${text('workout preview')}: ${preview.workout.title}` : undefined}
        onPointerEnter={cancelHide}
        onPointerLeave={scheduleHide}
        onFocusIn={cancelHide}
        onFocusOut={event => {
          if (
            !(
              event.relatedTarget instanceof Node &&
              (popover.current?.contains(event.relatedTarget) ||
                preview?.anchor.contains(event.relatedTarget))
            )
          )
            dismiss()
        }}
      >
        {preview && (
          <TrainingStructureHover id={`${id}-preview-structure`} resetKey={preview.workout.id}>
            <TrainingWorkoutCard
              workout={preview.workout}
              formatter={formatter}
              today={today}
              zones={calendar}
              detailId={detailId}
              selected={false}
            />
          </TrainingStructureHover>
        )}
      </div>
    </div>
  )
}
