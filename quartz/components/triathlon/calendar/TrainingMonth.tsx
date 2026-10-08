import type {
  TrainingPeaksCalendar,
  TrainingPeaksCalendarWorkout,
} from '../../../util/trainingpeaks-calendar'
import type { TriathlonFormatter } from '../runtime/formatter'
import { trainingPeaksExecution } from '../../../util/trainingpeaks-execution'
import { MonthGrid } from '../../controls/month-grid'
import { useAnchoredPreview } from '../../controls/use-anchored-preview'
import { calendarDateLabel, calendarWeekdayLabel } from './display'
import { trainingDayCovered, trainingDuration } from './training-display'
import { TrainingDayNotes } from './TrainingDayNotes'
import { TrainingStructureHover } from './TrainingStructureHover'
import { TrainingWorkoutCard } from './TrainingWorkout'

const MONTH_CLASSES = {
  root: 'tri-training-month',
  weekdays: 'tri-training-month-weekdays',
  grid: 'tri-training-month-grid',
  day: 'tri-training-month-day',
  dayHeader: 'tri-training-month-day-header',
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
  const { preview, popoverRef, anchorProps, popoverProps, onKeyDown, dismissFrom } =
    useAnchoredPreview<TrainingPeaksCalendarWorkout>({
      canShow: workout => workout.id !== selectedWorkoutId,
      resetKeys: [month, selectedWorkoutId, presentation.locale, presentation.distance],
      resetAnchor: popover =>
        selectedWorkoutId
          ? (popover
              ?.closest('.tri-training-month')
              ?.querySelector<HTMLButtonElement>(
                `[data-training-workout-open="${CSS.escape(selectedWorkoutId)}"]`,
              ) ?? null)
          : undefined,
    })

  return (
    <MonthGrid
      id={id}
      dates={dates}
      month={month}
      today={today}
      classes={MONTH_CLASSES}
      weekdayLabel={date => calendarDateLabel(date, null, 'weekday', presentation.locale)}
      dayLabel={(date, index) =>
        `${calendarWeekdayLabel(index % 7, presentation.locale)} ${calendarDateLabel(date, null, 'full', presentation.locale)}`
      }
      dayAttributes={date => ({
        'data-covered': calendar !== null && trainingDayCovered(calendar, date) ? 'true' : 'false',
      })}
      dayHeader={date => (
        <TrainingDayNotes
          id={`${id}-notes-${date}`}
          notes={calendar?.notes?.filter(note => note.date === date) ?? []}
          formatter={formatter}
        />
      )}
      onKeyDown={onKeyDown}
      onClickCapture={event => {
        if (!(event.target instanceof Element)) return
        const target = event.target.closest<HTMLButtonElement>('[data-training-workout-open]')
        if (target) dismissFrom(target)
      }}
      renderDay={date => {
        const sessions = workouts.filter(workout => workout.date === date)
        const covered = calendar !== null && trainingDayCovered(calendar, date)
        return (
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
                  {...anchorProps(workout)}
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
        )
      }}
    >
      <div
        ref={popoverRef}
        id={`${id}-preview`}
        class="tri-calendar-pop tri-training-month-pop"
        popover="manual"
        role="region"
        aria-label={preview ? `${text('workout preview')}: ${preview.item.title}` : undefined}
        {...popoverProps}
      >
        {preview && (
          <TrainingStructureHover id={`${id}-preview-structure`} resetKey={preview.item.id}>
            <TrainingWorkoutCard
              workout={preview.item}
              formatter={formatter}
              today={today}
              zones={calendar}
              detailId={detailId}
              selected={false}
            />
          </TrainingStructureHover>
        )}
      </div>
    </MonthGrid>
  )
}
