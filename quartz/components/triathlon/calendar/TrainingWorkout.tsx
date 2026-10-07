import { useState } from 'preact/hooks'
import type { TrainingPeaksCalendarWorkout } from '../../../util/trainingpeaks-calendar'
import type { TriathlonFormatter } from '../runtime/formatter'
import { trainingPeaksExecution } from '../../../util/trainingpeaks-execution'
import { triathlonActivityHref } from '../../../util/triathlon-date-route'
import { calendarDateLabel } from './display'
import { trainingDuration } from './training-display'
import { TrainingNotesButton, TrainingNotesTip } from './TrainingDayNotes'
import {
  TrainingStructure,
  TrainingStructureStrip,
  type TrainingZoneSource,
} from './TrainingStructure'

const EXECUTION_LABEL: Record<ReturnType<typeof trainingPeaksExecution>['status'], string> = {
  'on-target': 'on target',
  'off-target': 'off target',
  'far-off-target': 'far off target',
  missed: 'missed',
  planned: 'planned',
  unplanned: 'unplanned',
  unavailable: 'comparison unavailable',
}

interface TrainingWorkoutProps {
  workout: TrainingPeaksCalendarWorkout
  formatter: TriathlonFormatter
  today: string
  zones: TrainingZoneSource
}

export const TrainingWorkoutCard = ({
  workout,
  formatter,
  today,
  zones,
  detailId,
  selected,
}: TrainingWorkoutProps & { detailId: string; selected: boolean }) => {
  const { text, number } = formatter
  const [notesOpen, setNotesOpen] = useState(false)
  const note = workout.preActivityNote
  const notesId = `${detailId}-notes-${workout.id}`
  const metrics = workout.status === 'completed' ? workout.actual : workout.planned
  const execution = trainingPeaksExecution(workout, today)
  const activity = workout.activity
  const activityHref = activity ? triathlonActivityHref(activity.date, activity.id) : null
  const distance =
    metrics.distanceMeters === null
      ? null
      : formatter.distance(metrics.distanceMeters / 1000, workout.sport === 'swim' ? 'swim' : 'run')
  return (
    <article
      class="tri-training-session"
      data-training-workout={workout.id}
      data-sport={workout.sport}
      data-status={workout.status}
      data-execution={execution.status}
      data-execution-metric={execution.metric ?? undefined}
      data-execution-ratio={execution.ratio ?? undefined}
      data-selected={selected ? 'true' : undefined}
      onPointerEnter={note ? () => setNotesOpen(true) : undefined}
      onPointerLeave={note ? () => setNotesOpen(false) : undefined}
    >
      <div class="tri-training-workout-summary">
        <span class="tri-training-workout-kind">
          <span class="tri-training-sport" title={text(workout.sport)}>
            {text(workout.sport)}
          </span>
          {note && (
            <TrainingNotesButton
              id={notesId}
              label={`${text('pre-activity')}: ${workout.title}`}
              open={notesOpen}
              setOpen={setNotesOpen}
            />
          )}
          <span class="tri-training-status" title={text(EXECUTION_LABEL[execution.status])}>
            {text(workout.status)}
          </span>
        </span>
        <strong class="tri-training-workout-title" title={workout.title}>
          {workout.title}
        </strong>
        <span class="tri-training-workout-metrics">
          <span class="tri-training-workout-volume">
            {metrics.durationSeconds !== null && (
              <span title={trainingDuration(metrics.durationSeconds)}>
                {trainingDuration(metrics.durationSeconds)}
              </span>
            )}
            {distance !== null && <span title={distance}>{distance}</span>}
          </span>
          <span>{metrics.tss !== null && `${number(metrics.tss, 0)} TSS`}</span>
        </span>
        <TrainingStructureStrip workout={workout} zones={zones} formatter={formatter} />
        <span class="tri-training-workout-bottom">
          {activityHref && activity && (
            <a
              class="tri-training-activity-link internal"
              href={activityHref}
              aria-label={`${text('view activity')}: ${activity.title}`}
              data-training-activity
              data-no-popover
            >
              {text('activity')} →
            </a>
          )}
          <button
            type="button"
            class="tri-training-workout-open"
            data-training-workout-open={workout.id}
            aria-label={`${text('workout details')}: ${workout.title}`}
            aria-controls={detailId}
            aria-expanded={selected}
          >
            <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
              <path d="m6 4 4 4-4 4" />
            </svg>
          </button>
        </span>
      </div>
      {note && (
        <TrainingNotesTip
          id={notesId}
          open={notesOpen}
          notes={[{ key: 'pre-activity', title: text('pre-activity'), body: note }]}
        />
      )}
    </article>
  )
}

export const TrainingWorkoutDetail = ({
  workout,
  formatter,
  today,
  zones,
  sessions,
}: TrainingWorkoutProps & { sessions: TrainingPeaksCalendarWorkout[] }) => {
  const { text, number } = formatter
  const completed = workout.status === 'completed'
  const execution = trainingPeaksExecution(workout, today)
  const basis =
    execution.metric === 'durationSeconds'
      ? 'duration'
      : execution.metric === 'distanceMeters'
        ? 'distance'
        : 'TSS'
  const distance = (meters: number | null): string =>
    meters === null
      ? '—'
      : formatter.distance(meters / 1000, workout.sport === 'swim' ? 'swim' : 'run')
  const tss = (value: number | null): string => (value === null ? '—' : number(value, 0))
  const time = (completed ? workout.startTime : workout.startTimePlanned)?.slice(0, 5)
  const activity = workout.activity
  const activityHref = activity ? triathlonActivityHref(activity.date, activity.id) : null
  const others = sessions.filter(session => session.id !== workout.id)
  return (
    <article
      class="tri-calendar-card tri-training-workout-detail"
      data-execution={execution.status}
    >
      <header class="tri-pop-head tri-pop-head--detail">
        <div class="tri-pop-head-row">
          <time class="tri-pop-date" dateTime={workout.date}>
            {calendarDateLabel(workout.date, null, 'full', formatter.presentation.locale)}
          </time>
          <div class="tri-pop-head-actions">
            <button
              type="button"
              class="tri-ana-back tri-ana-back--ico"
              data-training-detail-close
              data-site-cursor-action
              aria-label={text('go back')}
            >
              <svg viewBox="0 0 24 24" aria-hidden="true" data-site-cursor-icon>
                <path d="M19 12H5M11 6l-6 6 6 6" />
              </svg>
            </button>
          </div>
        </div>
        <h3 class="tri-pop-title">{workout.title}</h3>
      </header>
      <p class="tri-calendar-card-meta">
        <span>{text(workout.sport)}</span>
        <span>{text(workout.status)}</span>
        {time && time !== '00:00' && <time dateTime={time}>{time}</time>}
      </p>
      {workout.description && <p class="tri-training-instructions">{workout.description}</p>}
      {workout.preActivityNote && (
        <p class="tri-training-instructions tri-training-pre-activity">
          <span class="tri-training-pre-activity-label">{text('pre-activity')}</span>
          {workout.preActivityNote}
        </p>
      )}
      <table class="tri-training-table">
        <thead>
          <tr>
            <td />
            <th scope="col">{text('planned')}</th>
            <th scope="col">{text('completed')}</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th scope="row">{text('duration')}</th>
            <td>{trainingDuration(workout.planned.durationSeconds)}</td>
            <td>{trainingDuration(workout.actual.durationSeconds)}</td>
          </tr>
          <tr>
            <th scope="row">{text('distance')}</th>
            <td>{distance(workout.planned.distanceMeters)}</td>
            <td>{distance(workout.actual.distanceMeters)}</td>
          </tr>
          <tr>
            <th scope="row">TSS</th>
            <td>{tss(workout.planned.tss)}</td>
            <td>{tss(workout.actual.tss)}</td>
          </tr>
        </tbody>
      </table>
      {execution.status !== 'planned' && (
        <p class="tri-training-execution">
          <span>{text(EXECUTION_LABEL[execution.status])}</span>
          {execution.ratio !== null && execution.metric !== null && (
            <span
              class="tri-training-execution-ratio"
              title={`${execution.ratio * 100}% ${text('of planned')} ${text(basis)}`}
            >
              {number(execution.ratio * 100, 0, 1)}% {text('of planned')} {text(basis)}
            </span>
          )}
        </p>
      )}
      {completed && !activity && (
        <p class="tri-training-instructions">{text('No matching local activity.')}</p>
      )}
      <TrainingStructure workout={workout} zones={zones} formatter={formatter} />
      {activityHref && activity && (
        <a
          class="tri-training-activity-link internal"
          href={activityHref}
          aria-label={`${text('view activity')}: ${activity.title}`}
          data-training-activity
          data-no-popover
        >
          {text('activity')} →
        </a>
      )}
      {others.length > 0 && (
        <div class="tri-calendar-card-also">
          <span>{text('also on this day')}</span>
          {others.map(other => (
            <button key={other.id} type="button" data-training-workout-open={other.id}>
              {other.title}
            </button>
          ))}
        </div>
      )}
    </article>
  )
}
