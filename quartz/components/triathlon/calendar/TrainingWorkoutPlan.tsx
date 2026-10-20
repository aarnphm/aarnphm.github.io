import type { TrainingPeaksCalendarWorkout } from '../../../util/trainingpeaks-calendar'
import type { TriathlonFormatter } from '../runtime/formatter'
import { trainingPeaksExecution } from '../../../util/trainingpeaks-execution'
import { effortCueSegments, effortCues } from './effort-cues'
import { trainingDuration } from './training-display'

export const EXECUTION_LABEL: Record<ReturnType<typeof trainingPeaksExecution>['status'], string> =
  {
    'on-target': 'on target',
    'off-target': 'off target',
    'far-off-target': 'far off target',
    missed: 'missed',
    planned: 'planned',
    unplanned: 'unplanned',
    unavailable: 'comparison unavailable',
  }

/** Planned against completed duration, distance and TSS, with the execution status. */
export const TrainingWorkoutTotals = ({
  workout,
  formatter,
  today,
}: {
  workout: TrainingPeaksCalendarWorkout
  formatter: TriathlonFormatter
  today: string
}) => {
  const { text, number } = formatter
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
  return (
    <>
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
    </>
  )
}

const CuedText = ({ value }: { value: string }) => (
  <>
    {effortCueSegments(value).map((segment, index) =>
      segment.cue ? (
        <mark key={index} class="tri-training-cue">
          {segment.text}
        </mark>
      ) : (
        segment.text
      ),
    )}
  </>
)

/** The coach's description and pre-activity note, with the effort cues written in them. */
export const TrainingInstructions = ({
  workout,
  formatter,
}: {
  workout: TrainingPeaksCalendarWorkout
  formatter: TriathlonFormatter
}) => {
  const { text } = formatter
  const note = workout.preActivityNote ?? ''
  const cues = effortCues(workout.description, note)
  return (
    <>
      {cues.length > 0 && (
        <p class="tri-training-cues">
          <span class="tri-training-pre-activity-label">{text('effort')}</span>
          {cues.map(cue => (
            <span key={cue} class="tri-training-cue">
              {cue}
            </span>
          ))}
        </p>
      )}
      {workout.description && (
        <p class="tri-training-instructions">
          <CuedText value={workout.description} />
        </p>
      )}
      {note && (
        <p class="tri-training-instructions tri-training-pre-activity">
          <span class="tri-training-pre-activity-label">{text('pre-activity')}</span>
          <CuedText value={note} />
        </p>
      )}
    </>
  )
}
