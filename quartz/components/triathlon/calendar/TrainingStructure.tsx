import { useState } from 'preact/hooks'
import type {
  TrainingPeaksCalendarStep,
  TrainingPeaksCalendarWorkout,
} from '../../../util/trainingpeaks-calendar'
import type { TriathlonFormatter } from '../runtime/formatter'
import { clock, zoneClock } from '../../../util/triathlon-card'
import {
  TARGET_LABEL,
  absoluteText,
  rangeValues,
  stepLevel,
  stepTop,
  targetText,
  thresholdPercent,
  zoneRange,
  zoneScaleFor,
  zoneSourceText,
  type StructuredWorkout,
  type TrainingZoneSource,
  type ZoneScale,
} from './structure-targets'
import {
  parseSwimDescription,
  type SwimEffort,
  type SwimRest,
  type SwimStep,
  type SwimStructure,
} from './swim-structure'
import { trainingDuration } from './training-display'
import { useStructureHover } from './TrainingStructureHover'

export type { TrainingZoneSource } from './structure-targets'

const INTENSITY_LABEL = {
  warmUp: 'warm up',
  active: 'active',
  rest: 'rest',
  coolDown: 'cool down',
} as const

// Swim efforts come from the coach's words, so they take fixed steps of the zone fill ramp.
const EFFORT_BAR: Record<SwimEffort, { height: number; level: number }> = {
  easy: { height: 30, level: 1 },
  technique: { height: 42, level: 2 },
  steady: { height: 58, level: 4 },
  build: { height: 78, level: 5 },
  hard: { height: 100, level: 7 },
}

interface Bar {
  step: string
  lap: number
  name: string
  details: string[]
  grow: number
  height: number
  level: number | null
}

interface WorkoutStructureProps {
  workout: TrainingPeaksCalendarWorkout
  zones: TrainingZoneSource
  formatter: TriathlonFormatter
}

const stepNotes = (step: TrainingPeaksCalendarStep, sport: string): string =>
  [
    step.notes,
    step.cadence ? `${rangeValues(step.cadence).join('–')} ${sport === 'run' ? 'spm' : 'rpm'}` : '',
  ]
    .filter(Boolean)
    .join(' · ')

/** TrainingPeaks bars: width is step time, height is the upper target. */
const structureBars = (
  workout: StructuredWorkout,
  scale: ZoneScale | null,
  formatter: TriathlonFormatter,
): { blocks: Bar[][]; threshold: number } => {
  const tops = workout.structure.blocks.flatMap(block => block.steps.map(stepTop))
  const ceiling = Math.max(
    110,
    ...tops.map(top => (top === null ? 0 : thresholdPercent(workout, top))),
  )
  let lap = 0
  return {
    threshold: (100 / ceiling) * 100,
    blocks: workout.structure.blocks.map((block, blockIndex) =>
      Array.from({ length: block.repeat }, (_, repeat) =>
        block.steps.map((step, stepIndex) => {
          const top = stepTop(step)
          const zone = zoneRange(workout, scale, step.target)
          return {
            step: `${blockIndex}:${stepIndex}`,
            lap: ++lap,
            name: step.name || formatter.text(INTENSITY_LABEL[step.intensity]),
            details: [
              zoneClock(step.seconds),
              step.target
                ? workout.structure.metric === 'rpe'
                  ? targetText(workout, step.target, formatter)
                  : `${targetText(workout, step.target, formatter)} ${formatter.text(TARGET_LABEL[workout.structure.metric]).replace(/^% /, '')}`
                : '',
              step.target && scale ? absoluteText(workout, scale, step.target, formatter) : '',
              zone?.short ?? '',
              block.repeat > 1 ? `${repeat + 1}/${block.repeat} ×` : '',
              stepNotes(step, workout.sport),
            ].filter(Boolean),
            grow: step.seconds,
            height:
              top === null ? 12 : Math.max(4, (thresholdPercent(workout, top) / ceiling) * 100),
            level: stepLevel(workout, scale, top),
          }
        }),
      ).flat(),
    ),
  }
}

/** Swim bars: width is distance (or time for timed sets), height is the written effort. */
const swimBars = (
  swim: SwimStructure & { unit: 'meters' | 'seconds' },
  formatter: TriathlonFormatter,
): Bar[][] => {
  let lap = 0
  return swim.blocks.flatMap((block, blockIndex) =>
    block.kind === 'set'
      ? [
          Array.from({ length: block.repeat }, (_, repeat) =>
            block.steps.flatMap((step, stepIndex) =>
              Array.from({ length: step.reps }, (_, rep) =>
                step.parts.map(part => ({
                  step: `${blockIndex}:${stepIndex}`,
                  lap: ++lap,
                  name: step.label,
                  details: [
                    swim.unit === 'meters'
                      ? formatter.distance((part.meters ?? 0) / 1000, 'swim')
                      : zoneClock(part.seconds ?? 0),
                    part.effort ? formatter.text(part.effort) : '',
                    step.reps > 1 ? `${rep + 1}/${step.reps} ×` : '',
                    block.repeat > 1
                      ? `${formatter.text('set')} ${repeat + 1}/${block.repeat}`
                      : '',
                    step.rest
                      ? `${formatter.text(step.rest.interval ? 'send-off interval' : 'rest')}: ${restText(step.rest)}`
                      : '',
                    step.notes.join(' · '),
                  ].filter(Boolean),
                  grow: part[swim.unit] ?? 0,
                  height: part.effort ? EFFORT_BAR[part.effort].height : 45,
                  level: part.effort ? EFFORT_BAR[part.effort].level : null,
                })),
              ).flat(),
            ),
          ).flat(),
        ]
      : [],
  )
}

const StructureChart = ({
  blocks,
  threshold,
  label,
  compact,
  workout,
  formatter,
}: {
  blocks: Bar[][]
  threshold: number | null
  label: string
  compact: boolean
  workout: TrainingPeaksCalendarWorkout
  formatter: TriathlonFormatter
}) => {
  const hover = useStructureHover()
  const [focusLap, setFocusLap] = useState(1)
  const total = blocks.reduce((sum, bars) => sum + bars.length, 0)
  const active = hover?.active?.workoutId === workout.id ? hover.active : null
  return (
    <div
      class={`tri-training-structure-chart${compact ? ' tri-training-structure-chart--compact' : ''}`}
      data-threshold={threshold === null ? undefined : ''}
      data-active={active ? '' : undefined}
      style={threshold === null ? undefined : { '--structure-threshold': `${threshold}%` }}
      role="group"
      aria-label={label || `${formatter.text('workout structure')}: ${workout.title}`}
    >
      {blocks.map((bars, blockIndex) => (
        <span
          key={blockIndex}
          class="tri-training-structure-block"
          style={{ flexGrow: bars.reduce((sum, bar) => sum + bar.grow, 0) }}
        >
          {bars.map(bar => {
            const title = `${formatter.text('step')} ${bar.lap}/${total} · ${bar.name}`
            const show = (anchor: HTMLElement) =>
              hover?.show({ ...bar, title, workoutId: workout.id, anchor })
            return (
              <button
                key={bar.lap}
                type="button"
                class="tri-training-structure-bar"
                style={{ flexGrow: bar.grow }}
                data-training-workout-open={compact ? workout.id : undefined}
                data-lap={bar.lap}
                data-step={bar.step}
                data-active={active?.lap === bar.lap ? '' : undefined}
                tabIndex={focusLap === bar.lap ? 0 : -1}
                aria-label={`${title}: ${bar.details.join(' · ')}`}
                aria-describedby={active?.lap === bar.lap ? hover?.tooltipId : undefined}
                onPointerEnter={event => show(event.currentTarget)}
                onPointerLeave={event => {
                  if (!event.currentTarget.matches(':focus-visible'))
                    hover?.clear(event.currentTarget)
                }}
                onFocus={event => {
                  setFocusLap(bar.lap)
                  show(event.currentTarget)
                }}
                onBlur={event => hover?.clear(event.currentTarget)}
                onClick={event => {
                  if (!compact) event.stopPropagation()
                  show(event.currentTarget)
                }}
                onKeyDown={event => {
                  if (event.key === 'Escape' && active) {
                    event.preventDefault()
                    event.stopPropagation()
                    hover?.clear(event.currentTarget)
                    return
                  }
                  const target =
                    event.key === 'Home'
                      ? 1
                      : event.key === 'End'
                        ? total
                        : event.key === 'ArrowRight'
                          ? Math.min(total, bar.lap + 1)
                          : event.key === 'ArrowLeft'
                            ? Math.max(1, bar.lap - 1)
                            : null
                  if (target === null) return
                  event.preventDefault()
                  event.stopPropagation()
                  event.currentTarget
                    .closest('.tri-training-structure-chart')
                    ?.querySelector<HTMLButtonElement>(`[data-lap="${target}"]`)
                    ?.focus()
                }}
              >
                <span
                  aria-hidden="true"
                  class={
                    bar.level === null
                      ? 'tri-zone-fill'
                      : `tri-zone-fill tri-zone-fill--${bar.level}`
                  }
                  style={{ height: `${bar.height}%` }}
                />
              </button>
            )
          })}
        </span>
      ))}
    </div>
  )
}

const structured = (workout: TrainingPeaksCalendarWorkout): StructuredWorkout | null =>
  workout.structure ? { ...workout, structure: workout.structure } : null

// TrainingPeaks keeps no structure for swims, so the coach's set list is read from the description.
const swimStructure = (workout: TrainingPeaksCalendarWorkout): SwimStructure | null =>
  !workout.structure && workout.sport === 'swim' ? parseSwimDescription(workout.description) : null

const chartable = (
  swim: SwimStructure | null,
): swim is SwimStructure & { unit: 'meters' | 'seconds' } => swim?.unit != null

export const TrainingStructureStrip = ({ workout, zones, formatter }: WorkoutStructureProps) => {
  const tp = structured(workout)
  if (tp)
    return (
      <StructureChart
        {...structureBars(tp, zoneScaleFor(tp, zones, formatter.text), formatter)}
        workout={workout}
        formatter={formatter}
        label=""
        compact
      />
    )
  const swim = swimStructure(workout)
  return chartable(swim) ? (
    <StructureChart
      blocks={swimBars(swim, formatter)}
      threshold={null}
      label=""
      compact
      workout={workout}
      formatter={formatter}
    />
  ) : null
}

const TrainingPeaksStructure = ({
  workout,
  zones,
  formatter,
}: Omit<WorkoutStructureProps, 'workout'> & { workout: StructuredWorkout }) => {
  const { text } = formatter
  const scale = zoneScaleFor(workout, zones, text)
  const hover = useStructureHover()
  const active = hover?.active?.workoutId === workout.id ? hover.active : null
  const total = workout.structure.blocks.reduce(
    (sum, block) =>
      sum + block.repeat * block.steps.reduce((steps, step) => steps + step.seconds, 0),
    0,
  )
  return (
    <section class="tri-training-structure">
      <h4 class="tri-training-structure-title">{text('structure')}</h4>
      <StructureChart
        {...structureBars(workout, scale, formatter)}
        workout={workout}
        formatter={formatter}
        label={`${text('workout structure')}: ${trainingDuration(total)}`}
        compact={false}
      />
      <table class="tri-training-table tri-training-structure-table">
        <thead>
          <tr>
            <th scope="col">{text('step')}</th>
            <th scope="col">{text('time')}</th>
            <th scope="col">{text(TARGET_LABEL[workout.structure.metric])}</th>
            <th scope="col">{text('zone')}</th>
          </tr>
        </thead>
        {workout.structure.blocks.map((block, blockIndex) => {
          const grouped = block.repeat > 1 || block.kind === 'rampUp' || block.kind === 'rampDown'
          const blockSeconds = block.steps.reduce((sum, step) => sum + step.seconds, 0)
          return (
            <tbody key={blockIndex} data-grouped={grouped ? 'true' : undefined}>
              {grouped && (
                <tr class="tri-training-structure-group">
                  <th scope="rowgroup">
                    {block.kind === 'rampUp'
                      ? text('ramp up')
                      : block.kind === 'rampDown'
                        ? text('ramp down')
                        : `${block.repeat} ×`}
                  </th>
                  <td>{zoneClock(blockSeconds * block.repeat)}</td>
                  <td colSpan={2} />
                </tr>
              )}
              {block.steps.map((step, stepIndex) => {
                const zone = zoneRange(workout, scale, step.target)
                const note = stepNotes(step, workout.sport)
                return (
                  <tr
                    key={stepIndex}
                    data-step={`${blockIndex}:${stepIndex}`}
                    data-active={active?.step === `${blockIndex}:${stepIndex}` ? '' : undefined}
                  >
                    <th scope="row">
                      {step.name || text(INTENSITY_LABEL[step.intensity])}
                      {note && <span class="tri-training-structure-note">{note}</span>}
                    </th>
                    <td>{zoneClock(step.seconds)}</td>
                    <td>
                      {step.target ? targetText(workout, step.target, formatter) : '—'}
                      {step.target && scale && (
                        <span class="tri-training-structure-note">
                          {absoluteText(workout, scale, step.target, formatter)}
                        </span>
                      )}
                    </td>
                    <td title={zone?.label}>{zone?.short ?? '—'}</td>
                  </tr>
                )
              })}
            </tbody>
          )
        })}
      </table>
      <p class="tri-training-structure-source">{zoneSourceText(workout, scale, formatter)}</p>
    </section>
  )
}

const restText = (rest: SwimRest | null): string =>
  !rest
    ? '—'
    : rest.interval
      ? `@ ${clock(rest.min)}`
      : rest.max === null
        ? zoneClock(rest.min)
        : `${zoneClock(rest.min)}–${zoneClock(rest.max)}`

const SwimStructureView = ({
  workout,
  swim,
  formatter,
}: Omit<WorkoutStructureProps, 'zones'> & { swim: SwimStructure }) => {
  const { text } = formatter
  const hover = useStructureHover()
  const active = hover?.active?.workoutId === workout.id ? hover.active : null
  const amount = (meters: number, seconds: number): string =>
    meters > 0 ? formatter.distance(meters / 1000, 'swim') : zoneClock(seconds)
  const roundAmount = (steps: readonly SwimStep[], repeat: number): string =>
    amount(
      repeat *
        steps.reduce(
          (sum, step) =>
            sum + step.reps * step.parts.reduce((parts, part) => parts + (part.meters ?? 0), 0),
          0,
        ),
      repeat *
        steps.reduce(
          (sum, step) =>
            sum + step.reps * step.parts.reduce((parts, part) => parts + (part.seconds ?? 0), 0),
          0,
        ),
    )
  const total = swim.unit === 'seconds' ? swim.seconds : swim.meters
  const planned =
    swim.unit === 'seconds' ? workout.planned.durationSeconds : workout.planned.distanceMeters
  const totalText = swim.unit === 'seconds' ? zoneClock(total) : amount(swim.meters, swim.seconds)
  const plannedText =
    planned === null || planned === total
      ? null
      : swim.unit === 'seconds'
        ? zoneClock(planned)
        : formatter.distance(planned / 1000, 'swim')
  return (
    <section class="tri-training-structure">
      <h4 class="tri-training-structure-title">{text('structure')}</h4>
      {chartable(swim) && (
        <StructureChart
          blocks={swimBars(swim, formatter)}
          workout={workout}
          formatter={formatter}
          threshold={null}
          label={`${text('workout structure')}: ${totalText}`}
          compact={false}
        />
      )}
      <table class="tri-training-table tri-training-structure-table tri-training-structure-table--swim">
        <thead>
          <tr>
            <th scope="col">{text('set')}</th>
            <th scope="col">{text('rest')}</th>
            <th scope="col">{text('effort')}</th>
          </tr>
        </thead>
        {swim.blocks.map((block, blockIndex) =>
          block.kind === 'rest' ? (
            <tbody key={blockIndex}>
              <tr>
                <th scope="row">{text('rest')}</th>
                <td>{restText(block.rest)}</td>
                <td>—</td>
              </tr>
            </tbody>
          ) : (
            <tbody key={blockIndex} data-grouped={block.repeat > 1 ? 'true' : undefined}>
              {block.repeat > 1 && (
                <tr class="tri-training-structure-group">
                  <th scope="rowgroup">
                    {block.repeat} ×
                    <span class="tri-training-structure-note">
                      {roundAmount(block.steps, block.repeat)}
                    </span>
                  </th>
                  <td title={block.rest ? text('rest between rounds') : undefined}>
                    {block.rest && `+${restText(block.rest)}`}
                  </td>
                  <td />
                </tr>
              )}
              {block.steps.map((step, stepIndex) => {
                const efforts = step.parts.map(part => (part.effort ? text(part.effort) : '—'))
                const effort = new Set(efforts).size === 1 ? efforts[0] : efforts.join(' / ')
                // A bare effort word ("easy", "HARD") already shows in the effort column.
                const notes =
                  step.notes[0]?.toLowerCase() === step.parts[0].effort
                    ? step.notes.slice(1)
                    : step.notes
                return (
                  <tr
                    key={stepIndex}
                    data-step={`${blockIndex}:${stepIndex}`}
                    data-active={active?.step === `${blockIndex}:${stepIndex}` ? '' : undefined}
                  >
                    <th scope="row">
                      {step.label}
                      {notes.length > 0 && (
                        <span class="tri-training-structure-note">{notes.join(' · ')}</span>
                      )}
                    </th>
                    <td title={step.rest?.interval ? text('send-off interval') : undefined}>
                      {restText(step.rest)}
                    </td>
                    <td>{effort}</td>
                  </tr>
                )
              })}
            </tbody>
          ),
        )}
      </table>
      <p class="tri-training-structure-source">
        {plannedText ? `${totalText} · ${plannedText} ${text('planned')}` : totalText}
      </p>
    </section>
  )
}

export const TrainingStructure = ({ workout, zones, formatter }: WorkoutStructureProps) => {
  const tp = structured(workout)
  if (tp)
    return (
      <TrainingPeaksStructure key={workout.id} workout={tp} zones={zones} formatter={formatter} />
    )
  const swim = swimStructure(workout)
  return swim ? (
    <SwimStructureView key={workout.id} workout={workout} swim={swim} formatter={formatter} />
  ) : null
}
