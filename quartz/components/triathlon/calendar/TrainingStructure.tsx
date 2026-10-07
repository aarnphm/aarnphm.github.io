import { useState } from 'preact/hooks'
import type {
  TrainingPeaksCalendar,
  TrainingPeaksCalendarRange,
  TrainingPeaksCalendarStep,
  TrainingPeaksCalendarStructure,
  TrainingPeaksCalendarWorkout,
  TrainingPeaksCalendarZone,
  TrainingPeaksCalendarZoneBounds,
} from '../../../util/trainingpeaks-calendar'
import type { TriathlonFormatter } from '../runtime/formatter'
import { RUN_PACE_ZONE_NAMES } from '../../../util/run-pace-zones'
import {
  HR_ZONE_NAMES,
  KM_TO_MI,
  POWER_ZONE_NAMES,
  clock,
  zoneClock,
} from '../../../util/triathlon-card'
import {
  parseSwimDescription,
  type SwimEffort,
  type SwimRest,
  type SwimStep,
  type SwimStructure,
} from './swim-structure'
import { trainingDuration } from './training-display'
import { useStructureHover } from './TrainingStructureHover'

const ZONE_SET = {
  percentOfFtp: 'power',
  percentOfThresholdPace: 'speed',
  percentOfThresholdHr: 'heartRate',
} as const

const TARGET_LABEL = {
  percentOfFtp: '% FTP',
  percentOfThresholdPace: '% threshold pace',
  percentOfThresholdHr: '% threshold HR',
} as const

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

export type TrainingZoneSource = Pick<TrainingPeaksCalendar, 'zones' | 'localZones'> | null

interface ZoneScale {
  source: 'trainingpeaks' | 'garden'
  threshold: number
  zones: TrainingPeaksCalendarZone[]
}

interface ZoneMatch {
  label: string
  short: string
  level: number
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

type StructuredWorkout = TrainingPeaksCalendarWorkout & {
  structure: TrainingPeaksCalendarStructure
}

interface WorkoutStructureProps {
  workout: TrainingPeaksCalendarWorkout
  zones: TrainingZoneSource
  formatter: TriathlonFormatter
}

// The garden FTP and pace zones each belong to one sport; its heart rate zones cover both.
const gardenZones = (
  workout: StructuredWorkout,
  source: TrainingZoneSource,
): [TrainingPeaksCalendarZoneBounds, readonly string[]] | null => {
  const local = source?.localZones
  const { metric } = workout.structure
  const sport = workout.sport
  const found =
    metric === 'percentOfThresholdHr' && (sport === 'bike' || sport === 'run')
      ? ([local?.heartRate, HR_ZONE_NAMES] as const)
      : metric === 'percentOfFtp' && sport === 'bike'
        ? ([local?.power, POWER_ZONE_NAMES] as const)
        : metric === 'percentOfThresholdPace' && sport === 'run'
          ? ([local?.runSpeed, RUN_PACE_ZONE_NAMES] as const)
          : null
  return found?.[0] ? [found[0], found[1]] : null
}

// As in TrainingPeaks, the athlete default (workoutTypeId 0) covers sports without their own set.
const zoneScaleFor = (
  workout: StructuredWorkout,
  source: TrainingZoneSource,
  text: TriathlonFormatter['text'],
): ZoneScale | null => {
  const sets = source?.zones?.[ZONE_SET[workout.structure.metric]] ?? []
  const set =
    sets.find(item => item.workoutTypeId === workout.workoutTypeId) ??
    sets.find(item => item.workoutTypeId === 0)
  if (set) return { source: 'trainingpeaks', threshold: set.threshold, zones: set.zones }
  const garden = gardenZones(workout, source)
  if (!garden) return null
  const [{ threshold, bounds }, names] = garden
  return {
    source: 'garden',
    threshold,
    zones: [...bounds, Infinity].map((max, index) => ({
      label: text(names[index] ?? `Z${index + 1}`),
      min: index === 0 ? 0 : bounds[index - 1],
      max,
    })),
  }
}

const zoneAt = (scale: ZoneScale, percent: number): ZoneMatch => {
  const value = (percent / 100) * scale.threshold
  const found = scale.zones.findIndex(zone => value <= zone.max)
  const index = found < 0 ? scale.zones.length - 1 : found
  const label = scale.zones[index].label
  const id =
    scale.source === 'trainingpeaks'
      ? /^(?:zone|z)\s*(\d+[a-z]?|[a-z])\b/i.exec(label)?.[1]
      : String(index + 1)
  return {
    label,
    short: id ? `Z${id.toUpperCase()}` : label,
    // Any zone count spreads over the seven steps of the shared zone fill ramp.
    level: 1 + Math.round((index * 6) / Math.max(1, scale.zones.length - 1)),
  }
}

const rangeValues = (range: TrainingPeaksCalendarRange): number[] =>
  range.max === null || range.max === range.min ? [range.min] : [range.min, range.max]

const stepTop = (step: TrainingPeaksCalendarStep): number | null =>
  step.target?.max ?? step.target?.min ?? null

const percentText = (range: TrainingPeaksCalendarRange, formatter: TriathlonFormatter): string =>
  `${rangeValues(range)
    .map(value => formatter.number(value, 0, 1))
    .join('–')}%`

// TrainingPeaks scales threshold speed, so a higher percentage is a faster pace.
const absoluteText = (
  workout: StructuredWorkout,
  threshold: number,
  range: TrainingPeaksCalendarRange,
  formatter: TriathlonFormatter,
): string => {
  const values = rangeValues(range).map(value => (value / 100) * threshold)
  const metric = workout.structure.metric
  if (metric === 'percentOfFtp') return `${values.map(value => Math.round(value)).join('–')} W`
  if (metric === 'percentOfThresholdHr')
    return `${values.map(value => Math.round(value)).join('–')} bpm`
  if (workout.sport === 'swim') return `${values.map(speed => clock(100 / speed)).join('–')} /100m`
  const imperial = formatter.presentation.distance === 'imperial'
  return `${values.map(speed => clock(1000 / speed / (imperial ? KM_TO_MI : 1))).join('–')} ${imperial ? '/mi' : '/km'}`
}

const zoneRange = (
  scale: ZoneScale | null,
  range: TrainingPeaksCalendarRange | null,
): { short: string; label: string } | null => {
  if (!scale || !range) return null
  const low = zoneAt(scale, range.min)
  const high = zoneAt(scale, range.max ?? range.min)
  return low.label === high.label
    ? low
    : { short: `${low.short}–${high.short}`, label: `${low.label} – ${high.label}` }
}

const thresholdText = (
  workout: StructuredWorkout,
  scale: ZoneScale,
  formatter: TriathlonFormatter,
): string => {
  const metric = workout.structure.metric
  if (metric === 'percentOfFtp') return `FTP ${Math.round(scale.threshold)} W`
  if (metric === 'percentOfThresholdHr')
    return `${formatter.text('threshold HR')} ${Math.round(scale.threshold)} bpm`
  return `${formatter.text('threshold pace')} ${absoluteText(workout, scale.threshold, { min: 100, max: null }, formatter)}`
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
  const ceiling = Math.max(110, ...tops.map(top => top ?? 0))
  let lap = 0
  return {
    threshold: (100 / ceiling) * 100,
    blocks: workout.structure.blocks.map((block, blockIndex) =>
      Array.from({ length: block.repeat }, (_, repeat) =>
        block.steps.map((step, stepIndex) => {
          const top = stepTop(step)
          const zone = zoneRange(scale, step.target)
          return {
            step: `${blockIndex}:${stepIndex}`,
            lap: ++lap,
            name: step.name || formatter.text(INTENSITY_LABEL[step.intensity]),
            details: [
              zoneClock(step.seconds),
              step.target
                ? `${percentText(step.target, formatter)} ${formatter.text(TARGET_LABEL[workout.structure.metric]).replace(/^% /, '')}`
                : '',
              step.target && scale
                ? absoluteText(workout, scale.threshold, step.target, formatter)
                : '',
              zone?.short ?? '',
              block.repeat > 1 ? `${repeat + 1}/${block.repeat} ×` : '',
              stepNotes(step, workout.sport),
            ].filter(Boolean),
            grow: step.seconds,
            height: top === null ? 12 : Math.max(4, (top / ceiling) * 100),
            level: scale && top !== null ? zoneAt(scale, top).level : null,
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
                const zone = zoneRange(scale, step.target)
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
                      {step.target ? percentText(step.target, formatter) : '—'}
                      {step.target && scale && (
                        <span class="tri-training-structure-note">
                          {absoluteText(workout, scale.threshold, step.target, formatter)}
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
      <p class="tri-training-structure-source">
        {scale
          ? `${text(scale.source === 'garden' ? 'Garden zones' : 'TrainingPeaks zones')} · ${thresholdText(workout, scale, formatter)}`
          : text('No zones for this sport.')}
      </p>
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
