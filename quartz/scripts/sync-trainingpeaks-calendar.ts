import { fromHtml } from 'hast-util-from-html'
import { toText } from 'hast-util-to-text'
import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { localIsoDay, shiftIsoDay } from '../util/local-date'
import {
  TrainingPeaksApi,
  type TrainingPeaksCalendarNoteItem,
  type TrainingPeaksWorkout,
  type TrainingPeaksZoneSet,
} from '../util/trainingpeaks-api'
import {
  emptyTrainingPeaksCalendar,
  isTrainingPeaksCalendarDay,
  mergeTrainingPeaksCalendar,
  trainingPeaksCalendarCompleted,
  type TrainingPeaksCalendarMetrics,
  type TrainingPeaksCalendarNote,
  type TrainingPeaksCalendarRange,
  type TrainingPeaksCalendarSport,
  type TrainingPeaksCalendarStep,
  type TrainingPeaksCalendarStructure,
  type TrainingPeaksCalendarWorkout,
  type TrainingPeaksCalendarZones,
  type TrainingPeaksCalendarZoneSet,
} from '../util/trainingpeaks-calendar'
import {
  readTrainingPeaksCalendarCache,
  TRAININGPEAKS_CALENDAR_CACHE,
  writeTrainingPeaksCalendarCache,
} from '../util/trainingpeaks-calendar-cache'
import { refreshTriathlonRouteSource } from '../util/triathlon-cache'
import { isRecord } from '../util/type-guards'

interface CalendarRange {
  since: string
  until: string
}

function calendarDay(value: string | undefined, argument: string): string {
  if (!isTrainingPeaksCalendarDay(value))
    throw new Error(`${argument} requires a valid YYYY-MM-DD date`)
  return value
}

export function parseTrainingPeaksCalendarArgs(
  argv: readonly string[],
  today = localIsoDay(),
): CalendarRange {
  calendarDay(today, 'today')
  const yearEnd = `${today.slice(0, 4)}-12-31`
  const horizon = shiftIsoDay(today, 90)
  let since = `${today.slice(0, 4)}-01-01`
  let until = horizon > yearEnd ? horizon : yearEnd
  for (let index = 0; index < argv.length; index++) {
    const argument = argv[index]
    if (argument === '--') continue
    if (argument === '--since') since = calendarDay(argv[++index], argument)
    else if (argument === '--until') until = calendarDay(argv[++index], argument)
    else throw new Error(`Unknown TrainingPeaks calendar argument: ${argument}`)
  }
  if (since > until) throw new Error('--since must be on or before --until')
  return { since, until }
}

export function trainingPeaksCalendarMonths(since: string, until: string): CalendarRange[] {
  calendarDay(since, '--since')
  calendarDay(until, '--until')
  if (since > until) throw new Error('--since must be on or before --until')
  const ranges: CalendarRange[] = []
  for (let day = since; day <= until; ) {
    const end = new Date(`${day}T00:00:00.000Z`)
    end.setUTCMonth(end.getUTCMonth() + 1, 0)
    const monthEnd = end.toISOString().slice(0, 10)
    const last = monthEnd < until ? monthEnd : until
    ranges.push({ since: day, until: last })
    day = shiftIsoDay(last, 1)
  }
  return ranges
}

function nativeMetric(workout: TrainingPeaksWorkout, field: string): number | null {
  const value = workout[field]
  if (value === null) return null
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0)
    throw new Error(`TrainingPeaks workout ${workout.workoutId} has invalid ${field}`)
  return value
}

function nativeMetrics(
  workout: TrainingPeaksWorkout,
  planned: boolean,
): TrainingPeaksCalendarMetrics {
  const hours = nativeMetric(workout, planned ? 'totalTimePlanned' : 'totalTime')
  return {
    durationSeconds: hours === null ? null : Math.round(hours * 3600),
    distanceMeters: nativeMetric(workout, planned ? 'distancePlanned' : 'distance'),
    tss: nativeMetric(workout, planned ? 'tssPlanned' : 'tssActual'),
  }
}

function nativeTime(value: string | null, workoutId: number): string | null {
  if (value === null) return null
  const match = /T((?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d)(?:\.\d+)?$/.exec(value)
  if (!match) throw new Error(`TrainingPeaks workout ${workoutId} has an invalid local start time`)
  return match[1]
}

function plainText(value: string | null): string {
  if (!value) return ''
  return toText(fromHtml(value, { fragment: true }), { whitespace: 'pre-wrap' })
    .replace(/\r\n?/g, '\n')
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim()
}

function sport(value: number): TrainingPeaksCalendarSport {
  if (value === 1) return 'swim'
  if (value === 2) return 'bike'
  if (value === 3) return 'run'
  if (value === 9) return 'strength'
  return 'other'
}

function structureRange(value: Record<string, unknown>): TrainingPeaksCalendarRange {
  const { minValue: min, maxValue: max } = value
  if (typeof min !== 'number' || !Number.isFinite(min) || min < 0)
    throw new Error('has a target without a minimum')
  if (max != null && (typeof max !== 'number' || !Number.isFinite(max) || max < min))
    throw new Error('has an invalid target maximum')
  return { min, max: typeof max === 'number' ? max : null }
}

function structureStep(value: unknown): TrainingPeaksCalendarStep {
  if (!isRecord(value) || value.steps !== undefined) throw new Error('nests repeated steps')
  const intensity = value.intensityClass
  if (
    intensity !== 'warmUp' &&
    intensity !== 'active' &&
    intensity !== 'rest' &&
    intensity !== 'coolDown'
  )
    throw new Error(`has intensity class ${String(intensity)}`)
  const length = value.length
  if (
    !isRecord(length) ||
    length.unit !== 'second' ||
    typeof length.value !== 'number' ||
    !Number.isFinite(length.value) ||
    length.value <= 0
  )
    throw new Error('has a step length outside seconds')
  // The primary target has no unit; cadence is the only secondary target shown.
  const targets = Array.isArray(value.targets) ? value.targets.filter(isRecord) : []
  const target = targets.find(item => item.unit === undefined)
  const cadence = targets.find(item => item.unit === 'roundOrStridePerMinute')
  return {
    name: typeof value.name === 'string' ? value.name.trim() : '',
    notes: typeof value.notes === 'string' ? value.notes.trim() : '',
    intensity,
    seconds: length.value,
    target: target ? structureRange(target) : null,
    cadence: cadence ? structureRange(cadence) : null,
  }
}

export function normalizeTrainingPeaksStructure(
  value: unknown,
): TrainingPeaksCalendarStructure | null {
  if (value == null) return null
  if (!isRecord(value)) throw new Error('is not an object')
  const metric = value.primaryIntensityMetric
  if (
    metric !== 'percentOfFtp' &&
    metric !== 'percentOfThresholdPace' &&
    metric !== 'percentOfThresholdHr'
  )
    throw new Error(`uses intensity metric ${String(metric)}`)
  if (!Array.isArray(value.structure) || value.structure.length === 0)
    throw new Error('has no steps')
  return {
    metric,
    blocks: value.structure.map(block => {
      if (!isRecord(block)) throw new Error('has an invalid block')
      const kind = block.type
      if (kind !== 'step' && kind !== 'repetition' && kind !== 'rampUp' && kind !== 'rampDown')
        throw new Error(`has block type ${String(kind)}`)
      const length = block.length
      if (
        !isRecord(length) ||
        length.unit !== 'repetition' ||
        typeof length.value !== 'number' ||
        !Number.isSafeInteger(length.value) ||
        length.value < 1
      )
        throw new Error('has an invalid repeat count')
      if (!Array.isArray(block.steps) || block.steps.length === 0)
        throw new Error('has an empty block')
      return { kind, repeat: length.value, steps: block.steps.map(structureStep) }
    }),
  }
}

const calendarZoneSets = (sets: TrainingPeaksZoneSet[]): TrainingPeaksCalendarZoneSet[] =>
  sets.map(({ workoutTypeId, threshold, zones }) => ({
    workoutTypeId,
    threshold,
    zones: zones.map(({ label, minimum, maximum }) => ({ label, min: minimum, max: maximum })),
  }))

export function normalizeTrainingPeaksCalendarWorkout(
  workout: TrainingPeaksWorkout,
): TrainingPeaksCalendarWorkout {
  const date = workout.workoutDay.slice(0, 10)
  if (!isTrainingPeaksCalendarDay(date))
    throw new Error(`TrainingPeaks workout ${workout.workoutId} has an invalid workoutDay`)
  const actual = nativeMetrics(workout, false)
  const preActivityNote = plainText(workout.coachComments ?? null)
  let structure: TrainingPeaksCalendarStructure | null = null
  try {
    structure = normalizeTrainingPeaksStructure(workout.structure)
  } catch (error) {
    // The workout metrics stay usable when TrainingPeaks adds a structure shape this view cannot draw.
    console.warn(
      `[trainingpeaks-calendar] workout ${workout.workoutId} structure ${error instanceof Error ? error.message : 'is invalid'}; omitted`,
    )
  }
  return {
    id: String(workout.workoutId),
    source: 'trainingpeaks',
    date,
    title: plainText(workout.title).replace(/\s+/g, ' ') || 'Untitled workout',
    sport: sport(workout.workoutTypeValueId),
    workoutTypeId: workout.workoutTypeValueId,
    description: plainText(workout.description),
    planned: nativeMetrics(workout, true),
    actual,
    status: trainingPeaksCalendarCompleted(actual) ? 'completed' : 'planned',
    startTime: nativeTime(workout.startTime, workout.workoutId),
    startTimePlanned: nativeTime(workout.startTimePlanned, workout.workoutId),
    order:
      typeof workout.orderOnDay === 'number' && Number.isFinite(workout.orderOnDay)
        ? workout.orderOnDay
        : null,
    ...(preActivityNote ? { preActivityNote } : {}),
    ...(structure ? { structure } : {}),
  }
}

/** Hidden notes stay private to TrainingPeaks. */
export function normalizeTrainingPeaksCalendarNotes(
  items: readonly TrainingPeaksCalendarNoteItem[],
): TrainingPeaksCalendarNote[] {
  return items.flatMap(item => {
    const title = plainText(item.title).replace(/\s+/g, ' ')
    const description = plainText(item.description)
    return item.hidden || !isTrainingPeaksCalendarDay(item.date) || (!title && !description)
      ? []
      : [{ id: String(item.id), date: item.date, title, description }]
  })
}

async function main(argv: readonly string[]): Promise<void> {
  if (argv.includes('--help') || argv.includes('-h')) {
    console.log(
      [
        'usage: pnpm trainingpeaks:calendar -- [--since YYYY-MM-DD] [--until YYYY-MM-DD]',
        'Reads TrainingPeaks in calendar-month chunks and updates the local calendar cache.',
        'Defaults to January 1 of the current local year through the later of December 31 or today + 90 days.',
        'No remote workouts are changed. A failed or incomplete pull leaves the existing cache untouched.',
      ].join('\n'),
    )
    return
  }
  const range = parseTrainingPeaksCalendarArgs(argv)
  const api = new TrainingPeaksApi({
    accessToken: process.env.TRAININGPEAKS_ACCESS_TOKEN,
    authCookie: process.env.TRAININGPEAKS_AUTH_COOKIE,
  })
  const athleteId = await api.athleteId()
  const workouts: TrainingPeaksCalendarWorkout[] = []
  let notes: TrainingPeaksCalendarNote[] | null = []
  for (const month of trainingPeaksCalendarMonths(range.since, range.until)) {
    const items = (await api.workouts(athleteId, month.since, month.until)).map(
      normalizeTrainingPeaksCalendarWorkout,
    )
    if (items.some(item => item.date < month.since || item.date > month.until))
      throw new Error(
        `TrainingPeaks returned workouts outside ${month.since}..${month.until}; cache retained`,
      )
    workouts.push(...items)
    console.log(
      `[trainingpeaks-calendar] fetched ${month.since}..${month.until}: ${items.length} workouts`,
    )
    if (!notes) continue
    try {
      const monthNotes = normalizeTrainingPeaksCalendarNotes(
        await api.calendarNotes(athleteId, month.since, month.until),
      )
      if (monthNotes.some(item => item.date < month.since || item.date > month.until))
        throw new Error(`returned notes outside ${month.since}..${month.until}`)
      notes.push(...monthNotes)
    } catch (error) {
      console.warn(
        `[trainingpeaks-calendar] notes ${error instanceof Error ? error.message : 'could not be read'}; kept the cached notes`,
      )
      notes = null
    }
  }
  let zones: TrainingPeaksCalendarZones | null = null
  try {
    const settings = await api.zones(athleteId)
    zones = {
      heartRate: calendarZoneSets(settings.heartRate),
      power: calendarZoneSets(settings.power),
      speed: calendarZoneSets(settings.speed),
    }
  } catch (error) {
    console.warn(
      `[trainingpeaks-calendar] zones ${error instanceof Error ? error.message : 'could not be read'}; kept the cached zones`,
    )
  }
  const previous = await readTrainingPeaksCalendarCache()
  if (previous && previous.athleteId !== athleteId)
    throw new Error('TrainingPeaks calendar belongs to a different athlete; cache retained')
  const fetchedAt = new Date().toISOString()
  const merged = mergeTrainingPeaksCalendar(
    previous?.calendar ?? emptyTrainingPeaksCalendar(),
    workouts,
    { ...range, fetchedAt },
    notes ?? undefined,
  )
  const calendar = zones ? { ...merged, zones } : merged
  await writeTrainingPeaksCalendarCache({ athleteId, calendar })
  await refreshTriathlonRouteSource()
  const completed = workouts.filter(workout => workout.status === 'completed').length
  const structured = workouts.filter(workout => workout.structure).length
  const preActivity = workouts.filter(workout => workout.preActivityNote).length
  console.log(
    `[trainingpeaks-calendar] saved ${workouts.length} workouts (${completed} completed, ${workouts.length - completed} planned, ${structured} structured, ${preActivity} with pre-activity notes); ${notes ? `${notes.length} notes; ` : ''}${calendar.workouts.length} total cached`,
  )
  console.log(
    `[trainingpeaks-calendar] coverage ${range.since}..${range.until}; cache ${TRAININGPEAKS_CALENDAR_CACHE}`,
  )
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main(process.argv.slice(2)).catch((error: unknown) => {
    console.error(
      `[trainingpeaks-calendar] ${error instanceof Error ? error.message : 'Sync failed'}`,
    )
    process.exitCode = 1
  })
}
