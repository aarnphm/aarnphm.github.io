import { isRecord } from './type-guards'

export type TrainingPeaksCalendarSport = 'swim' | 'bike' | 'run' | 'strength' | 'other'

export interface TrainingPeaksCalendarMetrics {
  durationSeconds: number | null
  distanceMeters: number | null
  tss: number | null
}

export const TRAINING_PEAK_SECONDS: readonly number[] = [5, 60, 300, 1200, 3600]

export interface TrainingPeaksCalendarPeak {
  seconds: number
  heartRateBpm: number | null
  heartRateSource: 'garmin' | 'strava' | null
  powerWatts: number | null
}

export interface TrainingPeaksCalendarActivityLink {
  id: number
  date: string
  title: string
  match: 'source-id' | 'recording'
  peaks?: TrainingPeaksCalendarPeak[]
}

/** Targets are percentages of the athlete threshold that the metric names. */
export type TrainingPeaksCalendarIntensityMetric =
  | 'percentOfFtp'
  | 'percentOfThresholdPace'
  | 'percentOfThresholdHr'

export interface TrainingPeaksCalendarRange {
  min: number
  max: number | null
}

export interface TrainingPeaksCalendarStep {
  name: string
  notes: string
  intensity: 'warmUp' | 'active' | 'rest' | 'coolDown'
  seconds: number
  target: TrainingPeaksCalendarRange | null
  cadence: TrainingPeaksCalendarRange | null
}

export interface TrainingPeaksCalendarBlock {
  kind: 'step' | 'repetition' | 'rampUp' | 'rampDown'
  repeat: number
  steps: TrainingPeaksCalendarStep[]
}

export interface TrainingPeaksCalendarStructure {
  metric: TrainingPeaksCalendarIntensityMetric
  blocks: TrainingPeaksCalendarBlock[]
}

export interface TrainingPeaksCalendarZone {
  label: string
  min: number
  max: number
}

/** workoutTypeId 0 is the athlete default for sports without their own set. */
export interface TrainingPeaksCalendarZoneSet {
  workoutTypeId: number
  threshold: number
  zones: TrainingPeaksCalendarZone[]
}

export interface TrainingPeaksCalendarZones {
  heartRate: TrainingPeaksCalendarZoneSet[]
  power: TrainingPeaksCalendarZoneSet[]
  speed: TrainingPeaksCalendarZoneSet[]
}

/** Ascending upper bounds; the zone above the last bound is open. */
export interface TrainingPeaksCalendarZoneBounds {
  threshold: number
  bounds: number[]
}

/** Garden zones added at emit time for bike and run workouts without a TrainingPeaks zone set. */
export interface TrainingPeaksCalendarLocalZones {
  /** W, on FTP. */
  power: TrainingPeaksCalendarZoneBounds | null
  /** m/s, on lactate threshold pace. */
  runSpeed: TrainingPeaksCalendarZoneBounds | null
  /** bpm, on lactate threshold heart rate. */
  heartRate: TrainingPeaksCalendarZoneBounds | null
}

export interface TrainingPeaksCalendarWorkout {
  id: string
  source: 'trainingpeaks'
  date: string
  title: string
  sport: TrainingPeaksCalendarSport
  workoutTypeId: number
  description: string
  planned: TrainingPeaksCalendarMetrics
  actual: TrainingPeaksCalendarMetrics
  status: 'completed' | 'planned'
  startTime: string | null
  startTimePlanned: string | null
  order: number | null
  /** The coach's pre-activity comment, when there is one. */
  preActivityNote?: string
  structure?: TrainingPeaksCalendarStructure
  activity?: TrainingPeaksCalendarActivityLink
}

export interface TrainingPeaksCalendarNote {
  id: string
  date: string
  title: string
  description: string
}

export interface TrainingPeaksCalendarCoverage {
  since: string
  until: string
  fetchedAt: string
}

export interface TrainingPeaksCalendar {
  version: 1
  source: 'trainingpeaks'
  fetchedAt: string | null
  coverage: TrainingPeaksCalendarCoverage[]
  workouts: TrainingPeaksCalendarWorkout[]
  notes?: TrainingPeaksCalendarNote[]
  zones?: TrainingPeaksCalendarZones
  localZones?: TrainingPeaksCalendarLocalZones
}

export function isTrainingPeaksCalendarDay(value: unknown): value is string {
  if (typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return false
  const date = new Date(`${value}T00:00:00.000Z`)
  return Number.isFinite(date.getTime()) && date.toISOString().slice(0, 10) === value
}

function timestamp(value: unknown): value is string {
  return typeof value === 'string' && value.endsWith('Z') && Number.isFinite(Date.parse(value))
}

function metric(value: unknown): value is number | null {
  return value === null || (typeof value === 'number' && Number.isFinite(value) && value >= 0)
}

function metrics(value: unknown): value is TrainingPeaksCalendarMetrics {
  return (
    isRecord(value) &&
    metric(value.durationSeconds) &&
    metric(value.distanceMeters) &&
    metric(value.tss)
  )
}

function localTime(value: unknown): value is string | null {
  return (
    value === null ||
    (typeof value === 'string' && /^(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d$/.test(value))
  )
}

function sport(value: unknown): value is TrainingPeaksCalendarSport {
  return (
    value === 'swim' ||
    value === 'bike' ||
    value === 'run' ||
    value === 'strength' ||
    value === 'other'
  )
}

function activityLink(value: unknown): value is TrainingPeaksCalendarActivityLink {
  return (
    isRecord(value) &&
    typeof value.id === 'number' &&
    Number.isSafeInteger(value.id) &&
    value.id > 0 &&
    isTrainingPeaksCalendarDay(value.date) &&
    typeof value.title === 'string' &&
    (value.match === 'source-id' || value.match === 'recording') &&
    (value.peaks === undefined || activityPeaks(value.peaks))
  )
}

function activityPeaks(value: unknown): value is TrainingPeaksCalendarPeak[] {
  if (!Array.isArray(value) || value.length === 0 || value.length > TRAINING_PEAK_SECONDS.length)
    return false
  const durations = new Set<number>()
  return value.every(point => {
    if (
      !isRecord(point) ||
      typeof point.seconds !== 'number' ||
      !TRAINING_PEAK_SECONDS.includes(point.seconds) ||
      durations.has(point.seconds) ||
      !metric(point.powerWatts) ||
      !metric(point.heartRateBpm) ||
      (point.heartRateBpm === null
        ? point.heartRateSource !== null
        : point.heartRateBpm <= 0 ||
          (point.heartRateSource !== 'garmin' && point.heartRateSource !== 'strava'))
    )
      return false
    durations.add(point.seconds)
    return true
  })
}

function positive(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value > 0
}

function targetRange(value: unknown): value is TrainingPeaksCalendarRange | null {
  return (
    value === null ||
    (isRecord(value) &&
      typeof value.min === 'number' &&
      Number.isFinite(value.min) &&
      value.min >= 0 &&
      (value.max === null ||
        (typeof value.max === 'number' && Number.isFinite(value.max) && value.max >= value.min)))
  )
}

function structureStep(value: unknown): value is TrainingPeaksCalendarStep {
  return (
    isRecord(value) &&
    typeof value.name === 'string' &&
    typeof value.notes === 'string' &&
    (value.intensity === 'warmUp' ||
      value.intensity === 'active' ||
      value.intensity === 'rest' ||
      value.intensity === 'coolDown') &&
    positive(value.seconds) &&
    targetRange(value.target) &&
    targetRange(value.cadence)
  )
}

function structure(value: unknown): value is TrainingPeaksCalendarStructure {
  return (
    isRecord(value) &&
    (value.metric === 'percentOfFtp' ||
      value.metric === 'percentOfThresholdPace' ||
      value.metric === 'percentOfThresholdHr') &&
    Array.isArray(value.blocks) &&
    value.blocks.length > 0 &&
    value.blocks.every(
      block =>
        isRecord(block) &&
        (block.kind === 'step' ||
          block.kind === 'repetition' ||
          block.kind === 'rampUp' ||
          block.kind === 'rampDown') &&
        typeof block.repeat === 'number' &&
        Number.isSafeInteger(block.repeat) &&
        block.repeat > 0 &&
        Array.isArray(block.steps) &&
        block.steps.length > 0 &&
        block.steps.every(structureStep),
    )
  )
}

function zoneSet(value: unknown): value is TrainingPeaksCalendarZoneSet {
  if (
    !isRecord(value) ||
    typeof value.workoutTypeId !== 'number' ||
    !Number.isSafeInteger(value.workoutTypeId) ||
    value.workoutTypeId < 0 ||
    !positive(value.threshold) ||
    !Array.isArray(value.zones) ||
    value.zones.length === 0
  )
    return false
  let previous = -Infinity
  for (const zone of value.zones) {
    if (
      !isRecord(zone) ||
      typeof zone.label !== 'string' ||
      typeof zone.min !== 'number' ||
      typeof zone.max !== 'number' ||
      !Number.isFinite(zone.min) ||
      !Number.isFinite(zone.max) ||
      zone.min > zone.max ||
      zone.max < previous
    )
      return false
    previous = zone.max
  }
  return true
}

function zoneSettings(value: unknown): value is TrainingPeaksCalendarZones {
  return (
    isRecord(value) &&
    [value.heartRate, value.power, value.speed].every(
      sets => Array.isArray(sets) && sets.every(zoneSet),
    )
  )
}

function zoneBounds(value: unknown): value is TrainingPeaksCalendarZoneBounds | null {
  return (
    value === null ||
    (isRecord(value) &&
      positive(value.threshold) &&
      Array.isArray(value.bounds) &&
      value.bounds.length > 0 &&
      value.bounds.every(
        (bound, index, bounds) => positive(bound) && (index === 0 || bound > bounds[index - 1]),
      ))
  )
}

function localZones(value: unknown): value is TrainingPeaksCalendarLocalZones {
  return (
    isRecord(value) &&
    zoneBounds(value.power) &&
    zoneBounds(value.runSpeed) &&
    zoneBounds(value.heartRate)
  )
}

const copyBounds = (
  value: TrainingPeaksCalendarZoneBounds | null,
): TrainingPeaksCalendarZoneBounds | null =>
  value && { threshold: value.threshold, bounds: [...value.bounds] }

const copyRange = (value: TrainingPeaksCalendarRange | null): TrainingPeaksCalendarRange | null =>
  value && { min: value.min, max: value.max }

function copyStructure(value: TrainingPeaksCalendarStructure): TrainingPeaksCalendarStructure {
  return {
    metric: value.metric,
    blocks: value.blocks.map(block => ({
      kind: block.kind,
      repeat: block.repeat,
      steps: block.steps.map(item => ({
        name: item.name,
        notes: item.notes,
        intensity: item.intensity,
        seconds: item.seconds,
        target: copyRange(item.target),
        cadence: copyRange(item.cadence),
      })),
    })),
  }
}

function copyZones(value: TrainingPeaksCalendarZones): TrainingPeaksCalendarZones {
  const sets = (items: TrainingPeaksCalendarZoneSet[]): TrainingPeaksCalendarZoneSet[] =>
    items.map(({ workoutTypeId, threshold, zones }) => ({
      workoutTypeId,
      threshold,
      zones: zones.map(({ label, min, max }) => ({ label, min, max })),
    }))
  return { heartRate: sets(value.heartRate), power: sets(value.power), speed: sets(value.speed) }
}

export function trainingPeaksCalendarCompleted(actual: TrainingPeaksCalendarMetrics): boolean {
  return (
    (actual.durationSeconds ?? 0) > 0 || (actual.distanceMeters ?? 0) > 0 || (actual.tss ?? 0) > 0
  )
}

function workout(value: unknown): value is TrainingPeaksCalendarWorkout {
  return (
    isRecord(value) &&
    typeof value.id === 'string' &&
    /^[1-9]\d*$/.test(value.id) &&
    value.source === 'trainingpeaks' &&
    isTrainingPeaksCalendarDay(value.date) &&
    typeof value.title === 'string' &&
    sport(value.sport) &&
    typeof value.workoutTypeId === 'number' &&
    Number.isSafeInteger(value.workoutTypeId) &&
    value.workoutTypeId > 0 &&
    typeof value.description === 'string' &&
    metrics(value.planned) &&
    metrics(value.actual) &&
    value.status === (trainingPeaksCalendarCompleted(value.actual) ? 'completed' : 'planned') &&
    localTime(value.startTime) &&
    localTime(value.startTimePlanned) &&
    (value.order === null || (typeof value.order === 'number' && Number.isFinite(value.order))) &&
    (value.preActivityNote === undefined ||
      (typeof value.preActivityNote === 'string' && value.preActivityNote !== '')) &&
    (value.structure === undefined || structure(value.structure)) &&
    (value.activity === undefined ||
      (value.status === 'completed' &&
        activityLink(value.activity) &&
        value.activity.date === value.date))
  )
}

function note(value: unknown): value is TrainingPeaksCalendarNote {
  return (
    isRecord(value) &&
    typeof value.id === 'string' &&
    /^[1-9]\d*$/.test(value.id) &&
    isTrainingPeaksCalendarDay(value.date) &&
    typeof value.title === 'string' &&
    typeof value.description === 'string' &&
    (value.title !== '' || value.description !== '')
  )
}

const noteOrder = (a: TrainingPeaksCalendarNote, b: TrainingPeaksCalendarNote): number =>
  a.date.localeCompare(b.date) || a.id.localeCompare(b.id)

function coverage(value: unknown): value is TrainingPeaksCalendarCoverage {
  return (
    isRecord(value) &&
    isTrainingPeaksCalendarDay(value.since) &&
    isTrainingPeaksCalendarDay(value.until) &&
    value.since <= value.until &&
    timestamp(value.fetchedAt)
  )
}

export function emptyTrainingPeaksCalendar(): TrainingPeaksCalendar {
  return { version: 1, source: 'trainingpeaks', fetchedAt: null, coverage: [], workouts: [] }
}

export function parseTrainingPeaksCalendar(value: unknown): TrainingPeaksCalendar | null {
  if (
    !isRecord(value) ||
    value.version !== 1 ||
    value.source !== 'trainingpeaks' ||
    !(value.fetchedAt === null || timestamp(value.fetchedAt)) ||
    !Array.isArray(value.coverage) ||
    !value.coverage.every(coverage) ||
    !Array.isArray(value.workouts) ||
    !value.workouts.every(workout) ||
    !(value.notes === undefined || (Array.isArray(value.notes) && value.notes.every(note))) ||
    !(value.zones === undefined || zoneSettings(value.zones)) ||
    !(value.localZones === undefined || localZones(value.localZones))
  )
    return null
  const ranges = value.coverage.toSorted((a, b) => a.since.localeCompare(b.since))
  if (ranges.some((range, index) => index > 0 && ranges[index - 1].until >= range.since))
    return null
  if (value.fetchedAt === null && (ranges.length > 0 || value.workouts.length > 0)) return null
  const ids = new Set<string>()
  for (const item of value.workouts) {
    if (
      ids.has(item.id) ||
      !ranges.some(range => range.since <= item.date && range.until >= item.date)
    )
      return null
    ids.add(item.id)
  }
  const noteIds = new Set<string>()
  for (const item of value.notes ?? []) {
    if (
      noteIds.has(item.id) ||
      !ranges.some(range => range.since <= item.date && range.until >= item.date)
    )
      return null
    noteIds.add(item.id)
  }
  return {
    version: 1,
    source: 'trainingpeaks',
    fetchedAt: value.fetchedAt,
    coverage: ranges.map(({ since, until, fetchedAt }) => ({ since, until, fetchedAt })),
    workouts: value.workouts.map(item => ({
      id: item.id,
      source: 'trainingpeaks',
      date: item.date,
      title: item.title,
      sport: item.sport,
      workoutTypeId: item.workoutTypeId,
      description: item.description,
      planned: {
        durationSeconds: item.planned.durationSeconds,
        distanceMeters: item.planned.distanceMeters,
        tss: item.planned.tss,
      },
      actual: {
        durationSeconds: item.actual.durationSeconds,
        distanceMeters: item.actual.distanceMeters,
        tss: item.actual.tss,
      },
      status: item.status,
      startTime: item.startTime,
      startTimePlanned: item.startTimePlanned,
      order: item.order,
      ...(item.preActivityNote ? { preActivityNote: item.preActivityNote } : {}),
      ...(item.structure ? { structure: copyStructure(item.structure) } : {}),
      ...(item.activity
        ? {
            activity: {
              id: item.activity.id,
              date: item.activity.date,
              title: item.activity.title,
              match: item.activity.match,
              ...(item.activity.peaks
                ? {
                    peaks: item.activity.peaks.map(
                      ({ seconds, heartRateBpm, heartRateSource, powerWatts }) => ({
                        seconds,
                        heartRateBpm,
                        heartRateSource,
                        powerWatts,
                      }),
                    ),
                  }
                : {}),
            },
          }
        : {}),
    })),
    ...(value.notes?.length
      ? {
          notes: value.notes
            .map(({ id, date, title, description }) => ({ id, date, title, description }))
            .sort(noteOrder),
        }
      : {}),
    ...(value.zones ? { zones: copyZones(value.zones) } : {}),
    ...(value.localZones
      ? {
          localZones: {
            power: copyBounds(value.localZones.power),
            runSpeed: copyBounds(value.localZones.runSpeed),
            heartRate: copyBounds(value.localZones.heartRate),
          },
        }
      : {}),
  }
}

function adjacentDay(day: string, offset: number): string {
  const date = new Date(`${day}T00:00:00.000Z`)
  date.setUTCDate(date.getUTCDate() + offset)
  return date.toISOString().slice(0, 10)
}

/** Without refreshed notes (a failed note pull), the cached notes stay as they were. */
export function mergeTrainingPeaksCalendar(
  previous: TrainingPeaksCalendar,
  refreshed: readonly TrainingPeaksCalendarWorkout[],
  range: TrainingPeaksCalendarCoverage,
  refreshedNotes?: readonly TrainingPeaksCalendarNote[],
): TrainingPeaksCalendar {
  if (!coverage(range))
    throw new Error('TrainingPeaks refresh requires valid inclusive dates and timestamp')
  const ids = new Set<string>()
  for (const item of refreshed) {
    if (!workout(item)) throw new Error('TrainingPeaks refresh contains an invalid workout')
    if (item.date < range.since || item.date > range.until)
      throw new Error(`TrainingPeaks workout ${item.id} falls outside the refresh range`)
    if (ids.has(item.id))
      throw new Error(`TrainingPeaks refresh contains duplicate workout ${item.id}`)
    ids.add(item.id)
  }
  const noteIds = new Set<string>()
  for (const item of refreshedNotes ?? []) {
    if (!note(item)) throw new Error('TrainingPeaks refresh contains an invalid note')
    if (item.date < range.since || item.date > range.until)
      throw new Error(`TrainingPeaks note ${item.id} falls outside the refresh range`)
    if (noteIds.has(item.id))
      throw new Error(`TrainingPeaks refresh contains duplicate note ${item.id}`)
    noteIds.add(item.id)
  }
  const notes = refreshedNotes
    ? [
        ...(previous.notes ?? []).filter(
          item => !noteIds.has(item.id) && (item.date < range.since || item.date > range.until),
        ),
        ...refreshedNotes,
      ].sort(noteOrder)
    : (previous.notes ?? [])
  const ranges: TrainingPeaksCalendarCoverage[] = [range]
  for (const old of previous.coverage) {
    if (old.until < range.since || old.since > range.until) ranges.push(old)
    else {
      if (old.since < range.since) ranges.push({ ...old, until: adjacentDay(range.since, -1) })
      if (old.until > range.until) ranges.push({ ...old, since: adjacentDay(range.until, 1) })
    }
  }
  const mergedCoverage: TrainingPeaksCalendarCoverage[] = []
  for (const item of ranges.toSorted((a, b) => a.since.localeCompare(b.since))) {
    const last = mergedCoverage.at(-1)
    if (last && last.fetchedAt === item.fetchedAt && adjacentDay(last.until, 1) === item.since)
      last.until = item.until
    else mergedCoverage.push({ ...item })
  }
  return {
    version: 1,
    source: 'trainingpeaks',
    fetchedAt: range.fetchedAt,
    coverage: mergedCoverage,
    ...(notes.length > 0 ? { notes } : {}),
    ...(previous.zones ? { zones: previous.zones } : {}),
    workouts: [
      ...previous.workouts.filter(
        item => !ids.has(item.id) && (item.date < range.since || item.date > range.until),
      ),
      ...refreshed,
    ].sort(
      (a, b) =>
        a.date.localeCompare(b.date) ||
        (a.order ?? Infinity) - (b.order ?? Infinity) ||
        (a.startTimePlanned ?? a.startTime ?? '').localeCompare(
          b.startTimePlanned ?? b.startTime ?? '',
        ) ||
        a.id.localeCompare(b.id),
    ),
  }
}
