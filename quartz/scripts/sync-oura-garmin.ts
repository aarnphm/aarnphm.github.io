import fs from 'node:fs/promises'
import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import type { OuraCache, OuraDayDetail, OuraSeries } from '../plugins/stores/oura'
import {
  cleanGarminConnectBaseUrl,
  DEFAULT_GARMIN_CONNECT_BASE,
  fetchGarminJson,
  garminErrorMessage,
  readGarminConnectSession,
  type GarminConnectSession,
} from '../util/garmin-session'
import { joinSegments, QUARTZ } from '../util/path'
import { isRecord, readNumber, readString, type UnknownRecord } from '../util/type-guards'

export type SleepLevel = 'unmeasurable' | 'awake' | 'light' | 'deep' | 'rem'

export interface SleepStage {
  startTime: Date
  level: SleepLevel
}

export interface HeartRateSample {
  time: Date
  heartRateBpm: number
}

export interface OuraSleepDurations {
  timeInBedSeconds: number
  totalSleepSeconds: number
  deepSeconds: number
  lightSeconds: number
  remSeconds: number
  awakeSeconds: number
}

type OuraSleepDurationFields = Pick<
  OuraDayDetail,
  'timeInBedS' | 'totalSleepS' | 'deepS' | 'lightS' | 'remS' | 'awakeS'
>

export interface GarminManualSleepPayload {
  calendarDate: string
  sleepStartTimestampGMT: number
  sleepEndTimestampGMT: number
  sleepTimeSeconds: number
  napTimeSeconds: 0
  sleepWindowConfirmed: true
  sleepWindowConfirmationType: 'manually_confirmed'
  userProfilePK: number
}

const PHASE_LEVELS: Readonly<Record<string, SleepLevel>> = {
  '1': 'deep',
  '2': 'light',
  '3': 'rem',
  '4': 'awake',
}
const PHASE_SECONDS = 300
const DEFAULT_SYNC_DELAY_MS = 1200
const ouraCacheFile = joinSegments(QUARTZ, '.quartz-cache', 'oura.json')

interface Args {
  days: string[]
  count: number
  since: string | null
  write: boolean
}

interface Night {
  day: string
  sleepStart: Date
  sleepEnd: Date
  stages: SleepStage[]
  durations: OuraSleepDurations | null
  heartRate: HeartRateSample[]
  restingHeartRate: number | null
}

interface GarminProfile {
  displayName: string
  userProfileId: number
}

interface GarminSleepState {
  dto: UnknownRecord | null
  levels: number
}

type SyncAction = 'created' | 'updated' | 'already present' | 'preserved Garmin-recorded'

function readArgValue(argv: string[], index: number, flag: string): string {
  const value = argv[index]
  if (!value || value.startsWith('--')) throw new Error(`${flag} needs a value`)
  return value
}

function cleanDay(value: string): string {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) throw new Error(`${value} is not YYYY-MM-DD`)
  return value
}

function parseArgs(argv: string[]): Args {
  const args: Args = { days: [], count: 1, since: null, write: false }
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]
    if (arg === '--write') args.write = true
    else if (arg === '--day') args.days.push(cleanDay(readArgValue(argv, ++i, arg)))
    else if (arg === '--since') args.since = cleanDay(readArgValue(argv, ++i, arg))
    else if (arg === '--days') {
      const count = Number(readArgValue(argv, ++i, arg))
      if (!Number.isInteger(count) || count < 1)
        throw new Error('--days must be a positive integer')
      args.count = count
    } else throw new Error(`unknown flag ${arg}`)
  }
  return args
}

export function sleepStages(phase5Min: string, sleepStart: Date): SleepStage[] {
  const stages: SleepStage[] = []
  for (let i = 0; i < phase5Min.length; i++) {
    const level = PHASE_LEVELS[phase5Min[i]]
    if (!level) continue
    stages.push({ startTime: new Date(sleepStart.valueOf() + i * PHASE_SECONDS * 1000), level })
  }
  return stages
}

export function seriesSamples(
  series: OuraSeries,
  sleepStart: Date,
  sleepEnd: Date,
): HeartRateSample[] {
  const startMs = Date.parse(series.startTs)
  if (!Number.isFinite(startMs)) throw new Error(`${series.startTs} is not a timestamp`)
  const out: HeartRateSample[] = []
  for (let i = 0; i < series.items.length; i++) {
    const value = series.items[i]
    if (value == null || !Number.isFinite(value)) continue
    const time = new Date(startMs + i * series.intervalS * 1000)
    if (time < sleepStart || time > sleepEnd) continue
    out.push({ time, heartRateBpm: value })
  }
  return out
}

function isDurationSeconds(value: number | null): value is number {
  return value != null && Number.isFinite(value) && value >= 0
}

export function ouraSleepDurations(detail: OuraSleepDurationFields): OuraSleepDurations | null {
  const { timeInBedS, totalSleepS, deepS, lightS, remS, awakeS } = detail
  if (
    !isDurationSeconds(timeInBedS) ||
    !isDurationSeconds(totalSleepS) ||
    !isDurationSeconds(deepS) ||
    !isDurationSeconds(lightS) ||
    !isDurationSeconds(remS) ||
    !isDurationSeconds(awakeS)
  )
    return null
  return {
    timeInBedSeconds: timeInBedS,
    totalSleepSeconds: totalSleepS,
    deepSeconds: deepS,
    lightSeconds: lightS,
    remSeconds: remS,
    awakeSeconds: awakeS,
  }
}

export function garminManualSleepPayload(
  day: string,
  sleepStart: Date,
  sleepEnd: Date,
  userProfilePK: number,
): GarminManualSleepPayload {
  const start = sleepStart.valueOf()
  const end = sleepEnd.valueOf()
  if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start)
    throw new Error('manual sleep needs a valid increasing interval')
  if (!Number.isSafeInteger(userProfilePK) || userProfilePK <= 0)
    throw new Error('userProfilePK must be a positive integer')
  return {
    calendarDate: cleanDay(day),
    sleepStartTimestampGMT: start,
    sleepEndTimestampGMT: end,
    sleepTimeSeconds: (end - start) / 1000,
    napTimeSeconds: 0,
    sleepWindowConfirmed: true,
    sleepWindowConfirmationType: 'manually_confirmed',
    userProfilePK,
  }
}

export function manualSleepMatches(value: unknown, expected: GarminManualSleepPayload): boolean {
  if (!isRecord(value)) return false
  return (
    readString(value, 'calendarDate') === expected.calendarDate &&
    readNumber(value, 'sleepStartTimestampGMT') === expected.sleepStartTimestampGMT &&
    readNumber(value, 'sleepEndTimestampGMT') === expected.sleepEndTimestampGMT &&
    readNumber(value, 'sleepTimeSeconds') === expected.sleepTimeSeconds &&
    value.sleepWindowConfirmed === true &&
    readString(value, 'sleepWindowConfirmationType')?.toLowerCase() ===
      expected.sleepWindowConfirmationType
  )
}

function toNight(detail: OuraDayDetail): Night | null {
  if (!detail.bedtimeStart || !detail.bedtimeEnd || !detail.phase5Min) return null
  const sleepStart = new Date(detail.bedtimeStart)
  const sleepEnd = new Date(detail.bedtimeEnd)
  if (!Number.isFinite(sleepStart.valueOf()) || !Number.isFinite(sleepEnd.valueOf())) return null
  const stages = sleepStages(detail.phase5Min, sleepStart).filter(
    stage => stage.startTime <= sleepEnd,
  )
  if (stages.length < 2) return null
  return {
    day: detail.date,
    sleepStart,
    sleepEnd,
    stages,
    durations: ouraSleepDurations(detail),
    heartRate: detail.hr ? seriesSamples(detail.hr, sleepStart, sleepEnd) : [],
    restingHeartRate: detail.lowestHr,
  }
}

function selectNights(cache: OuraCache, args: Args): Night[] {
  const nights: Night[] = []
  const details = cache.details ?? {}
  for (const day of Object.keys(details).sort()) {
    if (args.days.length && !args.days.includes(day)) continue
    if (args.since && day < args.since) continue
    const night = toNight(details[day])
    if (night) nights.push(night)
  }
  if (args.days.length || args.since) return nights
  return nights.slice(-args.count)
}

function isOuraCache(value: unknown): value is OuraCache {
  return (
    isRecord(value) &&
    typeof value.lastSync === 'number' &&
    isRecord(value.days) &&
    (value.details == null || isRecord(value.details))
  )
}

function durationMinutes(seconds: number): string {
  return `${Math.round(seconds / 60)}min`
}

function describe(night: Night): string {
  const durations = night.durations
  const summary = durations
    ? [
        `  Oura ${night.stages.length} five-minute stages, asleep ${durationMinutes(durations.totalSleepSeconds)} of ${durationMinutes(durations.timeInBedSeconds)} in bed`,
        `  deep ${durationMinutes(durations.deepSeconds)} light ${durationMinutes(durations.lightSeconds)} rem ${durationMinutes(durations.remSeconds)} awake ${durationMinutes(durations.awakeSeconds)}`,
      ]
    : [`  Oura ${night.stages.length} five-minute stages, exact duration totals unavailable`]
  return [
    `${night.day} ${night.sleepStart.toISOString()} → ${night.sleepEnd.toISOString()}`,
    ...summary,
    `  heart rate ${night.heartRate.length} samples, resting ${night.restingHeartRate ?? 'n/a'}`,
    '  Garmin manual sleep stores the bed/wake interval; stages and heart rate remain Oura-sourced',
  ].join('\n')
}

function sleepState(raw: unknown): GarminSleepState {
  if (!isRecord(raw)) return { dto: null, levels: 0 }
  return {
    dto: isRecord(raw.dailySleepDTO) ? raw.dailySleepDTO : null,
    levels: Array.isArray(raw.sleepLevels) ? raw.sleepLevels.length : 0,
  }
}

function sleepStateDescription(state: GarminSleepState): string {
  const dto = state.dto
  return `sleepTimeSeconds=${String(dto?.sleepTimeSeconds ?? null)} deep=${String(dto?.deepSleepSeconds ?? null)} light=${String(dto?.lightSleepSeconds ?? null)} rem=${String(dto?.remSleepSeconds ?? null)} awake=${String(dto?.awakeSleepSeconds ?? null)} levels=${state.levels} source=${String(dto?.sleepWindowConfirmationType ?? null)}`
}

async function readGarminSleep(
  session: GarminConnectSession,
  base: string,
  displayName: string,
  day: string,
): Promise<GarminSleepState> {
  return sleepState(
    await fetchGarminJson(
      session,
      base,
      `/wellness-service/wellness/dailySleepData/${encodeURIComponent(displayName)}`,
      new URLSearchParams({ date: day, nonSleepBufferMinutes: '60' }),
    ),
  )
}

async function readGarminProfile(
  session: GarminConnectSession,
  base: string,
  day: string,
): Promise<GarminProfile> {
  const profile = await fetchGarminJson(session, base, '/userprofile-service/socialProfile')
  if (!isRecord(profile)) throw new Error('Garmin profile returned a non-object response')
  const displayName = readString(profile, 'displayName')
  if (!displayName) throw new Error('Garmin profile omitted displayName')
  const summary = await fetchGarminJson(
    session,
    base,
    `/usersummary-service/usersummary/daily/${encodeURIComponent(displayName)}`,
    new URLSearchParams({ calendarDate: day }),
  )
  if (!isRecord(summary)) throw new Error('Garmin daily summary returned a non-object response')
  const userProfileId = readNumber(summary, 'userProfileId')
  if (userProfileId == null || !Number.isSafeInteger(userProfileId) || userProfileId <= 0)
    throw new Error('Garmin daily summary omitted a valid userProfileId')
  return { displayName, userProfileId }
}

function isManualSleep(dto: UnknownRecord): boolean {
  return readString(dto, 'sleepWindowConfirmationType')?.toLowerCase() === 'manually_confirmed'
}

function hasGarminSleep(dto: UnknownRecord): boolean {
  return (
    readNumber(dto, 'id') != null ||
    readNumber(dto, 'sleepStartTimestampGMT') != null ||
    readNumber(dto, 'sleepEndTimestampGMT') != null ||
    readNumber(dto, 'sleepTimeSeconds') != null
  )
}

async function syncNight(
  session: GarminConnectSession,
  base: string,
  profile: GarminProfile,
  night: Night,
): Promise<{ action: SyncAction; state: GarminSleepState }> {
  const payload = garminManualSleepPayload(
    night.day,
    night.sleepStart,
    night.sleepEnd,
    profile.userProfileId,
  )
  const before = await readGarminSleep(session, base, profile.displayName, night.day)
  if (manualSleepMatches(before.dto, payload)) return { action: 'already present', state: before }
  if (before.dto && hasGarminSleep(before.dto) && !isManualSleep(before.dto))
    return { action: 'preserved Garmin-recorded', state: before }

  let action: SyncAction = 'created'
  if (before.dto && isManualSleep(before.dto)) {
    const id = readNumber(before.dto, 'id')
    if (id == null || !Number.isSafeInteger(id) || id <= 0)
      throw new Error(`${night.day} manual Garmin sleep omitted a valid id`)
    await fetchGarminJson(session, base, `/sleep-service/sleep/dailySleep/${id}`, undefined, {
      method: 'PUT',
      body: JSON.stringify({ ...payload, id }),
    })
    action = 'updated'
  } else {
    await fetchGarminJson(session, base, '/sleep-service/sleep/dailySleep', undefined, {
      method: 'POST',
      body: JSON.stringify(payload),
    })
  }

  const after = await readGarminSleep(session, base, profile.displayName, night.day)
  if (!manualSleepMatches(after.dto, payload))
    throw new Error(`${night.day} Garmin manual sleep verification failed`)
  return { action, state: after }
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2))
  const cache: unknown = JSON.parse(await fs.readFile(ouraCacheFile, 'utf8'))
  if (!isOuraCache(cache)) throw new Error('Oura cache has an invalid shape')
  const nights = selectNights(cache, args)
  if (!nights.length) throw new Error('no Oura nights matched the selection')

  for (const night of nights) console.log(`[oura-garmin] ${describe(night)}`)
  if (!args.write) {
    console.log(`[oura-garmin] dry run, ${nights.length} night(s) selected; pass --write to sync`)
    return
  }

  const session = await readGarminConnectSession()
  const base = cleanGarminConnectBaseUrl(
    process.env.GARMIN_CONNECT_BASE_URL?.trim() || DEFAULT_GARMIN_CONNECT_BASE,
  )
  const profile = await readGarminProfile(session, base, nights[0].day)
  const delayMs = Number(process.env.GARMIN_CONNECT_DELAY_MS ?? DEFAULT_SYNC_DELAY_MS)
  if (!Number.isFinite(delayMs) || delayMs < 0)
    throw new Error('GARMIN_CONNECT_DELAY_MS must be nonnegative')

  for (const night of nights) {
    const result = await syncNight(session, base, profile, night)
    console.log(
      `[oura-garmin] ${night.day} ${result.action}; garmin reports ${sleepStateDescription(result.state)}`,
    )
    if (delayMs > 0) await new Promise(resolve => setTimeout(resolve, delayMs))
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch(err => {
    console.error(`[oura-garmin] sync failed: ${garminErrorMessage(err)}`)
    process.exit(1)
  })
}
