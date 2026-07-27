import fs from 'node:fs/promises'
import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import type { OuraCache, OuraDayDetail, OuraSeries } from '../plugins/stores/oura'
import {
  applyGarminSetCookies,
  cleanGarminConnectBaseUrl,
  DEFAULT_GARMIN_CONNECT_BASE,
  fetchGarminJson,
  garminConnectRequestHeaders,
  garminErrorMessage,
  garminResponseSummary,
  garminUrlFor,
  readGarminConnectSession,
  type GarminConnectSession,
} from '../util/garmin-session'
import {
  encodeGarminWellnessFit,
  type GarminSleepLevel,
  type GarminSleepStage,
  type GarminWellnessFitEncoding,
  type GarminWellnessHeartRateSample,
} from '../util/garmin-wellness-fit'
import { joinSegments, QUARTZ } from '../util/path'
import { isRecord } from '../util/type-guards'

const DEFAULT_SERIAL = 3_141_592_653
const PRODUCT_NAME = 'Oura Sleep Bridge'
const PHASE_LEVELS: Record<string, GarminSleepLevel> = {
  '1': 'deep',
  '2': 'light',
  '3': 'rem',
  '4': 'awake',
}
const PHASE_SECONDS = 300
const DEFAULT_UPLOAD_DELAY_MS = 1200
const ouraCacheFile = joinSegments(QUARTZ, '.quartz-cache', 'oura.json')
const outDir = joinSegments(QUARTZ, '.quartz-cache', 'oura-garmin')

interface Args {
  days: string[]
  count: number
  since: string | null
  serial: number
  write: boolean
}

interface Night {
  day: string
  sleepStart: Date
  sleepEnd: Date
  localOffsetMinutes: number
  stages: GarminSleepStage[]
  heartRate: GarminWellnessHeartRateSample[]
  restingHeartRate: number | null
}

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
  const args: Args = { days: [], count: 1, since: null, serial: DEFAULT_SERIAL, write: false }
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
    } else if (arg === '--serial') {
      const serial = Number(readArgValue(argv, ++i, arg))
      if (!Number.isSafeInteger(serial) || serial <= 0)
        throw new Error('--serial must be a positive integer')
      args.serial = serial
    } else throw new Error(`unknown flag ${arg}`)
  }
  return args
}

export function isoOffsetMinutes(value: string): number {
  const match = /(Z|[+-]\d{2}:\d{2})$/.exec(value.trim())
  if (!match) throw new Error(`${value} has no UTC offset`)
  if (match[1] === 'Z') return 0
  const sign = match[1].startsWith('-') ? -1 : 1
  const [hours, minutes] = match[1].slice(1).split(':').map(Number)
  return sign * (hours * 60 + minutes)
}

export function sleepStages(phase5Min: string, sleepStart: Date): GarminSleepStage[] {
  const stages: GarminSleepStage[] = []
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
): GarminWellnessHeartRateSample[] {
  const startMs = Date.parse(series.startTs)
  if (!Number.isFinite(startMs)) throw new Error(`${series.startTs} is not a timestamp`)
  const out: GarminWellnessHeartRateSample[] = []
  for (let i = 0; i < series.items.length; i++) {
    const value = series.items[i]
    if (value == null || !Number.isFinite(value)) continue
    const time = new Date(startMs + i * series.intervalS * 1000)
    if (time < sleepStart || time > sleepEnd) continue
    out.push({ time, heartRateBpm: value })
  }
  return out
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
    localOffsetMinutes: isoOffsetMinutes(detail.bedtimeStart),
    stages,
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

function stageMinutes(night: Night): Record<GarminSleepLevel, number> {
  const totals: Record<GarminSleepLevel, number> = {
    unmeasurable: 0,
    awake: 0,
    light: 0,
    deep: 0,
    rem: 0,
  }
  for (let i = 0; i < night.stages.length; i++) {
    const end = night.stages[i + 1]?.startTime ?? night.sleepEnd
    totals[night.stages[i].level] += (end.valueOf() - night.stages[i].startTime.valueOf()) / 60_000
  }
  return totals
}

async function uploadWellnessFit(
  session: GarminConnectSession,
  base: string,
  filename: string,
  bytes: Uint8Array,
): Promise<{ status: number; body: string }> {
  const form = new FormData()
  const buffer = new ArrayBuffer(bytes.byteLength)
  new Uint8Array(buffer).set(bytes)
  form.set('userfile', new Blob([buffer], { type: 'application/octet-stream' }), filename)
  const headers = new Headers(garminConnectRequestHeaders(session))
  headers.set('X-Requested-With', 'XMLHttpRequest')
  headers.delete('Content-Type')
  const res = await fetch(garminUrlFor(base, '/upload-service/upload/.fit'), {
    method: 'POST',
    headers,
    body: form,
    redirect: 'manual',
  })
  applyGarminSetCookies(session, res.headers)
  const text = await res.text()
  return { status: res.status, body: garminResponseSummary(res, text) }
}

async function readGarminSleep(
  session: GarminConnectSession,
  base: string,
  day: string,
): Promise<string> {
  const profile = await fetchGarminJson(session, base, '/userprofile-service/socialProfile')
  const displayName = isRecord(profile) ? profile.displayName : null
  if (typeof displayName !== 'string') throw new Error('Garmin profile omitted displayName')
  const raw = await fetchGarminJson(
    session,
    base,
    `/wellness-service/wellness/dailySleepData/${encodeURIComponent(displayName)}`,
    new URLSearchParams({ date: day, nonSleepBufferMinutes: '60' }),
  )
  if (!isRecord(raw) || !isRecord(raw.dailySleepDTO)) return 'no sleep payload'
  const dto = raw.dailySleepDTO
  const levels = Array.isArray(raw.sleepLevels) ? raw.sleepLevels.length : 0
  return `sleepTimeSeconds=${String(dto.sleepTimeSeconds)} deep=${String(dto.deepSleepSeconds)} light=${String(dto.lightSleepSeconds)} rem=${String(dto.remSleepSeconds)} awake=${String(dto.awakeSleepSeconds)} levels=${levels}`
}

function describe(night: Night, encoding: GarminWellnessFitEncoding): string {
  const minutes = stageMinutes(night)
  const total = minutes.deep + minutes.light + minutes.rem
  return [
    `${night.day} ${night.sleepStart.toISOString()} → ${night.sleepEnd.toISOString()}`,
    `  stages ${night.stages.length} transitions, asleep ${Math.round(total)}min`,
    `  deep ${Math.round(minutes.deep)}min light ${Math.round(minutes.light)}min rem ${Math.round(minutes.rem)}min awake ${Math.round(minutes.awake)}min`,
    `  heart rate ${night.heartRate.length} samples, resting ${night.restingHeartRate ?? 'n/a'}`,
    `  fit ${encoding.bytes.byteLength} bytes, valid=${encoding.validation.valid} ${JSON.stringify(encoding.validation.counts)}`,
    encoding.validation.errors.length ? `  errors ${encoding.validation.errors.join('; ')}` : '',
  ]
    .filter(Boolean)
    .join('\n')
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2))
  const cache = JSON.parse(await fs.readFile(ouraCacheFile, 'utf8')) as OuraCache
  const nights = selectNights(cache, args)
  if (!nights.length) throw new Error('no Oura nights matched the selection')

  await fs.mkdir(outDir, { recursive: true })
  const encodings = new Map<string, { night: Night; encoding: GarminWellnessFitEncoding }>()
  for (const night of nights) {
    const encoding = encodeGarminWellnessFit({
      serialNumber: args.serial,
      productName: PRODUCT_NAME,
      sleepStart: night.sleepStart,
      sleepEnd: night.sleepEnd,
      localOffsetMinutes: night.localOffsetMinutes,
      stages: night.stages,
      heartRate: night.heartRate,
      restingHeartRate: night.restingHeartRate ?? undefined,
    })
    const file = joinSegments(outDir, `oura-${night.day}.fit`)
    await fs.writeFile(file, encoding.bytes)
    console.log(`[oura-garmin] ${describe(night, encoding)}\n  wrote ${file}`)
    if (!encoding.validation.valid) throw new Error(`${night.day} produced an invalid FIT file`)
    encodings.set(night.day, { night, encoding })
  }

  if (!args.write) {
    console.log(`[oura-garmin] dry run, ${encodings.size} night(s) encoded; pass --write to upload`)
    return
  }

  const session = await readGarminConnectSession()
  const base = cleanGarminConnectBaseUrl(
    process.env.GARMIN_CONNECT_BASE_URL?.trim() || DEFAULT_GARMIN_CONNECT_BASE,
  )
  const delayMs = Number(process.env.GARMIN_CONNECT_DELAY_MS ?? DEFAULT_UPLOAD_DELAY_MS)
  if (!Number.isFinite(delayMs) || delayMs < 0)
    throw new Error('GARMIN_CONNECT_DELAY_MS must be nonnegative')
  for (const [day, { encoding }] of encodings) {
    const result = await uploadWellnessFit(session, base, `oura-${day}.fit`, encoding.bytes)
    console.log(`[oura-garmin] ${day} upload → ${result.status} ${result.body}`)
    try {
      console.log(
        `[oura-garmin] ${day} garmin now reports ${await readGarminSleep(session, base, day)}`,
      )
    } catch (err) {
      console.warn(`[oura-garmin] ${day} verification failed: ${garminErrorMessage(err)}`)
    }
    if (delayMs > 0) await new Promise(resolve => setTimeout(resolve, delayMs))
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch(err => {
    console.error(`[oura-garmin] sync failed: ${garminErrorMessage(err)}`)
    process.exit(1)
  })
}
