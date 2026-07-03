import { type UnknownRecord, isRecord, readNumber, readString } from './type-guards'

export const PACE_SCHEMA_VERSION = 2

export const PACE_SPORTS = ['swim', 'bike', 'run'] as const
export type PaceSport = (typeof PACE_SPORTS)[number]

export const PACE_FEATURE_NAMES = [
  'sport_swim',
  'sport_bike',
  'sport_run',
  'distance_km',
  'elevation_m',
  'temp_c',
  'wind_kph',
  'ctl',
  'atl',
  'tsb',
  'sport_ctl',
  'hrv',
  'rhr',
  'readiness',
  'sleep_s',
  'temp_dev_c',
  'weight_kg',
  'vthr',
  'hr_max',
  'effort',
] as const
export type PaceFeatureName = (typeof PACE_FEATURE_NAMES)[number]

export const PACE_FEATURE_DIM = PACE_FEATURE_NAMES.length
export const PACE_INPUT_DIM = PACE_FEATURE_DIM * 2

export const RACE_EFFORT_FRAC: Record<PaceSport, number> = { swim: 0.88, bike: 0.9, run: 0.92 }

export interface PaceDayState {
  date: string
  ctl: number
  atl: number
  tsb: number
  swimCtl: number
  bikeCtl: number
  runCtl: number
  hrv: number | null
  rhr: number | null
  readiness: number | null
  sleepDurationS: number | null
  tempDeviationC: number | null
  weightKg: number | null
}

export interface PaceLegSpec {
  sport: PaceSport
  distanceKm: number
  elevationM: number
  tempC: number | null
  windKph: number | null
}

export interface PaceContext {
  vThrBySport: Record<PaceSport, number>
  hrMax: number | null
  effortFrac?: number | null
}

export interface PaceFeatureVector {
  raw: Float32Array
  presence: Float32Array
}

export function isPaceSport(value: unknown): value is PaceSport {
  return value === 'swim' || value === 'bike' || value === 'run'
}

function sportCtl(day: PaceDayState, sport: PaceSport): number {
  if (sport === 'swim') return day.swimCtl
  if (sport === 'bike') return day.bikeCtl
  return day.runCtl
}

export function buildFeatureVector(
  day: PaceDayState,
  spec: PaceLegSpec,
  ctx: PaceContext,
): PaceFeatureVector {
  const raw = new Float32Array(PACE_FEATURE_DIM)
  const presence = new Float32Array(PACE_FEATURE_DIM).fill(1)
  const put = (i: number, v: number | null): void => {
    if (v == null || !Number.isFinite(v)) {
      presence[i] = 0
      return
    }
    raw[i] = v
  }
  raw[0] = spec.sport === 'swim' ? 1 : 0
  raw[1] = spec.sport === 'bike' ? 1 : 0
  raw[2] = spec.sport === 'run' ? 1 : 0
  raw[3] = spec.distanceKm
  raw[4] = spec.elevationM
  put(5, spec.tempC)
  put(6, spec.windKph)
  raw[7] = day.ctl
  raw[8] = day.atl
  raw[9] = day.tsb
  raw[10] = sportCtl(day, spec.sport)
  put(11, day.hrv)
  put(12, day.rhr)
  put(13, day.readiness)
  put(14, day.sleepDurationS)
  put(15, day.tempDeviationC)
  put(16, day.weightKg)
  raw[17] = ctx.vThrBySport[spec.sport]
  put(18, ctx.hrMax)
  put(19, ctx.effortFrac ?? RACE_EFFORT_FRAC[spec.sport])
  return { raw, presence }
}

export function dayStateFromFeedRow(row: unknown): PaceDayState | null {
  if (!isRecord(row)) return null
  const num = (k: string): number => readNumber(row, k) ?? 0
  const nul = (k: string): number | null => readNumber(row, k) ?? null
  return {
    date: readString(row, 'date') ?? '',
    ctl: num('ctl'),
    atl: num('atl'),
    tsb: num('tsb'),
    swimCtl: num('swimCtl'),
    bikeCtl: num('bikeCtl'),
    runCtl: num('runCtl'),
    hrv: nul('hrv'),
    rhr: nul('rhr'),
    readiness: nul('readiness'),
    sleepDurationS: nul('sleepDurationS'),
    tempDeviationC: nul('tempDeviationC'),
    weightKg: nul('weightKg'),
  }
}

export function legSpecFromActivityRow(row: unknown): PaceLegSpec | null {
  if (!isRecord(row)) return null
  const sport = readString(row, 'sport')
  if (!isPaceSport(sport)) return null
  return {
    sport,
    distanceKm: readNumber(row, 'distanceKm') ?? 0,
    elevationM: readNumber(row, 'elevationM') ?? 0,
    tempC: readNumber(row, 'avgTemp') ?? null,
    windKph: readNumber(row, 'windKph') ?? null,
  }
}

export function contextFromMetaRow(row: unknown): PaceContext {
  const vThrBySport: Record<PaceSport, number> = { swim: 0, bike: 0, run: 0 }
  let hrMax: number | null = null
  if (isRecord(row)) {
    const thresholds = row['thresholds']
    if (Array.isArray(thresholds)) {
      for (const t of thresholds) {
        if (!isRecord(t)) continue
        const sport = readString(t, 'sport')
        const vThr = readNumber(t, 'vThr')
        if (isPaceSport(sport) && vThr != null) vThrBySport[sport] = vThr
      }
    }
    const athlete = row['athlete']
    if (isRecord(athlete)) hrMax = readNumber(athlete, 'hrMaxEst') ?? null
  }
  return { vThrBySport, hrMax }
}

export interface PaceFeed {
  meta: UnknownRecord | null
  days: UnknownRecord[]
  activities: UnknownRecord[]
}

export function parsePaceFeed(text: string): PaceFeed {
  const days: UnknownRecord[] = []
  const activities: UnknownRecord[] = []
  let meta: UnknownRecord | null = null
  for (const line of text.split('\n')) {
    if (!line) continue
    let row: unknown
    try {
      row = JSON.parse(line)
    } catch {
      continue
    }
    if (!isRecord(row)) continue
    const kind = readString(row, 'kind')
    if (kind === 'meta') meta = row
    else if (kind === 'day') days.push(row)
    else if (kind === 'activity') activities.push(row)
  }
  return { meta, days, activities }
}
