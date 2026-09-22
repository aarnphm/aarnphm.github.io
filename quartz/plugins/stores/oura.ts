export interface OuraDaily {
  date: string
  readiness: number | null
  sleepScore: number | null
  hrv: number | null
  rhr: number | null
  sleepDurationS: number | null
  tempDeviationC: number | null
  totalCalories: number | null
  activeCalories: number | null
}

export interface OuraSeries {
  startTs: string
  intervalS: number
  items: (number | null)[]
}

export type OuraHeartRateSource = 'awake' | 'workout' | 'rest' | 'sleep' | 'live' | 'session'

export interface OuraHeartRateSample {
  timestamp: string
  bpm: number
  source: OuraHeartRateSource
}

export interface OuraSleepDetail {
  date: string
  bedtimeStart: string | null
  bedtimeEnd: string | null
  phase5Min: string | null
  phase30Sec?: string | null
  movement30Sec?: string | null
  lowBatteryAlert?: boolean
  sleepAlgorithmVersion?: string | null
  efficiency: number | null
  latencyS: number | null
  timeInBedS: number | null
  totalSleepS: number | null
  deepS: number | null
  lightS: number | null
  remS: number | null
  awakeS: number | null
  avgBreath: number | null
  avgHr: number | null
  avgHrv: number | null
  lowestHr: number | null
  restlessPeriods: number | null
  hrv: OuraSeries | null
  hr: OuraSeries | null
  readinessScore: number | null
  readinessContrib: Record<string, number | null> | null
  sleepScore: number | null
  sleepContrib: Record<string, number | null> | null
}

export interface OuraNap extends OuraSleepDetail {
  id: string
  type: 'sleep' | 'late_nap'
  reportedDay: string | null
  sleepScoreDelta: number | null
  readinessScoreDelta: number | null
}

export interface OuraDayDetail extends OuraSleepDetail {
  // Absent in caches that predate nap ingestion; an empty array means no recorded naps.
  naps?: OuraNap[]
  health?: OuraHealthDay
}

export interface OuraHealthDay {
  date: string
  failedCollections?: string[]
  stress: { stressS: number | null; restoredS: number | null; summary: string | null } | null
  resilience: {
    level: string | null
    sleepRecovery: number | null
    daytimeRecovery: number | null
    stress: number | null
  } | null
  spo2: { averagePct: number | null; breathingDisturbanceIndex: number | null } | null
  activity: {
    score: number | null
    steps: number | null
    activeCalories: number | null
    targetCalories: number | null
    equivalentWalkingDistanceM: number | null
    targetDistanceM: number | null
    inactivityAlerts: number | null
    averageMet: number | null
    restS: number | null
    sedentaryS: number | null
    lowS: number | null
    mediumS: number | null
    highS: number | null
    nonWearS: number | null
    contributors: Record<string, number | null> | null
  } | null
  cardiovascular: { vascularAge: number | null; pulseWaveVelocity: number | null } | null
  vo2Max: number | null
  sleepTime: {
    startOffsetS: number | null
    endOffsetS: number | null
    utcOffsetS: number | null
    recommendation: string | null
    status: string | null
  } | null
  temperatureTrendC: number | null
}

export interface OuraAuth {
  refreshToken: string
  obtainedAt: number
}

export interface OuraUser {
  id: string | null
  email: string | null
}

export interface OuraCache {
  version?: number
  auth?: OuraAuth
  user?: OuraUser
  lastSync: number
  days: Record<string, OuraDaily>
  details?: Record<string, OuraDayDetail>
  heartRate?: OuraHeartRateSample[]
}

export interface OuraSleepDateFields {
  day?: unknown
  bedtime_end?: unknown
}

const localDatePattern = /^\d{4}-\d{2}-\d{2}T/

export function ouraSleepCalendarDay(row: OuraSleepDateFields): string | null {
  if (typeof row.bedtime_end === 'string' && localDatePattern.test(row.bedtime_end))
    return row.bedtime_end.slice(0, 10)
  return typeof row.day === 'string' ? row.day : null
}

export function emptyOuraDaily(date: string): OuraDaily {
  return {
    date,
    readiness: null,
    sleepScore: null,
    hrv: null,
    rhr: null,
    sleepDurationS: null,
    tempDeviationC: null,
    totalCalories: null,
    activeCalories: null,
  }
}
