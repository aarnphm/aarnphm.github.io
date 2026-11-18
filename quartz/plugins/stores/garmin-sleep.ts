export interface GarminSleepRespirationSample {
  /** UTC Unix timestamp in milliseconds. */
  timestamp: number
  breathsPerMinute: number | null
}

export interface GarminSleepStageInterval {
  /** UTC Unix timestamps in milliseconds. */
  startTime: number
  endTime: number
  stage: 'deep' | 'light' | 'rem' | 'awake' | 'unmeasurable'
}

export type GarminSleepScoreFactorName =
  | 'totalDuration'
  | 'stress'
  | 'awakeCount'
  | 'restlessness'
  | 'lightPercentage'
  | 'deepPercentage'
  | 'remPercentage'

export interface GarminSleepScoreFactor {
  qualifier: string | null
  /** Native percentage for stage factors, not a contribution score. */
  value: number | null
}

export interface GarminOvernightSleep {
  totalSleepS: number
  deepS: number | null
  lightS: number | null
  remS: number | null
  awakeS: number | null
  unmeasurableS: number | null
  awakeCount: number | null
  sleepScore: number | null
  sleepScoreQualifier: string | null
  averageHeartRateBpm: number | null
  restingHeartRateBpm: number | null
  averageHrvMs: number | null
  skinTempDeviationC: number | null
  scoreFactors?: Partial<Record<GarminSleepScoreFactorName, GarminSleepScoreFactor>>
  stages?: GarminSleepStageInterval[]
  /** UTC Unix timestamps in milliseconds, at Garmin's native cadence. */
  heartRate?: { timestamp: number; bpm: number | null }[]
  hrv?: { timestamp: number; ms: number | null }[]
}

export interface GarminSleepNap {
  /** Stable local key from the provider's nap window. */
  id: string
  /** UTC Unix timestamps in milliseconds. */
  startTime: number
  endTime: number
  startUtcOffsetMinutes?: number
  endUtcOffsetMinutes?: number
  durationS: number
  providerSource: number | null
  feedback: string | null
}

export interface GarminSleepSummary {
  source: 'garmin'
  date: string
  startTime: string | null
  endTime: string | null
  utcOffsetMinutes?: number
  averageBreathsPerMinute: number | null
  lowestBreathsPerMinute: number | null
  highestBreathsPerMinute: number | null
  averageSpO2: number | null
  lowestSpO2: number | null
  bodyBatteryStart: number | null
  bodyBatteryEnd: number | null
  bodyBatteryChange: number | null
  averageStress: number | null
  restlessMoments: number | null
  respiration?: GarminSleepRespirationSample[]
  sleep?: GarminOvernightSleep
  naps?: GarminSleepNap[]
}

export interface GarminSleepDateRange {
  start: string
  end: string
}
