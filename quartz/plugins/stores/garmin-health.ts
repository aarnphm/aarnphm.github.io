export interface GarminHealthReading<T> {
  status: 'available' | 'unavailable' | 'error'
  fetchedAt: number | null
  attemptedAt: number
  value: T | null
}

export interface GarminBodyBattery {
  date: string
  charged: number | null
  drained: number | null
  utcOffsetMinutes: number | null
  samples: { timestamp: number; value: number | null }[]
}

export interface GarminReadinessFactor {
  name:
    | 'sleep score'
    | 'recovery time'
    | 'acute load'
    | 'stress history'
    | 'HRV status'
    | 'sleep history'
  percent: number | null
  feedback: string | null
}

export interface GarminTrainingReadiness {
  date: string
  timestamp: number | null
  timestampLocal: string | null
  score: number | null
  level: string | null
  feedback: string | null
  inputContext: string | null
  recoveryTimeMinutes: number | null
  recoveryTimeChange: string | null
  acuteLoad: number | null
  hrvWeeklyAverage: number | null
  factors: GarminReadinessFactor[]
}

export interface GarminLoadFocus {
  date: string
  feedback: string | null
  categories: {
    name: 'low aerobic' | 'high aerobic' | 'anaerobic'
    load: number | null
    targetMin: number | null
    targetMax: number | null
  }[]
}

export interface GarminTrainingStatus {
  date: string
  timestamp: number | null
  status: number | string | null
  feedback: string | null
  sinceDate: string | null
  acuteLoad: number | null
  chronicLoad: number | null
  loadRatio: number | null
  loadStatus: string | null
  targetMin: number | null
  targetMax: number | null
  loadFocus: GarminLoadFocus | null
}

export interface GarminEnduranceScore {
  date: string
  score: number
  classification: number | null
  thresholds: { label: string; lower: number }[]
  contributors: { activityTypeId: number | null; group: number | null; contribution: number }[]
}

export interface GarminHillScore {
  date: string
  score: number
  strength: number | null
  endurance: number | null
  classification: number | null
}

export interface GarminHealthDay {
  source: 'garmin'
  date: string
  bodyBattery: GarminHealthReading<GarminBodyBattery>
  trainingReadiness: GarminHealthReading<GarminTrainingReadiness[]>
  trainingStatus: GarminHealthReading<GarminTrainingStatus>
  enduranceScore: GarminHealthReading<GarminEnduranceScore>
  hillScore: GarminHealthReading<GarminHillScore>
}

export const latestGarminReadiness = (
  day: GarminHealthDay | null | undefined,
): GarminTrainingReadiness | null => day?.trainingReadiness.value?.at(-1) ?? null

export const morningGarminReadiness = (
  day: GarminHealthDay | null | undefined,
): GarminTrainingReadiness | null =>
  day?.trainingReadiness.value?.findLast(row => row.inputContext === 'AFTER_WAKEUP_RESET') ?? null
