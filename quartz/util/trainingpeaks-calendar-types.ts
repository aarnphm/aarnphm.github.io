export type TrainingPeaksCalendarSport = 'swim' | 'bike' | 'run' | 'strength' | 'other'

export interface TrainingPeaksCalendarMetrics {
  durationSeconds: number | null
  distanceMeters: number | null
  tss: number | null
}

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

/** Targets are percentages of the athlete threshold that the metric names; rpe targets are 0–10 exertion ratings. */
export type TrainingPeaksCalendarIntensityMetric =
  | 'percentOfFtp'
  | 'percentOfThresholdPace'
  | 'percentOfThresholdHr'
  | 'rpe'

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
