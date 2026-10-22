export type MultisportDiscipline = 'swim' | 'bike' | 'run'

export type ActivityDiscipline = MultisportDiscipline | 'transition'
export type TcxSport = 'Swimming' | 'Biking' | 'Running' | 'Other'

export interface TimedValue {
  iso: string
  timeMs: number
}

export interface HeartRateSample extends TimedValue {
  bpm: number
  sourceIso: string
  sourceTimeMs: number
}

export interface HeartRateTimestampRepair {
  sourceTimestamp: string
  outputTimestamp: string
  bpm: number
  reason: string
  provenance: string
}

export interface MultisportActivityBase {
  id: string
  activity: string
  start: TimedValue
  end: TimedValue
  durationS: number
  elapsedTimeS: number
  distanceM: number | null
  activeEnergyKcal: number | null
  averageHeartRateBpm: number | null
  averagePowerW: number | null
  averageCadencePerMinute: number | null
  lapCount: number | null
}

export interface SportActivity extends MultisportActivityBase {
  kind: 'sport'
  discipline: MultisportDiscipline
}

export interface TransitionActivity extends MultisportActivityBase {
  kind: 'transition'
  discipline: 'transition'
}

export type MultisportActivity = SportActivity | TransitionActivity

export interface MultisportWorkout {
  id: string
  activity: string
  start: TimedValue
  end: TimedValue
  gpxFile: string
  heartRate: HeartRateSample[]
  heartRateTimestampRepairs: HeartRateTimestampRepair[]
  activities: MultisportActivity[]
  legs: [SportActivity, SportActivity, SportActivity]
  transitions: TransitionActivity[]
}

export interface Position {
  latitude: number
  longitude: number
}

export interface RoutePoint extends TimedValue {
  position: Position
  altitudeM: number | null
  heartRateBpm: number | null
  cadencePerMinute: number | null
  speedMps: number | null
  powerW: number | null
}

export interface CourseRoutePoint {
  position: Position
  altitudeM: number | null
  distanceM: number
}

export interface CourseRoute {
  path: string
  points: CourseRoutePoint[]
  distanceM: number
}

export interface TcxTrackPoint extends TimedValue {
  position: Position | null
  altitudeM: number | null
  distanceM: number
  heartRateBpm: number | null
  cadencePerMinute: number | null
  speedMps: number | null
  powerW: number | null
}

export interface PreparedActivity {
  sourceActivity: MultisportActivity
  activity: MultisportActivity
  sport: TcxSport
  trackPoints: TcxTrackPoint[]
  routePointCount: number
  usedHeartRateFallback: boolean
  distanceM: number
  distanceOverrideM: number | null
  routeOverridePath: string | null
}

export interface ExportAppleMultisportTcxOptions {
  inputPath: string
  workoutId: string
  outputDir: string
  sports?: readonly MultisportDiscipline[]
  swimDistanceM?: number
  swimDurationS?: number
  swimElapsedTimeS?: number
  swimRoutePath?: string
  transition1DurationS?: number
  transition1RoutePath?: string
  includeTransitions?: boolean
}

export interface ExportedTcxFile {
  discipline: ActivityDiscipline
  sport: TcxSport
  filename: string
  path: string
  sourceActivityId: string
  trackPointCount: number
  routePointCount: number
  usedHeartRateFallback: boolean
  sourceDistanceM: number | null
  outputDistanceM: number
  distanceOverrideM: number | null
  sourceStart: string
  sourceEnd: string
  outputStart: string
  outputEnd: string
  outputDurationS: number
  outputElapsedTimeS: number
  routeOverridePath: string | null
}

export interface ExportAppleMultisportTcxResult {
  workoutId: string
  outputDir: string
  manifestPath: string
  files: ExportedTcxFile[]
}

export function timedValue(value: string, key: string): TimedValue {
  const timeMs = Date.parse(value)
  if (!Number.isFinite(timeMs)) throw new Error(`${key} must be an ISO timestamp`)
  return { iso: new Date(timeMs).toISOString(), timeMs }
}

function radians(value: number): number {
  return (value * Math.PI) / 180
}

export function haversineMeters(left: Position, right: Position): number {
  const earthRadiusM = 6_371_008.8
  const latitudeDelta = radians(right.latitude - left.latitude)
  const longitudeDelta = radians(right.longitude - left.longitude)
  const leftLatitude = radians(left.latitude)
  const rightLatitude = radians(right.latitude)
  const chord =
    Math.sin(latitudeDelta / 2) ** 2 +
    Math.cos(leftLatitude) * Math.cos(rightLatitude) * Math.sin(longitudeDelta / 2) ** 2
  return earthRadiusM * 2 * Math.atan2(Math.sqrt(chord), Math.sqrt(Math.max(0, 1 - chord)))
}
