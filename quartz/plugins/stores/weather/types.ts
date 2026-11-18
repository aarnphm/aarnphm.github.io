import { type GardenUvCalibrationArtifact } from '../../../util/activity-uv-score'
import { type SurfaceCurrentEstimate } from '../../../util/surface-current'

export interface WeatherActivity {
  activityId: string
  date: string
  start: string
  end: string
  latitude: number
  longitude: number
  durationS: number
  windKph: number | null
  windDir: string | null
  windDirDeg: number | null
  windGustKph: number | null
  averageRelativeHumidityPct: number | null
  relativeHumidityProvenance: WeatherRelativeHumidityProvenance | null
  temperatureC: number | null
  temperatureSeries?: WeatherTemperatureSample[]
  routeFingerprint?: string
  fetchedAt?: number
  routeHours?: WeatherRouteHour[]
  surfaceCurrent?: SurfaceCurrentEstimate
  source: 'weatherkit'
}

export interface WeatherRouteHour {
  forecastStart: string
  overlapStart: string
  overlapEnd: string
  elapsedStartS: number
  elapsedEndS: number
  latitude: number
  longitude: number
  uvIndex: number | null
  cloudCover: number | null
  temperatureC: number | null
  windSpeedKph: number | null
  windDirectionDeg: number | null
  windGustKph: number | null
  relativeHumidity: number | null
  pressureHpa: number | null
  daylight: boolean | null
}

export interface WeatherAttribution {
  serviceName: string
  logoLightUrl: string
  logoDarkUrl: string
  legalPageUrl: string
}

export interface WeatherRelativeHumidityProvenance {
  source: 'weatherkit'
  sourceKind: 'modeled'
  samplingMethod: 'route-hour'
  inputTimestamp: string
  coveragePct: number
}

export interface WeatherTemperatureSample {
  elapsedS: number
  temperatureC: number
}

export interface WeatherDay {
  date: string
  activityCount: number
  durationS: number
  windKph: number | null
  windDir: string | null
  windDirDeg: number | null
  windGustKph: number | null
}

export interface WeatherCache {
  version?: number
  lastSync: number
  current: WeatherSnapshot | null
  attribution: WeatherAttribution | null
  uvCalibration: GardenUvCalibrationArtifact | null
  activities: Record<string, WeatherActivity>
  days: Record<string, WeatherDay>
}

export interface WeatherHour {
  forecastStart: string
  windSpeed: number | null
  windDirection: number | null
  windGust: number | null
  relativeHumidity: number | null
  temperature: number | null
  uvIndex: number | null
  cloudCover: number | null
  pressure: number | null
  daylight: boolean | null
  conditionCode: string | null
  precipitationChance: number | null
  precipitationType: string | null
}

export interface WeatherSnapshot {
  forecastStart: string
  latitude: number
  longitude: number
  temperatureC: number | null
  conditionCode: string | null
  precipitationChance: number | null
  precipitationType: string | null
  source: 'weatherkit'
}

export interface WeatherActivityCandidate {
  activityId: string
  date: string
  start: string
  end: string
  latitude: number
  longitude: number
  durationS: number
  routeFingerprint?: string
}
