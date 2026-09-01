import type { RawStravaActivity, Sport, StravaStreams } from '../plugins/stores/strava'

export interface TimedStravaStreams {
  time: number[]
  latlng: [number, number][]
  altitude: number[]
  distance: number[]
  heartrate: number[]
  cadence: number[]
  watts: number[]
  temp: number[]
}

export function timedStravaStreamsFromCache(streams: StravaStreams): TimedStravaStreams {
  return {
    time: streams.time ?? [],
    latlng: streams.latlng,
    altitude: streams.altitude,
    distance: streams.distance,
    heartrate: streams.heartrate ?? [],
    cadence: streams.cadence ?? [],
    watts: streams.watts ?? [],
    temp: [],
  }
}

function xml(value: string | number): string {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;')
}

function pointNumber(values: number[], index: number, sampleCount: number): number | null {
  if (values.length !== sampleCount) return null
  const value = values[index]
  return value != null && Number.isFinite(value) && value > 0 ? value : null
}

function pointDistance(values: number[], index: number, sampleCount: number): number | null {
  if (values.length !== sampleCount) return null
  const value = values[index]
  return value != null && Number.isFinite(value) && value >= 0 ? value : null
}

function pointTime(startMs: number, elapsedS: number): string {
  return new Date(startMs + Math.round(elapsedS * 1000)).toISOString()
}

function average(values: number[]): number | null {
  const positives = values.filter(value => value > 0 && Number.isFinite(value))
  if (positives.length === 0) return null
  return positives.reduce((sum, value) => sum + value, 0) / positives.length
}

function maximum(values: number[]): number | null {
  const positives = values.filter(value => value > 0 && Number.isFinite(value))
  if (positives.length === 0) return null
  return Math.max(...positives)
}

function integer(value: number | null): number | null {
  return value == null || !Number.isFinite(value) ? null : Math.round(value)
}

function byte(value: number | null): number | null {
  if (value == null) return null
  return Math.max(0, Math.min(254, Math.round(value)))
}

function element(name: string, value: string | number | null): string {
  return value == null ? '' : `<${name}>${xml(value)}</${name}>`
}

function heartRateElement(value: number | null): string {
  return value == null ? '' : `<HeartRateBpm><Value>${Math.round(value)}</Value></HeartRateBpm>`
}

function trackpoint(
  activity: RawStravaActivity,
  streams: TimedStravaStreams,
  sport: Sport,
  index: number,
  startMs: number,
): string {
  const sampleCount = streams.time.length
  const time = streams.time[index]
  const latlng = streams.latlng.length === sampleCount ? streams.latlng[index] : undefined
  const altitude = pointNumber(streams.altitude, index, sampleCount)
  const distance = pointDistance(streams.distance, index, sampleCount)
  const heartrate = pointNumber(streams.heartrate, index, sampleCount)
  const cadence = byte(pointNumber(streams.cadence, index, sampleCount))
  const watts = integer(pointNumber(streams.watts, index, sampleCount))
  const temp = integer(pointNumber(streams.temp, index, sampleCount))
  const parts = [`<Time>${pointTime(startMs, time)}</Time>`]
  if (latlng)
    parts.push(
      `<Position><LatitudeDegrees>${latlng[0]}</LatitudeDegrees><LongitudeDegrees>${latlng[1]}</LongitudeDegrees></Position>`,
    )
  parts.push(element('AltitudeMeters', altitude == null ? null : altitude.toFixed(1)))
  parts.push(element('DistanceMeters', distance == null ? null : distance.toFixed(1)))
  parts.push(heartRateElement(heartrate))
  if (cadence != null && sport !== 'swim') parts.push(element('Cadence', cadence))
  const tpx = [
    watts == null ? '' : `<ns3:Watts>${watts}</ns3:Watts>`,
    cadence == null || sport !== 'run' ? '' : `<ns3:RunCadence>${cadence}</ns3:RunCadence>`,
    temp == null ? '' : `<ns3:Temp>${temp}</ns3:Temp>`,
  ].filter(Boolean)
  if (tpx.length > 0) parts.push(`<Extensions><ns3:TPX>${tpx.join('')}</ns3:TPX></Extensions>`)
  if (parts.length <= 1) throw new Error(`no TCX samples available for ${activity.id}`)
  return `<Trackpoint>${parts.join('')}</Trackpoint>`
}

function tcxSport(sport: Sport): string {
  if (sport === 'bike') return 'Biking'
  if (sport === 'run') return 'Running'
  return 'Other'
}

function creatorName(sport: Sport): string {
  if (sport === 'bike') return 'Strava Bike Backfill'
  if (sport === 'run') return 'Strava Run Backfill'
  return 'Strava Swim Backfill'
}

function validateTimes(activity: RawStravaActivity, times: readonly number[]): void {
  if (times.length < 2) throw new Error(`Strava activity ${activity.id} has no timed stream`)
  let previous = -1
  for (const time of times) {
    if (!Number.isFinite(time) || time < 0 || time < previous)
      throw new Error(`Strava activity ${activity.id} has an invalid timed stream`)
    previous = time
  }
}

export function stravaActivityTcx(
  activity: RawStravaActivity,
  streams: TimedStravaStreams,
  sport: Sport,
): string {
  validateTimes(activity, streams.time)
  const startMs = Date.parse(activity.startDate)
  if (!Number.isFinite(startMs))
    throw new Error(`Strava activity ${activity.id} has invalid startDate`)
  if (!Number.isFinite(activity.distance) || activity.distance < 0)
    throw new Error(`Strava activity ${activity.id} has invalid distance`)
  const lastElapsed = streams.time[streams.time.length - 1]
  const totalTimeS = Math.max(lastElapsed, activity.elapsedTime || activity.movingTime)
  if (!Number.isFinite(totalTimeS) || totalTimeS <= 0)
    throw new Error(`Strava activity ${activity.id} has invalid duration`)
  const avgHr = integer(average(streams.heartrate))
  const maxHr = integer(maximum(streams.heartrate))
  const calories = integer(activity.calories ?? null) ?? 0
  const points = streams.time
    .map((_, index) => trackpoint(activity, streams, sport, index, startMs))
    .join('')
  return [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<TrainingCenterDatabase xmlns="http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2" xmlns:ns3="http://www.garmin.com/xmlschemas/ActivityExtension/v2" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2 http://www.garmin.com/xmlschemas/TrainingCenterDatabasev2.xsd">',
    '<Activities>',
    `<Activity Sport="${tcxSport(sport)}">`,
    `<Id>${new Date(startMs).toISOString()}</Id>`,
    `<Lap StartTime="${new Date(startMs).toISOString()}">`,
    element('TotalTimeSeconds', totalTimeS.toFixed(1)),
    element('DistanceMeters', activity.distance.toFixed(1)),
    element('Calories', calories),
    '<Intensity>Active</Intensity>',
    heartRateElement(avgHr),
    maxHr == null ? '' : `<MaximumHeartRateBpm><Value>${maxHr}</Value></MaximumHeartRateBpm>`,
    '<TriggerMethod>Manual</TriggerMethod>',
    `<Track>${points}</Track>`,
    '</Lap>',
    `<Notes>${xml(`Strava ${activity.id}: ${activity.name}`)}</Notes>`,
    `<Creator xsi:type="Device_t"><Name>${creatorName(sport)}</Name><UnitId>0</UnitId><ProductID>0</ProductID></Creator>`,
    '</Activity>',
    '</Activities>',
    '</TrainingCenterDatabase>',
  ].join('')
}
