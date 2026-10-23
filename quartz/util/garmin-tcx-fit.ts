import {
  Encoder,
  Profile,
  type ActivityMesg,
  type DeviceInfoMesg,
  type EventMesg,
  type FileIdMesg,
  type SessionMesg,
} from '@garmin/fitsdk'
import { validateGarminFit, type GarminFitEncoding } from './garmin-fit'
import {
  activityDistance,
  activityTimerSeconds,
  applyStatistics,
  gpsBounds,
  lapMessage,
  recordMessage,
  statistics,
  uint16,
} from './garmin-tcx-fit/fit-messages'
import { parseTcxActivity } from './garmin-tcx-fit/tcx-activity'

function hashSourceId(sourceId: string | number): number {
  const text = String(sourceId)
  let hash = 2_166_136_261
  for (let index = 0; index < text.length; index++) {
    hash ^= text.charCodeAt(index)
    hash = Math.imul(hash, 16_777_619)
  }
  return hash >>> 0 || 1
}

export function encodeGarminTcxActivityFit(
  tcx: Uint8Array,
  sourceId: string | number,
): GarminFitEncoding {
  const activity = parseTcxActivity(tcx)
  const elapsedSeconds = (activity.endTime.getTime() - activity.startTime.getTime()) / 1000
  const timerSeconds = activityTimerSeconds(activity)
  const distance = activityDistance(activity)
  const values = statistics(activity.trackpoints)
  const averageSpeed = distance != null && timerSeconds > 0 ? distance / timerSeconds : null
  const calories = activity.laps.map(lap => lap.calories)
  const serialNumber = hashSourceId(sourceId)
  const encoder = new Encoder()
  const fileId: FileIdMesg = {
    type: 'activity',
    manufacturer: 'development',
    product: 0,
    serialNumber,
    timeCreated: activity.startTime,
  }
  const deviceInfo: DeviceInfoMesg = {
    timestamp: activity.startTime,
    deviceIndex: 'creator',
    manufacturer: 'development',
    product: 0,
    serialNumber,
    productName: 'Garmin TCX Bridge',
    softwareVersion: 1,
  }
  const timerStart: EventMesg = {
    timestamp: activity.startTime,
    event: 'timer',
    eventType: 'start',
  }
  encoder.onMesg(Profile.MesgNum.FILE_ID, fileId)
  encoder.onMesg(Profile.MesgNum.DEVICE_INFO, deviceInfo)
  encoder.onMesg(Profile.MesgNum.EVENT, timerStart)
  for (let index = 0; index < activity.trackpoints.length; index++)
    encoder.onMesg(Profile.MesgNum.RECORD, recordMessage(activity.trackpoints, index))
  for (const [index, lap] of activity.laps.entries())
    encoder.onMesg(
      Profile.MesgNum.LAP,
      lapMessage(lap, index, activity.laps.length, activity.sport),
    )
  const timerStop: EventMesg = { timestamp: activity.endTime, event: 'timer', eventType: 'stop' }
  const session: SessionMesg = {
    messageIndex: 0,
    timestamp: activity.endTime,
    event: 'session',
    eventType: 'stop',
    startTime: activity.startTime,
    sport: activity.sport,
    totalElapsedTime: elapsedSeconds,
    totalTimerTime: timerSeconds,
    totalDistance: distance ?? undefined,
    totalCalories: calories.some(value => value != null)
      ? uint16(calories.reduce<number>((sum, value) => sum + (value ?? 0), 0))
      : undefined,
    avgSpeed: averageSpeed ?? undefined,
    enhancedAvgSpeed: averageSpeed ?? undefined,
    maxSpeed: values.maximumSpeed ?? undefined,
    enhancedMaxSpeed: values.maximumSpeed ?? undefined,
    firstLapIndex: 0,
    numLaps: activity.laps.length,
    trigger: 'activityEnd',
    totalMovingTime: timerSeconds,
  }
  const bounds = gpsBounds(activity.trackpoints)
  if (bounds) {
    session.startPositionLat = bounds.firstLatitude
    session.startPositionLong = bounds.firstLongitude
    session.endPositionLat = bounds.lastLatitude
    session.endPositionLong = bounds.lastLongitude
    session.necLat = bounds.northeastLatitude
    session.necLong = bounds.northeastLongitude
    session.swcLat = bounds.southwestLatitude
    session.swcLong = bounds.southwestLongitude
  }
  applyStatistics(session, values)
  const activityMessage: ActivityMesg = {
    timestamp: activity.endTime,
    totalTimerTime: timerSeconds,
    numSessions: 1,
    type: 'manual',
    event: 'activity',
    eventType: 'stop',
  }
  encoder.onMesg(Profile.MesgNum.EVENT, timerStop)
  encoder.onMesg(Profile.MesgNum.SESSION, session)
  encoder.onMesg(Profile.MesgNum.ACTIVITY, activityMessage)
  const bytes = encoder.close()
  const validation = validateGarminFit(bytes)
  if (!validation.valid)
    throw new Error(`Garmin TCX conversion produced invalid FIT: ${validation.errors.join('; ')}`)
  return { bytes, validation }
}
