import type { ActivityDiscipline, PreparedActivity, TcxTrackPoint } from './conversion-model'

export const MAX_FILE_BYTES = 25 * 1024 * 1024
const TCX_NAMESPACE = 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2'
const ACTIVITY_EXTENSION_NAMESPACE = 'http://www.garmin.com/xmlschemas/ActivityExtension/v2'

function escapeXml(value: string): string {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&apos;')
}

function decimal(value: number, places = 3): string {
  if (!Number.isFinite(value)) throw new Error('cannot serialize a nonfinite number')
  return value.toFixed(places)
}

function integer(value: number): string {
  if (!Number.isFinite(value)) throw new Error('cannot serialize a nonfinite number')
  return String(Math.round(value))
}

function heartRateXml(tag: string, bpm: number, indent: string): string[] {
  return [`${indent}<${tag}>`, `${indent} <Value>${integer(bpm)}</Value>`, `${indent}</${tag}>`]
}

function trackPointXml(point: TcxTrackPoint, discipline: ActivityDiscipline): string {
  const lines = ['      <Trackpoint>', `       <Time>${point.iso}</Time>`]
  if (point.position) {
    lines.push(
      '       <Position>',
      `        <LatitudeDegrees>${decimal(point.position.latitude, 7)}</LatitudeDegrees>`,
      `        <LongitudeDegrees>${decimal(point.position.longitude, 7)}</LongitudeDegrees>`,
      '       </Position>',
    )
  }
  if (point.altitudeM != null)
    lines.push(`       <AltitudeMeters>${decimal(point.altitudeM, 1)}</AltitudeMeters>`)
  lines.push(`       <DistanceMeters>${decimal(point.distanceM)}</DistanceMeters>`)
  if (point.heartRateBpm != null)
    lines.push(...heartRateXml('HeartRateBpm', point.heartRateBpm, '       '))
  if ((discipline === 'bike' || discipline === 'transition') && point.cadencePerMinute != null)
    lines.push(`       <Cadence>${integer(point.cadencePerMinute)}</Cadence>`)
  const extensions: string[] = []
  if (discipline === 'run' && point.cadencePerMinute != null)
    extensions.push(`         <ns3:RunCadence>${integer(point.cadencePerMinute)}</ns3:RunCadence>`)
  if (point.speedMps != null)
    extensions.push(`         <ns3:Speed>${decimal(point.speedMps)}</ns3:Speed>`)
  if (point.powerW != null)
    extensions.push(`         <ns3:Watts>${integer(point.powerW)}</ns3:Watts>`)
  if (extensions.length > 0)
    lines.push(
      '       <Extensions>',
      '        <ns3:TPX>',
      ...extensions,
      '        </ns3:TPX>',
      '       </Extensions>',
    )
  lines.push('      </Trackpoint>')
  return lines.join('\n')
}

function lapExtensionsXml(prepared: PreparedActivity): string[] {
  const { activity, distanceM } = prepared
  const values: string[] = []
  if (activity.durationS > 0)
    values.push(`       <ns3:AvgSpeed>${decimal(distanceM / activity.durationS)}</ns3:AvgSpeed>`)
  if (activity.discipline === 'run' && activity.averageCadencePerMinute != null)
    values.push(
      `       <ns3:AvgRunCadence>${integer(activity.averageCadencePerMinute)}</ns3:AvgRunCadence>`,
    )
  if (activity.averagePowerW != null && activity.averagePowerW > 0)
    values.push(`       <ns3:AvgWatts>${integer(activity.averagePowerW)}</ns3:AvgWatts>`)
  if (values.length === 0) return []
  return ['     <Extensions>', '      <ns3:LX>', ...values, '      </ns3:LX>', '     </Extensions>']
}

export function tcxXml(prepared: PreparedActivity, parentWorkoutId: string): string {
  const { activity, sport, trackPoints, distanceM } = prepared
  const lines = [
    '<?xml version="1.0" encoding="UTF-8"?>',
    `<TrainingCenterDatabase xmlns="${TCX_NAMESPACE}" xmlns:ns3="${ACTIVITY_EXTENSION_NAMESPACE}" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="${TCX_NAMESPACE} http://www.garmin.com/xmlschemas/TrainingCenterDatabasev2.xsd ${ACTIVITY_EXTENSION_NAMESPACE} http://www.garmin.com/xmlschemas/ActivityExtensionv2.xsd">`,
    ' <Activities>',
    `  <Activity Sport="${sport}">`,
    `   <Id>${activity.start.iso}</Id>`,
    `   <Lap StartTime="${activity.start.iso}">`,
    `    <TotalTimeSeconds>${decimal(activity.durationS)}</TotalTimeSeconds>`,
    `    <DistanceMeters>${decimal(distanceM)}</DistanceMeters>`,
    `    <Calories>${integer(activity.activeEnergyKcal ?? 0)}</Calories>`,
  ]
  if (activity.averageHeartRateBpm != null && activity.averageHeartRateBpm > 0)
    lines.push(...heartRateXml('AverageHeartRateBpm', activity.averageHeartRateBpm, '    '))
  lines.push('    <Intensity>Active</Intensity>')
  if (
    (activity.discipline === 'bike' || activity.discipline === 'transition') &&
    activity.averageCadencePerMinute != null &&
    activity.averageCadencePerMinute > 0
  )
    lines.push(`    <Cadence>${integer(activity.averageCadencePerMinute)}</Cadence>`)
  lines.push(
    '    <TriggerMethod>Manual</TriggerMethod>',
    '    <Track>',
    ...trackPoints.map(point => trackPointXml(point, activity.discipline)),
    '    </Track>',
    ...lapExtensionsXml(prepared),
    '   </Lap>',
    `   <Notes>${escapeXml(`HealthExporter multisport ${parentWorkoutId}; source activity ${activity.id}`)}</Notes>`,
    '  </Activity>',
    ' </Activities>',
    '</TrainingCenterDatabase>',
    '',
  )
  const xml = lines.join('\n')
  if (/(?:NaN|Infinity)/.test(xml)) throw new Error(`${activity.discipline} TCX is nonfinite`)
  if (Buffer.byteLength(xml) >= MAX_FILE_BYTES)
    throw new Error(`${activity.discipline} TCX is 25MB or larger`)
  return xml
}
