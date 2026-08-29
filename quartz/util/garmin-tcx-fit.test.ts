import { Decoder, Stream, type FitMessages } from '@garmin/fitsdk'
import assert from 'node:assert/strict'
import test from 'node:test'
import { encodeGarminTcxActivityFit } from './garmin-tcx-fit'

function tcxBytes(xml = runningTcx()): Uint8Array {
  return new TextEncoder().encode(xml)
}

function decode(bytes: Uint8Array): FitMessages {
  const result = new Decoder(Stream.fromByteArray(bytes)).read()
  assert.deepEqual(result.errors, [])
  return result.messages
}

function runningTcx(): string {
  return `<?xml version="1.0" encoding="UTF-8"?>
<TrainingCenterDatabase xmlns="http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2" xmlns:ns3="http://www.garmin.com/xmlschemas/ActivityExtension/v2">
  <Activities>
    <Activity Sport="Running">
      <Id>2026-07-19T12:00:00.000Z</Id>
      <Lap StartTime="2026-07-19T12:00:00.000Z">
        <TotalTimeSeconds>60</TotalTimeSeconds>
        <DistanceMeters>200</DistanceMeters>
        <Calories>50</Calories>
        <Track>
          <Trackpoint>
            <Time>2026-07-19T12:00:00.000Z</Time>
            <Position><LatitudeDegrees>43.6532</LatitudeDegrees><LongitudeDegrees>-79.3832</LongitudeDegrees></Position>
            <AltitudeMeters>75</AltitudeMeters>
            <DistanceMeters>0</DistanceMeters>
            <HeartRateBpm><Value>120</Value></HeartRateBpm>
            <Extensions><ns3:TPX><ns3:RunCadence>81</ns3:RunCadence><ns3:Watts>240</ns3:Watts><ns3:Temp>18</ns3:Temp></ns3:TPX></Extensions>
          </Trackpoint>
          <Trackpoint>
            <Time>2026-07-19T12:01:00.000Z</Time>
            <Position><LatitudeDegrees>43.6540</LatitudeDegrees><LongitudeDegrees>-79.3824</LongitudeDegrees></Position>
            <AltitudeMeters>78</AltitudeMeters>
            <DistanceMeters>200</DistanceMeters>
            <HeartRateBpm><Value>145</Value></HeartRateBpm>
            <Cadence>82</Cadence>
            <Extensions><ns3:TPX><ns3:Speed>3.3</ns3:Speed><ns3:Watts>260</ns3:Watts><ns3:Temp>19</ns3:Temp></ns3:TPX></Extensions>
          </Trackpoint>
        </Track>
      </Lap>
      <Lap StartTime="2026-07-19T12:01:00.000Z">
        <TotalTimeSeconds>60</TotalTimeSeconds>
        <DistanceMeters>210</DistanceMeters>
        <Calories>55</Calories>
        <Track>
          <Trackpoint>
            <Time>2026-07-19T12:01:00.000Z</Time>
            <Position><LatitudeDegrees>43.6540</LatitudeDegrees><LongitudeDegrees>-79.3824</LongitudeDegrees></Position>
            <AltitudeMeters>78</AltitudeMeters>
            <DistanceMeters>200</DistanceMeters>
            <HeartRateBpm><Value>145</Value></HeartRateBpm>
            <Cadence>82</Cadence>
            <Extensions><ns3:TPX><ns3:Speed>3.4</ns3:Speed><ns3:Watts>260</ns3:Watts><ns3:Temp>19</ns3:Temp></ns3:TPX></Extensions>
          </Trackpoint>
          <Trackpoint>
            <Time>2026-07-19T12:02:00.000Z</Time>
            <Position><LatitudeDegrees>43.6548</LatitudeDegrees><LongitudeDegrees>-79.3816</LongitudeDegrees></Position>
            <AltitudeMeters>76</AltitudeMeters>
            <DistanceMeters>410</DistanceMeters>
            <HeartRateBpm><Value>150</Value></HeartRateBpm>
            <Cadence>84</Cadence>
            <Extensions><ns3:TPX><ns3:Watts>280</ns3:Watts><ns3:Temp>20</ns3:Temp></ns3:TPX></Extensions>
          </Trackpoint>
        </Track>
      </Lap>
    </Activity>
  </Activities>
</TrainingCenterDatabase>`
}

test('converts a Garmin TCX run into a complete FIT activity', () => {
  const encoded = encodeGarminTcxActivityFit(tcxBytes(), '23516096233')
  const messages = decode(encoded.bytes)
  const records = messages.recordMesgs ?? []
  const laps = messages.lapMesgs ?? []
  const session = messages.sessionMesgs?.[0]

  assert.equal(encoded.validation.valid, true)
  assert.deepEqual(encoded.validation.counts, {
    fileIds: 1,
    deviceInfos: 1,
    events: 2,
    records: 4,
    lengths: 0,
    laps: 2,
    sessions: 1,
    activities: 1,
  })
  assert.equal(records[0].positionLat, Math.round((43.6532 * 2 ** 31) / 180))
  assert.equal(records[0].positionLong, Math.round((-79.3832 * 2 ** 31) / 180))
  assert.equal(records[0].heartRate, 120)
  assert.equal(records[0].cadence, 81)
  assert.equal(records[0].power, 240)
  assert.equal(records[0].temperature, 18)
  assert.equal(records[1].enhancedSpeed, 3.3)
  assert.equal(records[3].enhancedSpeed, 3.5)
  assert.deepEqual(
    laps.map(lap => lap.lapTrigger),
    ['manual', 'sessionEnd'],
  )
  assert.equal(session?.sport, 'running')
  assert.equal(session?.totalElapsedTime, 120)
  assert.equal(session?.totalTimerTime, 120)
  assert.equal(session?.totalDistance, 410)
  assert.equal(session?.totalCalories, 105)
  assert.equal(session?.numLaps, 2)
  assert.equal(session?.avgRunningCadence, 82)
  assert.equal(session?.maxRunningCadence, 84)
  assert.equal(session?.avgPower, 260)
  assert.equal(session?.maxPower, 280)
  assert.equal(session?.totalAscent, 3)
  assert.equal(session?.totalDescent, 2)
})

test('rejects TCX documents with a doctype', () => {
  assert.throws(
    () =>
      encodeGarminTcxActivityFit(
        tcxBytes(`<!DOCTYPE TrainingCenterDatabase>${runningTcx()}`),
        'activity',
      ),
    /must not contain a doctype/,
  )
})

test('keeps unavailable TCX summaries absent from FIT', () => {
  const xml = runningTcx()
    .replace(/\s*<DistanceMeters>[^<]+<\/DistanceMeters>/g, '')
    .replace(/\s*<Calories>[^<]+<\/Calories>/g, '')
    .replace(/<ns3:Speed>[^<]+<\/ns3:Speed>/g, '')
  const session = decode(encodeGarminTcxActivityFit(tcxBytes(xml), 'activity').bytes)
    .sessionMesgs?.[0]

  assert.equal(session?.totalDistance, undefined)
  assert.equal(session?.totalCalories, undefined)
  assert.equal(session?.avgSpeed, undefined)
  assert.equal(session?.maxSpeed, undefined)
})

test('rejects TCX documents with multiple activities', () => {
  const second = '<Activity Sport="Running"><Id>2026-07-20T12:00:00Z</Id></Activity>'
  const xml = runningTcx().replace('</Activities>', `${second}</Activities>`)

  assert.throws(
    () => encodeGarminTcxActivityFit(tcxBytes(xml), 'activity'),
    /must contain exactly one activity, found 2/,
  )
})
