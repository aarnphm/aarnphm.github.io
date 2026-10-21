import { Decoder, Stream } from '@garmin/fitsdk'

export interface GarminFitMessageCounts {
  fileIds: number
  deviceInfos: number
  events: number
  records: number
  lengths: number
  laps: number
  sessions: number
  activities: number
}

export interface GarminFitValidation {
  valid: boolean
  isFit: boolean
  integrity: boolean
  errors: readonly string[]
  counts: GarminFitMessageCounts
}

export interface GarminFitEncoding {
  bytes: Uint8Array
  validation: GarminFitValidation
}

function emptyCounts(): GarminFitMessageCounts {
  return {
    fileIds: 0,
    deviceInfos: 0,
    events: 0,
    records: 0,
    lengths: 0,
    laps: 0,
    sessions: 0,
    activities: 0,
  }
}

export function validateGarminFit(bytes: Uint8Array): GarminFitValidation {
  const errors: string[] = []
  let isFit = false
  let integrity = false
  let counts = emptyCounts()
  try {
    isFit = Decoder.isFIT(Stream.fromByteArray(bytes))
  } catch (error) {
    errors.push(error instanceof Error ? error.message : String(error))
  }
  if (!isFit) errors.push('input is not a FIT file')
  if (isFit) {
    try {
      integrity = new Decoder(Stream.fromByteArray(bytes)).checkIntegrity()
    } catch (error) {
      errors.push(error instanceof Error ? error.message : String(error))
    }
    if (!integrity) errors.push('FIT integrity check failed')
    try {
      const decoded = new Decoder(Stream.fromByteArray(bytes)).read()
      errors.push(...decoded.errors.map(error => error.message))
      counts = {
        fileIds: decoded.messages.fileIdMesgs?.length ?? 0,
        deviceInfos: decoded.messages.deviceInfoMesgs?.length ?? 0,
        events: decoded.messages.eventMesgs?.length ?? 0,
        records: decoded.messages.recordMesgs?.length ?? 0,
        lengths: decoded.messages.lengthMesgs?.length ?? 0,
        laps: decoded.messages.lapMesgs?.length ?? 0,
        sessions: decoded.messages.sessionMesgs?.length ?? 0,
        activities: decoded.messages.activityMesgs?.length ?? 0,
      }
    } catch (error) {
      errors.push(error instanceof Error ? error.message : String(error))
    }
  }
  const complete =
    counts.fileIds === 1 &&
    counts.deviceInfos >= 1 &&
    counts.events >= 2 &&
    counts.records >= 1 &&
    counts.laps >= 1 &&
    counts.sessions >= 1 &&
    counts.activities === 1
  if (isFit && integrity && !complete) errors.push('FIT activity messages are incomplete')
  return {
    valid: isFit && integrity && complete && errors.length === 0,
    isFit,
    integrity,
    errors,
    counts,
  }
}
