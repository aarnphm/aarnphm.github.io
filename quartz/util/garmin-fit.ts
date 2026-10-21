import { Decoder, Stream, type FitMessages } from '@garmin/fitsdk'
import { inflateRawSync } from 'node:zlib'
import type {
  GarminCyclingDynamics,
  GarminFitTrainingEffect,
  GarminGearShift,
  GarminSwimData,
} from '../plugins/stores/garmin'
import {
  fitCyclingDynamics,
  fitGearShifts,
  garminFitSwim,
  garminFitTrainingEffect,
} from './garmin-fit/decode'
export { fitCyclingDynamics, fitGearShifts } from './garmin-fit/decode'
export type {
  GarminOpenWaterSwimInput,
  GarminPoolSwimInput,
  GarminPoolSwimLength,
  GarminSwimFitInput,
  GarminSwimSample,
  GarminSwimStroke,
} from './garmin-fit/swim-prepare'
export { encodeGarminSwimFit } from './garmin-fit/swim-encode'
export { validateGarminFit } from './garmin-fit/validate'
export type { GarminFitEncoding } from './garmin-fit/validate'

export type GarminActivityArchiveFile =
  | { kind: 'fit'; bytes: Uint8Array }
  | { kind: 'tcx'; bytes: Uint8Array }

interface ZipEntry {
  compressedSize: number
  compressionMethod: number
  flags: number
  localHeaderOffset: number
  name: string
  uncompressedSize: number
}

const ZIP_CENTRAL_HEADER = 0x02014b50
const ZIP_END = 0x06054b50
const ZIP_LOCAL_HEADER = 0x04034b50
const ZIP_MAX_END_SEARCH = 65_557

function requireZipRange(bytes: Uint8Array, offset: number, length: number): void {
  if (
    !Number.isInteger(offset) ||
    !Number.isInteger(length) ||
    offset < 0 ||
    length < 0 ||
    offset + length > bytes.byteLength
  )
    throw new Error('Garmin activity archive is truncated')
}

function zipEndOffset(bytes: Uint8Array, view: DataView): number {
  const start = Math.max(0, bytes.byteLength - ZIP_MAX_END_SEARCH)
  for (let offset = bytes.byteLength - 22; offset >= start; offset--)
    if (view.getUint32(offset, true) === ZIP_END) return offset
  throw new Error('Garmin activity archive has no ZIP end record')
}

function zipEntries(bytes: Uint8Array, view: DataView): ZipEntry[] {
  const end = zipEndOffset(bytes, view)
  requireZipRange(bytes, end, 22)
  const count = view.getUint16(end + 10, true)
  const centralSize = view.getUint32(end + 12, true)
  const centralOffset = view.getUint32(end + 16, true)
  requireZipRange(bytes, centralOffset, centralSize)
  const decoder = new TextDecoder()
  const entries: ZipEntry[] = []
  let offset = centralOffset
  for (let index = 0; index < count; index++) {
    requireZipRange(bytes, offset, 46)
    if (view.getUint32(offset, true) !== ZIP_CENTRAL_HEADER)
      throw new Error('Garmin activity archive has an invalid ZIP directory')
    const nameLength = view.getUint16(offset + 28, true)
    const extraLength = view.getUint16(offset + 30, true)
    const commentLength = view.getUint16(offset + 32, true)
    const entryLength = 46 + nameLength + extraLength + commentLength
    requireZipRange(bytes, offset, entryLength)
    const compressedSize = view.getUint32(offset + 20, true)
    const uncompressedSize = view.getUint32(offset + 24, true)
    const localHeaderOffset = view.getUint32(offset + 42, true)
    if (
      compressedSize === 0xffffffff ||
      uncompressedSize === 0xffffffff ||
      localHeaderOffset === 0xffffffff
    )
      throw new Error('Garmin activity archive uses unsupported ZIP64 fields')
    entries.push({
      flags: view.getUint16(offset + 8, true),
      compressionMethod: view.getUint16(offset + 10, true),
      compressedSize,
      uncompressedSize,
      localHeaderOffset,
      name: decoder.decode(bytes.subarray(offset + 46, offset + 46 + nameLength)),
    })
    offset += entryLength
  }
  return entries
}

function zipEntryBytes(
  archive: Uint8Array,
  view: DataView,
  entry: ZipEntry,
  kind: 'FIT' | 'TCX',
): Uint8Array {
  if ((entry.flags & 1) !== 0) throw new Error(`Garmin activity ${kind} file is encrypted`)
  if (entry.compressionMethod !== 0 && entry.compressionMethod !== 8)
    throw new Error(
      `Garmin activity ${kind} uses unsupported ZIP method ${entry.compressionMethod}`,
    )
  requireZipRange(archive, entry.localHeaderOffset, 30)
  if (view.getUint32(entry.localHeaderOffset, true) !== ZIP_LOCAL_HEADER)
    throw new Error(`Garmin activity archive has an invalid ${kind} entry`)
  const nameLength = view.getUint16(entry.localHeaderOffset + 26, true)
  const extraLength = view.getUint16(entry.localHeaderOffset + 28, true)
  const dataOffset = entry.localHeaderOffset + 30 + nameLength + extraLength
  requireZipRange(archive, dataOffset, entry.compressedSize)
  const compressed = archive.subarray(dataOffset, dataOffset + entry.compressedSize)
  const bytes =
    entry.compressionMethod === 0
      ? Uint8Array.from(compressed)
      : Uint8Array.from(inflateRawSync(compressed))
  if (bytes.byteLength !== entry.uncompressedSize)
    throw new Error(`Garmin activity ${kind} size does not match its ZIP entry`)
  return bytes
}

export function garminActivityFileFromArchive(archive: Uint8Array): GarminActivityArchiveFile {
  const view = new DataView(archive.buffer, archive.byteOffset, archive.byteLength)
  const entries = zipEntries(archive, view)
  const fit = entries.find(candidate => candidate.name.toLowerCase().endsWith('.fit'))
  if (fit) return { kind: 'fit', bytes: zipEntryBytes(archive, view, fit, 'FIT') }
  const tcx = entries.find(candidate => candidate.name.toLowerCase().endsWith('.tcx'))
  if (tcx) return { kind: 'tcx', bytes: zipEntryBytes(archive, view, tcx, 'TCX') }
  throw new Error('Garmin activity archive contains no FIT or TCX file')
}

export function garminFitBytesFromArchive(archive: Uint8Array): Uint8Array {
  const view = new DataView(archive.buffer, archive.byteOffset, archive.byteLength)
  const entry = zipEntries(archive, view).find(candidate =>
    candidate.name.toLowerCase().endsWith('.fit'),
  )
  if (!entry) throw new Error('Garmin activity archive contains no FIT file')
  return zipEntryBytes(archive, view, entry, 'FIT')
}

function decodeGarminMessages(bytes: Uint8Array): FitMessages {
  if (!Decoder.isFIT(Stream.fromByteArray(bytes))) throw new Error('Garmin activity is not FIT')
  if (!new Decoder(Stream.fromByteArray(bytes)).checkIntegrity())
    throw new Error('Garmin activity FIT integrity check failed')
  const decoded = new Decoder(Stream.fromByteArray(bytes)).read({
    expandSubFields: true,
    expandComponents: true,
  })
  if (decoded.errors.length > 0)
    throw new Error(decoded.errors.map(error => error.message).join('; '))
  return decoded.messages
}

export interface GarminActivityFitData {
  gearShifts: GarminGearShift[]
  cyclingDynamics: GarminCyclingDynamics
  trainingEffect: GarminFitTrainingEffect
  swim: GarminSwimData | null
}

export function decodeGarminActivityFit(bytes: Uint8Array): GarminActivityFitData {
  const messages = decodeGarminMessages(bytes)
  return {
    gearShifts: fitGearShifts(messages),
    cyclingDynamics: fitCyclingDynamics(messages),
    trainingEffect: garminFitTrainingEffect(messages),
    swim: garminFitSwim(messages),
  }
}

export function garminActivityFitFromArchive(archive: Uint8Array): GarminActivityFitData {
  return decodeGarminActivityFit(garminFitBytesFromArchive(archive))
}
