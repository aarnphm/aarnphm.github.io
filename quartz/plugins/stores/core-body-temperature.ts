import type { RawStravaActivity } from './strava'
import { isRecord, numberValue, stringValue } from '../../util/type-guards'

const MIN_CORE_TEMPERATURE_C = 25
const MAX_CORE_TEMPERATURE_C = 45
const MIN_SKIN_TEMPERATURE_C = 0
const MAX_SKIN_TEMPERATURE_C = 50
const MIN_HEAT_STRAIN_INDEX = 0
const MAX_HEAT_STRAIN_INDEX = 20

export interface CoreBodyTemperatureSample {
  time: string
  coreTemperatureC: number | null
  skinTemperatureC: number | null
  heatStrainIndex: number | null
  quality: number | null
  heartRate: number | null
}

export interface CoreBodyTemperatureActivitySample extends CoreBodyTemperatureSample {
  elapsedS: number
}

export interface CoreBodyTemperatureCache {
  version?: number
  lastSync: number
  samples: CoreBodyTemperatureSample[]
}

const normalizedHeader = (value: string): string =>
  value
    .trim()
    .toLowerCase()
    .replaceAll('°', '')
    .replace(/[^a-z0-9]+/g, '')

function delimiterFor(input: string): string {
  const firstLine = input.split(/\r?\n/, 1)[0] ?? ''
  const counts = new Map<string, number>([
    [',', 0],
    [';', 0],
    ['\t', 0],
  ])
  let quoted = false
  for (let index = 0; index < firstLine.length; index++) {
    const character = firstLine[index]
    if (character === '"') {
      if (quoted && firstLine[index + 1] === '"') index++
      else quoted = !quoted
      continue
    }
    if (!quoted && counts.has(character)) counts.set(character, (counts.get(character) ?? 0) + 1)
  }
  return [...counts].sort((left, right) => right[1] - left[1])[0]?.[0] ?? ','
}

function csvRows(input: string): string[][] {
  const delimiter = delimiterFor(input)
  const rows: string[][] = []
  let row: string[] = []
  let value = ''
  let quoted = false
  for (let index = 0; index < input.length; index++) {
    const character = input[index]
    if (quoted) {
      if (character === '"' && input[index + 1] === '"') {
        value += '"'
        index++
      } else if (character === '"') {
        quoted = false
      } else {
        value += character
      }
      continue
    }
    if (character === '"') {
      quoted = true
    } else if (character === delimiter) {
      row.push(value.trim())
      value = ''
    } else if (character === '\n') {
      row.push(value.trim())
      if (row.some(cell => cell.length > 0)) rows.push(row)
      row = []
      value = ''
    } else if (character !== '\r') {
      value += character
    }
  }
  row.push(value.trim())
  if (row.some(cell => cell.length > 0)) rows.push(row)
  return rows
}

function columnIndex(headers: string[], aliases: readonly string[]): number {
  return headers.findIndex(header => aliases.includes(header))
}

function metric(row: string[], index: number, min: number, max: number): number | null {
  if (index < 0) return null
  const raw = row[index]?.trim().replace(',', '.')
  if (!raw) return null
  const value = Number(raw)
  return Number.isFinite(value) && value >= min && value <= max ? value : null
}

function qualityValue(row: string[], index: number): number | null {
  const value = metric(row, index, 0, 255)
  if (value == null) return null
  const quality = Math.trunc(value) & 15
  return quality >= 1 && quality <= 4 ? quality : null
}

function localDateTime(value: string): string {
  const european = value.match(
    /^(\d{2})[./](\d{2})[./](\d{4})[ T](\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)$/,
  )
  if (!european) return value
  return `${european[3]}-${european[2]}-${european[1]}T${european[4]}`
}

function timestamp(raw: string, assumeUtc: boolean): string | null {
  const value = raw.trim()
  if (!value) return null
  const numeric = Number(value)
  let milliseconds: number
  if (Number.isFinite(numeric) && numeric > 1_000_000_000) {
    milliseconds = numeric > 1_000_000_000_000 ? numeric : numeric * 1_000
  } else {
    let normalized = localDateTime(value).replace(/^(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2})/, '$1T$2')
    if (assumeUtc && /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?$/.test(normalized))
      normalized += 'Z'
    milliseconds = Date.parse(normalized)
  }
  return Number.isFinite(milliseconds) ? new Date(milliseconds).toISOString() : null
}

function sampleFromRecord(value: unknown): CoreBodyTemperatureSample | null {
  if (!isRecord(value)) return null
  const time = stringValue(value.time)
  if (!time || !Number.isFinite(Date.parse(time))) return null
  const numberOrNull = (raw: unknown): number | null => numberValue(raw) ?? null
  const sample = {
    time: new Date(time).toISOString(),
    coreTemperatureC: numberOrNull(value.coreTemperatureC),
    skinTemperatureC: numberOrNull(value.skinTemperatureC),
    heatStrainIndex: numberOrNull(value.heatStrainIndex),
    quality: numberOrNull(value.quality),
    heartRate: numberOrNull(value.heartRate),
  }
  if (
    sample.coreTemperatureC == null &&
    sample.skinTemperatureC == null &&
    sample.heatStrainIndex == null &&
    sample.heartRate == null
  )
    return null
  return sample
}

export function parseCoreBodyTemperatureApiSamples(raw: unknown): CoreBodyTemperatureSample[] {
  if (!Array.isArray(raw)) return []
  const samples: CoreBodyTemperatureSample[] = []
  for (const value of raw) {
    if (!isRecord(value)) continue
    const rawTime = stringValue(value.timeUtc)
    if (!rawTime) continue
    const time = timestamp(rawTime, true)
    if (!time) continue
    const bounded = (rawMetric: unknown, min: number, max: number): number | null => {
      const metricValue = numberValue(rawMetric)
      return metricValue != null && metricValue >= min && metricValue <= max ? metricValue : null
    }
    const rawQuality = numberValue(value.quality)
    const quality = rawQuality == null ? null : Math.trunc(rawQuality) & 15
    const sample = {
      time,
      coreTemperatureC: bounded(value.coreTemp, MIN_CORE_TEMPERATURE_C, MAX_CORE_TEMPERATURE_C),
      skinTemperatureC: bounded(value.skinTemp, MIN_SKIN_TEMPERATURE_C, MAX_SKIN_TEMPERATURE_C),
      heatStrainIndex: bounded(value.heatStrainIndex, MIN_HEAT_STRAIN_INDEX, MAX_HEAT_STRAIN_INDEX),
      quality: quality != null && quality >= 1 && quality <= 4 ? quality : null,
      heartRate: bounded(value.heartrate, 1, 260),
    }
    if (
      sample.coreTemperatureC != null ||
      sample.skinTemperatureC != null ||
      sample.heatStrainIndex != null ||
      sample.heartRate != null
    )
      samples.push(sample)
  }
  return mergeCoreBodyTemperatureSamples([], samples)
}

export function parseCoreBodyTemperatureCsv(input: string): CoreBodyTemperatureSample[] {
  const rows = csvRows(input)
  const rawHeaders = rows.shift()
  if (!rawHeaders) return []
  const headers = rawHeaders.map(normalizedHeader)
  const timestampColumn = columnIndex(headers, [
    'timestamp',
    'timestamputc',
    'datetime',
    'datetimeutc',
    'dateandtime',
    'dateandtimeutc',
    'sampletime',
    'sampletimeutc',
  ])
  const dateColumn = columnIndex(headers, ['date', 'sampledate'])
  const timeColumn = columnIndex(headers, ['time', 'sampleclocktime'])
  const coreColumn = columnIndex(headers, [
    'coretemperature',
    'coretemperaturec',
    'corebodytemperature',
    'corebodytemperaturec',
    'coretemp',
    'coretempc',
  ])
  const skinColumn = columnIndex(headers, [
    'skintemperature',
    'skintemperaturec',
    'skintemp',
    'skintempc',
  ])
  const heatStrainColumn = columnIndex(headers, [
    'heatstrainindex',
    'heatstrain',
    'hsi',
    'thermalstrainindex',
  ])
  const qualityColumn = columnIndex(headers, [
    'tempquality',
    'temperaturequality',
    'coredataquality',
    'quality',
  ])
  const heartRateColumn = columnIndex(headers, ['heartrate', 'heartratebpm', 'hr', 'bpm'])
  const timestampHeader = rawHeaders[timestampColumn] ?? ''
  const assumeUtc = /\b(?:utc|gmt)\b/i.test(timestampHeader)
  const samples: CoreBodyTemperatureSample[] = []
  for (const row of rows) {
    const rawTime =
      timestampColumn >= 0
        ? (row[timestampColumn] ?? '')
        : dateColumn >= 0 && timeColumn >= 0
          ? `${row[dateColumn] ?? ''}T${row[timeColumn] ?? ''}`
          : ''
    const time = timestamp(rawTime, assumeUtc)
    if (!time) continue
    const sample = {
      time,
      coreTemperatureC: metric(row, coreColumn, MIN_CORE_TEMPERATURE_C, MAX_CORE_TEMPERATURE_C),
      skinTemperatureC: metric(row, skinColumn, MIN_SKIN_TEMPERATURE_C, MAX_SKIN_TEMPERATURE_C),
      heatStrainIndex: metric(row, heatStrainColumn, MIN_HEAT_STRAIN_INDEX, MAX_HEAT_STRAIN_INDEX),
      quality: qualityValue(row, qualityColumn),
      heartRate: metric(row, heartRateColumn, 1, 260),
    }
    if (
      sample.coreTemperatureC != null ||
      sample.skinTemperatureC != null ||
      sample.heatStrainIndex != null ||
      sample.heartRate != null
    )
      samples.push(sample)
  }
  return mergeCoreBodyTemperatureSamples([], samples)
}

export function mergeCoreBodyTemperatureSamples(
  current: readonly CoreBodyTemperatureSample[],
  incoming: readonly CoreBodyTemperatureSample[],
): CoreBodyTemperatureSample[] {
  const merged = new Map<string, CoreBodyTemperatureSample>()
  for (const sample of [...current, ...incoming]) {
    const previous = merged.get(sample.time)
    merged.set(
      sample.time,
      previous
        ? {
            time: sample.time,
            coreTemperatureC: sample.coreTemperatureC ?? previous.coreTemperatureC,
            skinTemperatureC: sample.skinTemperatureC ?? previous.skinTemperatureC,
            heatStrainIndex: sample.heatStrainIndex ?? previous.heatStrainIndex,
            quality: sample.quality ?? previous.quality,
            heartRate: sample.heartRate ?? previous.heartRate,
          }
        : sample,
    )
  }
  return [...merged.values()].sort((left, right) => left.time.localeCompare(right.time))
}

export function parseCoreBodyTemperatureCache(raw: unknown): CoreBodyTemperatureCache | null {
  if (!isRecord(raw)) return null
  const lastSync = numberValue(raw.lastSync)
  if (lastSync == null || !Array.isArray(raw.samples)) return null
  const samples = raw.samples
    .map(sampleFromRecord)
    .filter((sample): sample is CoreBodyTemperatureSample => sample != null)
    .sort((left, right) => left.time.localeCompare(right.time))
  return { version: numberValue(raw.version), lastSync, samples }
}

export function isUsableCoreTemperatureSample(sample: CoreBodyTemperatureSample): boolean {
  return (
    sample.coreTemperatureC != null &&
    sample.coreTemperatureC >= MIN_CORE_TEMPERATURE_C &&
    sample.coreTemperatureC <= MAX_CORE_TEMPERATURE_C &&
    (sample.quality == null || sample.quality >= 2)
  )
}

export function matchCoreBodyTemperatureActivity(
  activity: RawStravaActivity,
  cache: CoreBodyTemperatureCache | null | undefined,
): CoreBodyTemperatureActivitySample[] {
  const durationS = Math.max(
    1,
    activity.elapsedTime > 0 ? activity.elapsedTime : activity.movingTime,
  )
  return coreBodyTemperatureSamplesForWindow(activity.startDate, durationS, cache)
}

export function coreBodyTemperatureSamplesForWindow(
  start: string,
  durationS: number,
  cache: CoreBodyTemperatureCache | null | undefined,
): CoreBodyTemperatureActivitySample[] {
  if (!cache || !Number.isFinite(durationS) || durationS <= 0) return []
  const startMs = Date.parse(start)
  if (!Number.isFinite(startMs)) return []
  const endMs = startMs + durationS * 1_000
  const samples: CoreBodyTemperatureActivitySample[] = []
  for (const sample of cache.samples) {
    const timeMs = Date.parse(sample.time)
    if (!Number.isFinite(timeMs) || timeMs < startMs || timeMs > endMs) continue
    samples.push({ ...sample, elapsedS: (timeMs - startMs) / 1_000 })
  }
  return samples.sort((left, right) => left.elapsedS - right.elapsedS)
}
