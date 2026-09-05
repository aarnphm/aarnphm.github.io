import type { DistanceSystem } from './triathlon-presentation'
import { KM_TO_MI } from './triathlon-card'
import { isRecord, type UnknownRecord } from './type-guards'

export type TriathlonMaintenancePosition = 'front' | 'rear'
export type TriathlonMaintenancePart = 'tire' | 'tube'

export interface TriathlonChainMaintenance {
  id: string
  distanceMiles: number | null
  lubricant: string
  since: string
  waxed: boolean
}

export interface TriathlonMaintenanceRange {
  start: string
  end: string | null
}

export interface TriathlonComponentMaintenance {
  component: string
  type: string
  distanceMiles: number | null
  ranges: TriathlonMaintenanceRange[]
  reason: string | null
}

export interface TriathlonServiceMaintenance {
  bike: string
  date: string
  distanceMiles: number | null
  place: string
}

export interface TriathlonWheelMaintenance {
  position: TriathlonMaintenancePosition
  part: TriathlonMaintenancePart
  type: string
  distanceMiles: number | null
  ranges: TriathlonMaintenanceRange[]
  reason: string | null
  repaired: boolean | null
}

export interface TriathlonMaintenance {
  services: TriathlonServiceMaintenance[]
  components: TriathlonComponentMaintenance[]
  chains: TriathlonChainMaintenance[]
  wheels: TriathlonWheelMaintenance[]
}

interface ParsedMaintenanceRange extends TriathlonMaintenanceRange {
  reason: string | null | undefined
  repaired: boolean | null | undefined
}

const nullableString = (record: UnknownRecord, key: string): string | null | undefined => {
  const value = record[key]
  if (value === null) return null
  return typeof value === 'string' ? value : undefined
}

const nullableDistanceMiles = (record: UnknownRecord): number | null | undefined => {
  const value = record.distance
  if (value === null) return null
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : undefined
}

export const formatTriathlonMaintenanceDistance = (
  distanceMiles: number,
  system: DistanceSystem,
): string => {
  const value = system === 'imperial' ? distanceMiles : distanceMiles / KM_TO_MI
  return `${value.toLocaleString('en-US', { maximumFractionDigits: 2 })} ${system === 'imperial' ? 'mi' : 'km'}`
}

const parseChain = (id: string, value: unknown): TriathlonChainMaintenance | null => {
  if (!isRecord(value)) return null
  const distanceMiles = nullableDistanceMiles(value)
  const lubricant = value.lubricant
  const since = value.since
  const waxed = value.waxed
  if (
    distanceMiles === undefined ||
    typeof lubricant !== 'string' ||
    typeof since !== 'string' ||
    typeof waxed !== 'boolean'
  ) {
    return null
  }
  return { id, distanceMiles, lubricant, since, waxed }
}

const mergeMaintenanceFields = (value: unknown): UnknownRecord | null => {
  if (!Array.isArray(value)) return null
  const record: UnknownRecord = {}
  for (const field of value) {
    if (!isRecord(field)) return null
    for (const [key, fieldValue] of Object.entries(field)) record[key] = fieldValue
  }
  return record
}

const parseRange = (value: unknown): ParsedMaintenanceRange | null => {
  if (!isRecord(value)) return null
  const start = value.start
  const end = nullableString(value, 'end')
  const reason = nullableString(value, 'reason')
  const repaired = value.repaired
  if (
    typeof start !== 'string' ||
    end === undefined ||
    (value.reason !== undefined && reason === undefined) ||
    (repaired !== undefined && repaired !== null && typeof repaired !== 'boolean')
  ) {
    return null
  }
  return {
    start,
    end,
    reason,
    repaired: typeof repaired === 'boolean' || repaired === null ? repaired : undefined,
  }
}

const mergeRangeFields = (values: unknown[]): UnknownRecord[] | null => {
  const records: UnknownRecord[] = []
  let record: UnknownRecord = {}
  for (const value of values) {
    if (!isRecord(value)) return null
    if (Object.hasOwn(record, 'start') && Object.hasOwn(value, 'start')) {
      records.push(record)
      record = {}
    }
    for (const [key, fieldValue] of Object.entries(value)) {
      if (Object.hasOwn(record, key)) return null
      record[key] = fieldValue
    }
  }
  if (Object.keys(record).length > 0) records.push(record)
  return records
}

const parseRanges = (record: UnknownRecord): ParsedMaintenanceRange[] | null => {
  if (record.range !== undefined) {
    if (!Array.isArray(record.range) || record.range.length === 0) return null
    const records = mergeRangeFields(record.range)
    if (!records) return null
    const ranges: ParsedMaintenanceRange[] = []
    for (const value of records) {
      const range = parseRange(value)
      if (!range) return null
      ranges.push(range)
    }
    return ranges.sort((left, right) => left.start.localeCompare(right.start))
  }

  const range = parseRange(record)
  return range ? [range] : null
}

const lastDefined = <T>(values: Array<T | undefined>): T | undefined => {
  for (let index = values.length - 1; index >= 0; index -= 1) {
    const value = values[index]
    if (value !== undefined) return value
  }
  return undefined
}

const parseRangeMetadata = (
  record: UnknownRecord,
  ranges: ParsedMaintenanceRange[],
): { reason: string | null | undefined; repaired: boolean | null | undefined } | null => {
  const recordReason = nullableString(record, 'reason')
  const recordRepaired = record.repaired
  if (
    (record.reason !== undefined && recordReason === undefined) ||
    (recordRepaired !== undefined && recordRepaired !== null && typeof recordRepaired !== 'boolean')
  ) {
    return null
  }
  const rangeReason = lastDefined(ranges.map(range => range.reason))
  const rangeRepaired = lastDefined(ranges.map(range => range.repaired))
  return {
    reason: rangeReason !== undefined ? rangeReason : recordReason,
    repaired: rangeRepaired !== undefined ? rangeRepaired : recordRepaired,
  }
}

const maintenanceRanges = (ranges: ParsedMaintenanceRange[]): TriathlonMaintenanceRange[] =>
  ranges.map(({ start, end }) => ({ start, end }))

const sortByCurrentUse = <T extends { ranges: TriathlonMaintenanceRange[] }>(entries: T[]): T[] =>
  entries.sort((left, right) => {
    const leftCurrent = left.ranges.some(range => range.end === null)
    const rightCurrent = right.ranges.some(range => range.end === null)
    if (leftCurrent && !rightCurrent) return -1
    if (!leftCurrent && rightCurrent) return 1
    const leftLatest = left.ranges[left.ranges.length - 1]
    const rightLatest = right.ranges[right.ranges.length - 1]
    return rightLatest.start.localeCompare(leftLatest.start)
  })

const parseComponentEntry = (
  component: string,
  value: unknown,
): TriathlonComponentMaintenance | null => {
  const record = mergeMaintenanceFields(value)
  if (!record) return null
  const type = record.type
  const distanceMiles = nullableDistanceMiles(record)
  const ranges = parseRanges(record)
  if (typeof type !== 'string' || distanceMiles === undefined || !ranges) return null
  const metadata = parseRangeMetadata(record, ranges)
  if (!metadata) return null
  return {
    component,
    type,
    distanceMiles,
    ranges: maintenanceRanges(ranges),
    reason: metadata.reason ?? null,
  }
}

const isReservedMaintenanceSection = (section: string): boolean =>
  section === 'chain' || section === 'service' || section === 'tires'

const parseComponents = (value: UnknownRecord): TriathlonComponentMaintenance[] => {
  const components: TriathlonComponentMaintenance[] = []
  for (const [component, records] of Object.entries(value)) {
    if (isReservedMaintenanceSection(component) || !Array.isArray(records)) continue
    const entries: TriathlonComponentMaintenance[] = []
    for (const raw of records) {
      const entry = parseComponentEntry(component, raw)
      if (entry) entries.push(entry)
    }
    components.push(...sortByCurrentUse(entries))
  }
  return components
}

const parseServiceEntry = (bike: string, value: unknown): TriathlonServiceMaintenance | null => {
  if (!isRecord(value)) return null
  const date = value.date
  const distanceMiles = nullableDistanceMiles(value)
  const place = value.place
  if (typeof date !== 'string' || distanceMiles === undefined || typeof place !== 'string')
    return null
  return { bike, date, distanceMiles, place }
}

const parseServices = (value: unknown): TriathlonServiceMaintenance[] => {
  if (!isRecord(value)) return []
  const services: TriathlonServiceMaintenance[] = []
  for (const [bike, records] of Object.entries(value)) {
    if (!Array.isArray(records)) continue
    for (const raw of records) {
      const service = parseServiceEntry(bike, raw)
      if (service) services.push(service)
    }
  }
  return services.sort((left, right) => right.date.localeCompare(left.date))
}

const parseWheelEntry = (
  position: TriathlonMaintenancePosition,
  part: TriathlonMaintenancePart,
  value: unknown,
): TriathlonWheelMaintenance | null => {
  const record = mergeMaintenanceFields(value)
  if (!record) return null
  const type = record.type
  const distanceMiles = nullableDistanceMiles(record)
  const ranges = parseRanges(record)
  if (typeof type !== 'string' || distanceMiles === undefined || !ranges) return null
  const metadata = parseRangeMetadata(record, ranges)
  if (!metadata || metadata.reason === undefined) return null
  return {
    position,
    part,
    type,
    distanceMiles,
    ranges: maintenanceRanges(ranges),
    reason: metadata.reason,
    repaired: typeof metadata.repaired === 'boolean' ? metadata.repaired : null,
  }
}

const parseChains = (value: unknown): TriathlonChainMaintenance[] => {
  if (!isRecord(value)) return []
  const entries: TriathlonChainMaintenance[] = []
  for (const [id, raw] of Object.entries(value)) {
    const entry = parseChain(id, raw)
    if (entry) entries.push(entry)
  }
  return entries.sort((left, right) => right.since.localeCompare(left.since))
}

const positions: TriathlonMaintenancePosition[] = ['front', 'rear']
const parts: ReadonlyArray<{ key: string; part: TriathlonMaintenancePart }> = [
  { key: 'tires', part: 'tire' },
  { key: 'tube', part: 'tube' },
]

const parseWheels = (value: unknown): TriathlonWheelMaintenance[] => {
  if (!isRecord(value)) return []
  const entries: TriathlonWheelMaintenance[] = []
  for (const position of positions) {
    const wheel = value[position]
    if (!isRecord(wheel)) continue
    for (const { key, part } of parts) {
      const records = wheel[key]
      if (!Array.isArray(records)) continue
      for (const raw of records) {
        const entry = parseWheelEntry(position, part, raw)
        if (entry) entries.push(entry)
      }
    }
  }
  return sortByCurrentUse(entries)
}

export const parseTriathlonMaintenance = (value: unknown): TriathlonMaintenance | null => {
  if (!isRecord(value)) return null
  const services = parseServices(value.service)
  const components = parseComponents(value)
  const chains = parseChains(value.chain)
  const wheels = parseWheels(value.tires)
  return services.length > 0 || components.length > 0 || chains.length > 0 || wheels.length > 0
    ? { services, components, chains, wheels }
    : null
}
