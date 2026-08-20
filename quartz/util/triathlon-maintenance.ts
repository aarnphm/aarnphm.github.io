import { isRecord, type UnknownRecord } from './type-guards'

export type TriathlonMaintenancePosition = 'front' | 'rear'
export type TriathlonMaintenancePart = 'tire' | 'tube'

export interface TriathlonChainMaintenance {
  id: string
  distance: string | null
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
  distance: string | null
  ranges: TriathlonMaintenanceRange[]
  reason: string | null
}

export interface TriathlonServiceMaintenance {
  bike: string
  date: string
  distance: string | null
  place: string
}

export interface TriathlonWheelMaintenance {
  position: TriathlonMaintenancePosition
  part: TriathlonMaintenancePart
  type: string
  distance: string | null
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

const nullableString = (record: UnknownRecord, key: string): string | null | undefined => {
  const value = record[key]
  if (value === null) return null
  return typeof value === 'string' ? value : undefined
}

const parseChain = (id: string, value: unknown): TriathlonChainMaintenance | null => {
  if (!isRecord(value)) return null
  const distance = nullableString(value, 'distance')
  const lubricant = value.lubricant
  const since = value.since
  const waxed = value.waxed
  if (
    distance === undefined ||
    typeof lubricant !== 'string' ||
    typeof since !== 'string' ||
    typeof waxed !== 'boolean'
  ) {
    return null
  }
  return { id, distance, lubricant, since, waxed }
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

const parseRange = (value: unknown): TriathlonMaintenanceRange | null => {
  if (!isRecord(value)) return null
  const start = value.start
  const end = nullableString(value, 'end')
  if (typeof start !== 'string' || end === undefined) return null
  return { start, end }
}

const parseRanges = (record: UnknownRecord): TriathlonMaintenanceRange[] | null => {
  if (record.range !== undefined) {
    if (!Array.isArray(record.range) || record.range.length === 0) return null
    const ranges: TriathlonMaintenanceRange[] = []
    for (const value of record.range) {
      const range = parseRange(value)
      if (!range) return null
      ranges.push(range)
    }
    return ranges.sort((left, right) => left.start.localeCompare(right.start))
  }

  const range = parseRange(record)
  return range ? [range] : null
}

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
  const distance = nullableString(record, 'distance')
  const ranges = parseRanges(record)
  const reason = record.reason
  if (
    typeof type !== 'string' ||
    distance === undefined ||
    !ranges ||
    (reason !== undefined && reason !== null && typeof reason !== 'string')
  ) {
    return null
  }
  return { component, type, distance, ranges, reason: typeof reason === 'string' ? reason : null }
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
  const distance = nullableString(value, 'distance')
  const place = value.place
  if (typeof date !== 'string' || distance === undefined || typeof place !== 'string') return null
  return { bike, date, distance, place }
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
  const distance = nullableString(record, 'distance')
  const ranges = parseRanges(record)
  const reason = nullableString(record, 'reason')
  const repaired = record.repaired
  if (
    typeof type !== 'string' ||
    distance === undefined ||
    !ranges ||
    reason === undefined ||
    (repaired !== undefined && repaired !== null && typeof repaired !== 'boolean')
  ) {
    return null
  }
  return {
    position,
    part,
    type,
    distance,
    ranges,
    reason,
    repaired: typeof repaired === 'boolean' ? repaired : null,
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
