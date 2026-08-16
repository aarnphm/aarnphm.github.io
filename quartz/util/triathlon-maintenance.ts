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

export interface TriathlonWheelMaintenance {
  position: TriathlonMaintenancePosition
  part: TriathlonMaintenancePart
  type: string
  distance: string | null
  start: string
  end: string | null
  reason: string | null
}

export interface TriathlonMaintenance {
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

const parseWheelEntry = (
  position: TriathlonMaintenancePosition,
  part: TriathlonMaintenancePart,
  value: unknown,
): TriathlonWheelMaintenance | null => {
  const record = mergeMaintenanceFields(value)
  if (!record) return null
  const type = record.type
  const distance = nullableString(record, 'distance')
  const start = record.start
  const end = nullableString(record, 'end')
  const reason = nullableString(record, 'reason')
  if (
    typeof type !== 'string' ||
    distance === undefined ||
    typeof start !== 'string' ||
    end === undefined ||
    reason === undefined
  ) {
    return null
  }
  return { position, part, type, distance, start, end, reason }
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
  return entries.sort((left, right) => {
    if (left.end === null && right.end !== null) return -1
    if (left.end !== null && right.end === null) return 1
    return right.start.localeCompare(left.start)
  })
}

export const parseTriathlonMaintenance = (value: unknown): TriathlonMaintenance | null => {
  if (!isRecord(value)) return null
  const chains = parseChains(value.chain)
  const wheels = parseWheels(value.tires)
  return chains.length > 0 || wheels.length > 0 ? { chains, wheels } : null
}
