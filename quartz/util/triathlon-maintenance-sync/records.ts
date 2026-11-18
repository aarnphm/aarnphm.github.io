import { isRecord, type UnknownRecord } from '../type-guards'

export interface UsageRange {
  start: string
  end: string | null
}

export interface DistanceRecord {
  label: string
  bikeId: string
  includeVirtual: boolean
  target: UnknownRecord
  ranges: UsageRange[]
  group?: string
  replacementWins?: boolean
  replacementDate?: string
}

export const lastDate = '9999-12-31'

export function bikeId(value: unknown): string {
  if (typeof value !== 'string' || !/^b\d+$/.test(value))
    throw new Error('Maintenance record requires a Strava bikeId (b followed by digits)')
  return value
}

export function localDate(value: unknown): string {
  if (typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(value))
    throw new Error(`Invalid maintenance date: ${String(value)}`)
  const date = new Date(`${value}T00:00:00Z`)
  if (!Number.isFinite(date.getTime()) || date.toISOString().slice(0, 10) !== value)
    throw new Error(`Invalid maintenance date: ${value}`)
  return value
}

function range(value: UnknownRecord): UsageRange {
  const start = localDate(value.start)
  const end = value.end == null ? null : localDate(value.end)
  if (end && end < start) throw new Error(`Maintenance range ends before it starts: ${start}`)
  return { start, end }
}

function ranges(value: UnknownRecord): UsageRange[] {
  if (value.range === undefined) return [range(value)]
  if (!Array.isArray(value.range)) throw new Error('Maintenance range must be a list')
  const result: UsageRange[] = []
  let current: UnknownRecord = {}
  for (const fields of value.range) {
    if (!isRecord(fields)) throw new Error('Maintenance range fields must be mappings')
    if (Object.hasOwn(fields, 'start') && Object.hasOwn(current, 'start')) {
      result.push(range(current))
      current = {}
    }
    Object.assign(current, fields)
  }
  if (Object.keys(current).length) result.push(range(current))
  return result
}

export function intersection(left: UsageRange, right: UsageRange): UsageRange | null {
  const start = left.start > right.start ? left.start : right.start
  const end = (left.end ?? lastDate) < (right.end ?? lastDate) ? left.end : right.end
  return start <= (end ?? lastDate) ? { start, end } : null
}

function components(section: string, values: unknown, replacementWins = false): DistanceRecord[] {
  if (!Array.isArray(values)) throw new Error(`${section} must be a list of component records`)
  return values.map((fields, index) => {
    if (!Array.isArray(fields) || !fields.length || !fields.every(isRecord))
      throw new Error(`${section}.${index + 1} must be a list of field mappings`)
    const value: UnknownRecord = Object.assign({}, ...fields)
    const target = fields.find(field => Object.hasOwn(field, 'distance')) ?? fields[0]
    return {
      label: `${section}.${index + 1} ${String(value.type ?? '')}`.trimEnd(),
      bikeId: bikeId(value.bikeId),
      // The Cog setup uses the drivetrain and saddle; wheels and brakes stay stationary.
      includeVirtual: ['OSPW', 'bottom bracket', 'chainrings', 'saddle'].includes(section),
      target,
      ranges: ranges(value),
      group: section,
      replacementWins,
    }
  })
}

function chains(values: unknown): DistanceRecord[] {
  if (!isRecord(values)) throw new Error('Maintenance chains must be a mapping')
  const result: DistanceRecord[] = []
  for (const [id, chain] of Object.entries(values)) {
    if (!isRecord(chain)) throw new Error(`Invalid maintenance chain ${id}`)
    const chainBikeId = bikeId(chain.bikeId)
    const since = localDate(chain.since)
    const cycles: UnknownRecord[] = []
    if (chain.cycles !== undefined) {
      if (!Array.isArray(chain.cycles) || !chain.cycles.every(isRecord))
        throw new Error(`Invalid cycles for chain ${id}`)
      cycles.push(...chain.cycles)
    }
    const intervals = cycles.map((cycle, index) =>
      range({
        start: cycle.start ?? (index === 0 ? since : cycles[index - 1].end),
        end: cycle.end,
      }),
    )
    const usage =
      chain.range !== undefined
        ? ranges(chain)
        : chain.status === 'backup'
          ? []
          : chain.status === 'active'
            ? [{ start: since, end: null }]
            : intervals.length
              ? intervals
              : [{ start: since, end: null }]
    result.push({
      label: `chain.${id}`,
      bikeId: chainBikeId,
      includeVirtual: true,
      target: chain,
      ranges: usage,
      group: 'chain',
    })
    for (const [index, cycle] of cycles.entries()) {
      result.push({
        label: `chain.${id}.cycle.${index + 1}`,
        bikeId: chainBikeId,
        includeVirtual: true,
        target: cycle,
        ranges: usage.flatMap(period => {
          const overlap = intersection(period, intervals[index])
          return overlap ? [overlap] : []
        }),
        group: `chain.${id}.cycles`,
        replacementWins: true,
        replacementDate: intervals[index].start,
      })
    }
  }
  return result
}

export function distanceRecords(maintenance: UnknownRecord): DistanceRecord[] {
  const records: DistanceRecord[] = []
  for (const [section, value] of Object.entries(maintenance)) {
    if (section === 'chain') records.push(...chains(value))
    else if (section === 'tires') {
      if (!isRecord(value)) throw new Error('Maintenance tires must be a mapping')
      for (const [position, parts] of Object.entries(value)) {
        if (!isRecord(parts)) throw new Error(`Invalid tire records at ${position}`)
        for (const [part, entries] of Object.entries(parts))
          records.push(...components(`tires.${position}.${part}`, entries, part === 'tires'))
      }
    } else if (section !== 'service' && Array.isArray(value))
      records.push(...components(section, value))
  }
  if (maintenance.service !== undefined) {
    if (!isRecord(maintenance.service)) throw new Error('Maintenance service must be a mapping')
    for (const [bike, services] of Object.entries(maintenance.service)) {
      if (services === null) continue
      if (!Array.isArray(services) || !services.every(isRecord))
        throw new Error(`Invalid maintenance services for ${bike}`)
      for (const service of services) {
        const serviceBikeId = bikeId(service.bikeId)
        const firstUsage = records
          .filter(record => record.bikeId === serviceBikeId)
          .flatMap(record => record.ranges.map(period => period.start))
          .sort()[0]
        const end = localDate(service.date)
        if (!firstUsage)
          throw new Error(`Service mileage needs an installation date for ${serviceBikeId}`)
        records.push({
          label: `service.${bike}.${end}`,
          bikeId: serviceBikeId,
          includeVirtual: true,
          target: service,
          ranges: [{ start: firstUsage, end }],
        })
      }
    }
  }
  return records
}
