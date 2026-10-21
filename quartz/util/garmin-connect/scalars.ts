import { readNumber, readString, type UnknownRecord } from '../type-guards'

function numeric(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (typeof value !== 'string') return null
  const parsed = Number(value.replace(/,/g, '').trim())
  return Number.isFinite(parsed) ? parsed : null
}

export function finite(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : numeric(value)
}

export function positive(value: number | null): number | null {
  return value != null && Number.isFinite(value) && value > 0 ? value : null
}

export function rounded(value: number | null): number | null {
  const n = positive(value)
  return n == null ? null : Math.round(n)
}

export function roundedFloat(value: number | null, dp: number): number | null {
  const n = positive(value)
  if (n == null) return null
  const factor = 10 ** dp
  return Math.round(n * factor) / factor
}

export function firstNumber(
  records: readonly UnknownRecord[],
  keys: readonly string[],
): number | null {
  for (const record of records) {
    for (const key of keys) {
      const value = readNumber(record, key) ?? numeric(record[key])
      if (value != null) return value
    }
  }
  return null
}

export function firstString(
  records: readonly UnknownRecord[],
  keys: readonly string[],
): string | null {
  for (const record of records) {
    for (const key of keys) {
      const value = readString(record, key)
      if (value?.trim()) return value.trim()
      const n = readNumber(record, key)
      if (n != null) return String(n)
    }
  }
  return null
}

export function normalizeDate(value: string | number | Date | null): string | null {
  if (!value) return null
  if (value instanceof Date) return Number.isFinite(value.valueOf()) ? value.toISOString() : null
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return null
    const ms = value > 1_000_000_000_000 ? value : value * 1000
    return new Date(ms).toISOString()
  }
  const trimmed = value.trim().replace(' ', 'T')
  if (!trimmed) return null
  const zoned = /(?:Z|[+-]\d{2}:?\d{2})$/.test(trimmed) ? trimmed : `${trimmed}Z`
  const ms = Date.parse(zoned)
  return Number.isFinite(ms) ? new Date(ms).toISOString() : null
}
