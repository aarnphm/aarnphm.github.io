export type UnknownRecord = Record<string, unknown>

export type JsonPrimitive = string | number | boolean | null
export type JsonValue = JsonPrimitive | JsonObject | JsonValue[]
export type JsonObject = { [key: string]: JsonValue }

export function isRecord(value: unknown): value is UnknownRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

export function isStringRecord(value: unknown): value is Record<string, string> {
  return isRecord(value) && Object.values(value).every(item => typeof item === 'string')
}

export function isJsonValue(value: unknown): value is JsonValue {
  if (value === null) return true
  if (typeof value === 'string' || typeof value === 'boolean') return true
  if (typeof value === 'number') return Number.isFinite(value)
  if (Array.isArray(value)) return value.every(isJsonValue)
  if (!isRecord(value)) return false
  return Object.values(value).every(isJsonValue)
}

export function isJsonObject(value: unknown): value is JsonObject {
  return isRecord(value) && isJsonValue(value)
}

export function stringValue(value: unknown): string | undefined {
  return typeof value === 'string' ? value : undefined
}

export function numberValue(value: unknown): number | undefined {
  return typeof value === 'number' ? value : undefined
}

export function objectValue(value: JsonValue | undefined): JsonObject | undefined {
  return isJsonObject(value) ? value : undefined
}

export function arrayValue(value: JsonValue | undefined): JsonValue[] | undefined {
  return Array.isArray(value) ? value : undefined
}

export function readString(record: UnknownRecord, key: string): string | undefined {
  return stringValue(record[key])
}

export function readNumber(record: UnknownRecord, key: string): number | undefined {
  return numberValue(record[key])
}
