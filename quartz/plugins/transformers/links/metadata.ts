import { isRecord } from '../../../util/type-guards'

const FALSE_LIKE_STRINGS = new Set(['false', '0', 'no', 'off'])

export function metadataDisablesPopover(metadata?: Record<string, unknown>): boolean {
  if (!metadata) return false
  const value = metadata.popover ?? metadata.noPopover ?? metadata['no-popover']
  if (value === undefined) return false
  if (typeof value === 'boolean') return value === false
  if (typeof value === 'number') return value === 0
  if (typeof value === 'string') {
    return FALSE_LIKE_STRINGS.has(value.trim().toLowerCase())
  }
  return false
}

export function readLinkMetadata(value: unknown): Record<string, unknown> | undefined {
  if (isRecord(value)) return value
  if (typeof value !== 'string' || !value.trim().startsWith('{')) return undefined
  try {
    const parsed: unknown = JSON.parse(value)
    return isRecord(parsed) ? parsed : undefined
  } catch {
    return undefined
  }
}
