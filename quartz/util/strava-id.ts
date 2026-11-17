/** Strava IDs are decimal strings. A number is usable only before it has lost precision. */
export function stravaId(value: unknown): string | null {
  if (typeof value === 'string') return /^[1-9]\d*$/.test(value) ? value : null
  return typeof value === 'number' && Number.isSafeInteger(value) && value > 0
    ? String(value)
    : null
}

export function requireStravaId(value: unknown): string {
  const id = stravaId(value)
  if (id === null)
    throw new Error('Invalid Strava ID; expected an exact positive decimal string or safe integer')
  return id
}

export function stravaObjectId(value: Record<string, unknown>): string {
  if (value.id_str !== undefined) {
    if (typeof value.id_str !== 'string') throw new Error('Invalid Strava string ID')
    return requireStravaId(value.id_str)
  }
  return requireStravaId(value.id)
}

export const compareStravaIds = (left: string, right: string): number =>
  left.length - right.length || left.localeCompare(right)
