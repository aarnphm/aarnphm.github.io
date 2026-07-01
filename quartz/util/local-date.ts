const DEFAULT_LOCAL_TIME_ZONE = 'America/Toronto'

interface LocalParts {
  year: number
  month: number
  day: number
  hour: number
  minute: number
  second: number
}

export function resolveLocalTimeZone(): string {
  const configured =
    process.env.HEALTH_TIMEZONE?.trim() ||
    process.env.LOCAL_TIMEZONE?.trim() ||
    process.env.TZ?.trim()
  if (configured) return configured
  return Intl.DateTimeFormat().resolvedOptions().timeZone || DEFAULT_LOCAL_TIME_ZONE
}

function localParts(ms: number, timeZone: string): LocalParts {
  const values: Record<string, string> = {}
  const formatter = new Intl.DateTimeFormat('en-CA', {
    timeZone,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hourCycle: 'h23',
  })
  for (const part of formatter.formatToParts(new Date(ms))) {
    if (part.type !== 'literal') values[part.type] = part.value
  }
  return {
    year: Number(values.year),
    month: Number(values.month),
    day: Number(values.day),
    hour: Number(values.hour),
    minute: Number(values.minute),
    second: Number(values.second),
  }
}

function isoFromUtcMs(ms: number): string {
  return new Date(ms).toISOString().slice(0, 10)
}

function isoDayParts(day: string): { year: number; month: number; day: number } {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(day)
  if (!match) throw new Error(`${day} is not YYYY-MM-DD`)
  return { year: Number(match[1]), month: Number(match[2]), day: Number(match[3]) }
}

export function localIsoDay(ms = Date.now(), timeZone = resolveLocalTimeZone()): string {
  const parts = localParts(ms, timeZone)
  return [
    String(parts.year).padStart(4, '0'),
    String(parts.month).padStart(2, '0'),
    String(parts.day).padStart(2, '0'),
  ].join('-')
}

export function shiftIsoDay(day: string, offsetDays: number): string {
  const parts = isoDayParts(day)
  return isoFromUtcMs(Date.UTC(parts.year, parts.month - 1, parts.day + offsetDays))
}

export function localIsoDayOffset(
  offsetDays: number,
  ms = Date.now(),
  timeZone = resolveLocalTimeZone(),
): string {
  return shiftIsoDay(localIsoDay(ms, timeZone), offsetDays)
}

function timezoneOffsetMs(ms: number, timeZone: string): number {
  const parts = localParts(ms, timeZone)
  const rounded = Math.floor(ms / 1000) * 1000
  return (
    Date.UTC(parts.year, parts.month - 1, parts.day, parts.hour, parts.minute, parts.second) -
    rounded
  )
}

function localWallTimeUtcMs(parts: LocalParts, timeZone: string): number {
  const wall = Date.UTC(
    parts.year,
    parts.month - 1,
    parts.day,
    parts.hour,
    parts.minute,
    parts.second,
  )
  let guess = wall
  for (let i = 0; i < 4; i++) {
    const next = wall - timezoneOffsetMs(guess, timeZone)
    if (next === guess) return next
    guess = next
  }
  return guess
}

export function localDayStartUtcMs(day: string, timeZone = resolveLocalTimeZone()): number {
  const parts = isoDayParts(day)
  return localWallTimeUtcMs(
    { year: parts.year, month: parts.month, day: parts.day, hour: 0, minute: 0, second: 0 },
    timeZone,
  )
}

export function localDayEndUtcMs(day: string, timeZone = resolveLocalTimeZone()): number {
  return localDayStartUtcMs(shiftIsoDay(day, 1), timeZone) - 1
}
