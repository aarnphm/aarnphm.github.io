import { isRecord } from './type-guards'

export type LunarName =
  | 'new moon'
  | 'waxing crescent'
  | 'first quarter'
  | 'waxing gibbous'
  | 'full moon'
  | 'waning gibbous'
  | 'last quarter'
  | 'waning crescent'

export type LunarEvent = { at: number; phase: number; name: LunarName }
export type LunarPhase = {
  phase: number
  illumination: number
  waxing: boolean
  name: LunarName
  source: 'usno' | 'estimate'
}

const day = 86400000
const synodic = 29.530588853 * day
const newMoon = Date.UTC(2000, 0, 6, 18, 14)
const wrap = (phase: number) => ((phase % 1) + 1) % 1
const integer = (value: unknown): value is number =>
  typeof value === 'number' && Number.isInteger(value)

const primaryPhases = new Map<string, { phase: number; name: LunarName }>([
  ['New Moon', { phase: 0, name: 'new moon' }],
  ['First Quarter', { phase: 0.25, name: 'first quarter' }],
  ['Full Moon', { phase: 0.5, name: 'full moon' }],
  ['Last Quarter', { phase: 0.75, name: 'last quarter' }],
])

// USNO reports primary phases in Universal Time, independent of the viewer's time zone.
export function readLunarEvents(value: unknown): LunarEvent[] | null {
  if (
    !isRecord(value) ||
    'error' in value ||
    !Array.isArray(value.phasedata) ||
    value.phasedata.length < 2 ||
    value.phasedata.length > 99
  )
    return null
  const events: LunarEvent[] = []
  for (const entry of value.phasedata) {
    if (
      !isRecord(entry) ||
      !integer(entry.year) ||
      entry.year < 1700 ||
      entry.year > 2100 ||
      !integer(entry.month) ||
      !integer(entry.day) ||
      typeof entry.time !== 'string' ||
      typeof entry.phase !== 'string'
    )
      return null
    const phase = primaryPhases.get(entry.phase)
    const time = /^(\d{2}):(\d{2})$/.exec(entry.time)
    if (!phase || !time) return null
    const hour = Number(time[1])
    const minute = Number(time[2])
    if (hour > 23 || minute > 59) return null
    const at = Date.UTC(entry.year, entry.month - 1, entry.day, hour, minute)
    const date = new Date(at)
    if (
      date.getUTCFullYear() !== entry.year ||
      date.getUTCMonth() !== entry.month - 1 ||
      date.getUTCDate() !== entry.day
    )
      return null
    const previous = events.at(-1)
    if (
      previous &&
      (at <= previous.at ||
        at - previous.at > 12 * day ||
        phase.phase !== wrap(previous.phase + 0.25))
    )
      return null
    events.push({ at, ...phase })
  }
  return events
}

export function lunarPhaseName(phase: number): LunarName {
  if (phase < 0.0125 || phase > 0.9875) return 'new moon'
  if (Math.abs(phase - 0.25) < 0.0125) return 'first quarter'
  if (Math.abs(phase - 0.5) < 0.0125) return 'full moon'
  if (Math.abs(phase - 0.75) < 0.0125) return 'last quarter'
  return phase < 0.25
    ? 'waxing crescent'
    : phase < 0.5
      ? 'waxing gibbous'
      : phase < 0.75
        ? 'waning gibbous'
        : 'waning crescent'
}

export function lunarPhase(at: number, events: readonly LunarEvent[] = []): LunarPhase {
  const index = events.findIndex(event => event.at > at)
  const before = events[index - 1]
  const after = events[index]
  const anchored = Boolean(before && after)
  // Quarter spacing varies. Interpolate only inside the published interval; the projected
  // illuminated fraction is an estimate, rather than a provider-native illumination reading.
  const phase = wrap(
    before && after
      ? before.phase + (0.25 * (at - before.at)) / (after.at - before.at)
      : (at - newMoon) / synodic,
  )
  const exact = events.find(event => event.at === at)
  const current = exact?.phase ?? phase
  return {
    phase: current,
    illumination: (1 - Math.cos(current * Math.PI * 2)) / 2,
    waxing: current < 0.5,
    name: exact?.name ?? lunarPhaseName(current),
    source: exact || anchored ? 'usno' : 'estimate',
  }
}

// A sphere's terminator projects to an ellipse. Quantize its scanlines to the scene's pixel grid.
export function lunarDiscPath(phase: number, radius: number, pixel: number): string {
  const waxing = wrap(phase) < 0.5
  const cosine = Math.cos(wrap(phase) * Math.PI * 2)
  const paths: string[] = []
  for (let y = -radius; y < radius; y += pixel) {
    const edge = Math.floor(Math.sqrt(radius * radius - (y + pixel / 2) ** 2) / pixel) * pixel
    const terminator = Math.round((edge * cosine) / pixel) * pixel
    const left = waxing ? terminator : -edge
    const width = edge - terminator
    if (width > 0) paths.push(`M${left} ${y}h${width}v${pixel}h-${width}Z`)
  }
  return paths.join('')
}
