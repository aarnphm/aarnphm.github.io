export interface TrackEntry {
  date: string
  weightLbs: number | null
  weightKg: number | null
  race: boolean
  event: string | null
}

export interface RaceEvent {
  date: string
  event: string | null
}

export interface TrackingData {
  days: TrackEntry[]
  races: RaceEvent[]
}

const LB_TO_KG = 0.45359237

export function parseTrackingMeta(meta: string | null | undefined): {
  race: boolean
  event: string | null
} {
  let race = false
  let event: string | null = null
  if (!meta) return { race, event }
  const re = /(\w+)\s*=\s*(?:"([^"]*)"|'([^']*)'|(\S+))/g
  let m: RegExpExecArray | null
  while ((m = re.exec(meta)) !== null) {
    const key = m[1].toLowerCase()
    const val = m[2] ?? m[3] ?? m[4] ?? ''
    if (key === 'race') race = val === 'true' || val === '1' || val === 'yes'
    else if (key === 'event') event = val
  }
  return { race, event }
}

export function parseTrackingBlock(
  meta: string | null | undefined,
  value: string,
): TrackEntry | null {
  const body: Record<string, string> = {}
  for (const line of value.split('\n')) {
    const idx = line.indexOf(':')
    if (idx < 0) continue
    const k = line.slice(0, idx).trim().toLowerCase()
    const v = line.slice(idx + 1).trim()
    if (k) body[k] = v
  }
  const date = body.date
  if (!date || !/^\d{4}-\d{2}-\d{2}/.test(date)) return null
  const wl = body.weight != null ? Number(body.weight) : NaN
  const weightLbs = Number.isFinite(wl) ? wl : null
  const weightKg = weightLbs != null ? Math.round(weightLbs * LB_TO_KG * 10) / 10 : null
  const { race, event } = parseTrackingMeta(meta)
  return { date: date.slice(0, 10), weightLbs, weightKg, race, event }
}
