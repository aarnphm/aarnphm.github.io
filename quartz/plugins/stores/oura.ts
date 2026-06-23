export interface OuraDaily {
  date: string
  readiness: number | null
  sleepScore: number | null
  hrv: number | null
  rhr: number | null
  sleepDurationS: number | null
  tempDeviationC: number | null
  totalCalories: number | null
  activeCalories: number | null
}

export interface OuraAuth {
  refreshToken: string
  obtainedAt: number
}

export interface OuraUser {
  id: string | null
  email: string | null
}

export interface OuraCache {
  version?: number
  auth?: OuraAuth
  user?: OuraUser
  lastSync: number
  days: Record<string, OuraDaily>
}

export interface OuraSleepDateFields {
  day?: unknown
  bedtime_end?: unknown
}

const localDatePattern = /^\d{4}-\d{2}-\d{2}T/

export function ouraSleepCalendarDay(row: OuraSleepDateFields): string | null {
  if (typeof row.bedtime_end === 'string' && localDatePattern.test(row.bedtime_end))
    return row.bedtime_end.slice(0, 10)
  return typeof row.day === 'string' ? row.day : null
}

export function emptyOuraDaily(date: string): OuraDaily {
  return {
    date,
    readiness: null,
    sleepScore: null,
    hrv: null,
    rhr: null,
    sleepDurationS: null,
    tempDeviationC: null,
    totalCalories: null,
    activeCalories: null,
  }
}
