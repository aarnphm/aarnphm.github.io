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
