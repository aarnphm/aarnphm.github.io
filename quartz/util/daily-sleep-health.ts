import type { AppleSleepDay } from '../plugins/stores/apple-sleep'
import type { GarminSleepSummary } from '../plugins/stores/garmin'
import type { OuraDaily, OuraSleepDetail } from '../plugins/stores/oura'
import { resolveSleepTotals } from './triathlon-day-sleep'

export interface DailySleepSources {
  sleepDurationS?: 'oura' | 'garmin' | 'apple'
  sleepScore?: 'oura' | 'garmin' | 'apple'
  napDurationS?: 'oura' | 'garmin' | 'apple'
  napCount?: 'oura' | 'garmin' | 'apple'
}

export function resolveDailySleepHealth(
  date: string,
  oura: OuraDaily | undefined,
  detail: OuraSleepDetail | undefined,
  garmin: GarminSleepSummary | undefined,
  apple?: AppleSleepDay,
): { sleepDurationS: number | null; sleepScore: number | null; sleepSources: DailySleepSources } {
  return resolveSleepTotals(
    detail,
    garmin?.date === date ? garmin : null,
    oura,
    apple?.date === date ? apple : null,
  )
}
