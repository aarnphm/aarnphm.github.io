import type { GarminSleepSummary } from '../plugins/stores/garmin'
import type { OuraDaily, OuraSleepDetail } from '../plugins/stores/oura'
import { resolveSleepTotals } from './triathlon-day-sleep'

export interface DailySleepSources {
  sleepDurationS?: 'oura' | 'garmin'
  sleepScore?: 'oura' | 'garmin'
  napDurationS?: 'oura' | 'garmin'
  napCount?: 'oura' | 'garmin'
}

export function resolveDailySleepHealth(
  date: string,
  oura: OuraDaily | undefined,
  detail: OuraSleepDetail | undefined,
  garmin: GarminSleepSummary | undefined,
): { sleepDurationS: number | null; sleepScore: number | null; sleepSources: DailySleepSources } {
  return resolveSleepTotals(detail, garmin?.date === date ? garmin : null, oura)
}
