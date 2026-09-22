import type { GarminHealthDay, GarminHealthReading } from '../plugins/stores/garmin-health'
import {
  garminBodyBattery,
  garminTrainingReadiness,
  garminTrainingStatus,
  garminEnduranceScore,
  garminHillScore,
  garminHealthFetchResult,
  garminHealthFetchError,
} from '../util/garmin-health'
import { fetchGarminJson, type GarminConnectSession } from '../util/garmin-session'
import { shiftIsoDay } from '../util/local-date'

export async function fetchGarminHealthRange(
  session: GarminConnectSession,
  base: string,
  previous: Readonly<Record<string, GarminHealthDay>>,
  start: string,
  end: string,
  delayMs: number,
): Promise<{ health: Record<string, GarminHealthDay>; responses: number; failures: number }> {
  const health = { ...previous }
  let responses = 0
  let failures = 0
  for (let date = start; date <= end; date = shiftIsoDay(date, 1)) {
    const prior = previous[date]
    const read = async <T>(
      path: string,
      params: URLSearchParams | undefined,
      parser: (raw: unknown, date: string) => T | null,
      cached: GarminHealthReading<T> | undefined,
    ): Promise<GarminHealthReading<T>> => {
      const now = Date.now()
      try {
        const raw = await fetchGarminJson(session, base, path, params, {
          signal: AbortSignal.timeout(30_000),
        })
        const value = parser(raw, date)
        responses++
        return garminHealthFetchResult(value, now)
      } catch (error) {
        failures++
        console.warn(
          `[garmin] health ${date} ${path}: ${error instanceof Error ? error.message : 'request failed'}`,
        )
        return garminHealthFetchError(cached, now)
      } finally {
        if (delayMs > 0) await new Promise(resolve => setTimeout(resolve, delayMs))
      }
    }
    health[date] = {
      date,
      source: 'garmin',
      bodyBattery: await read(
        '/wellness-service/wellness/bodyBattery/reports/daily',
        new URLSearchParams({ startDate: date, endDate: date }),
        garminBodyBattery,
        prior?.bodyBattery,
      ),
      trainingReadiness: await read(
        `/metrics-service/metrics/trainingreadiness/${date}`,
        undefined,
        garminTrainingReadiness,
        prior?.trainingReadiness,
      ),
      trainingStatus: await read(
        `/metrics-service/metrics/trainingstatus/aggregated/${date}`,
        undefined,
        garminTrainingStatus,
        prior?.trainingStatus,
      ),
      enduranceScore: await read(
        '/metrics-service/metrics/endurancescore',
        new URLSearchParams({ calendarDate: date }),
        garminEnduranceScore,
        prior?.enduranceScore,
      ),
      hillScore: await read(
        '/metrics-service/metrics/hillscore',
        new URLSearchParams({ calendarDate: date }),
        garminHillScore,
        prior?.hillScore,
      ),
    }
    console.log(
      `[garmin] health ${date}: ${Object.values(health[date]).filter(value => typeof value === 'object' && value.status === 'available').length}/5 available`,
    )
  }
  return { health, responses, failures }
}
