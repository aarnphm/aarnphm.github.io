import type {
  GarminCache,
  GarminLactateThresholdValue,
  GarminRunningLactateThreshold,
  GarminSleepSummary,
  GarminVo2Day,
  GarminWeightSample,
} from '../../plugins/stores/garmin'
import {
  garminConnectLactateThresholdHistory,
  garminConnectRunningLactateThreshold,
  garminConnectSleep,
  garminConnectVo2,
  garminConnectWeightSamples,
} from '../../util/garmin-connect'
import { fetchGarminJson, type GarminConnectSession } from '../../util/garmin-session'
import { localIsoDayOffset, shiftIsoDay } from '../../util/local-date'
import { syncRefreshDays } from '../../util/sync-refresh-window'
import { isRecord, readString } from '../../util/type-guards'
import {
  cleanDay,
  endDate,
  garminLactateThresholdRefreshStart,
  readTriathlonStart,
  sleep,
  type GarminFetchOutcome,
} from './refresh-window'

export function resolveGarminWeightDay(
  day: string,
  outcome: GarminFetchOutcome<GarminWeightSample[]>,
  summary: GarminWeightSample,
  previous: GarminWeightSample[],
): GarminWeightSample[] {
  if (outcome.ok) return outcome.value?.length ? outcome.value : [summary]
  const prior = previous.filter(sample => sample.date === day)
  return prior.length ? prior : [summary]
}

export function mergeGarminVo2Range(
  previous: Record<string, GarminVo2Day>,
  fetched: Record<string, GarminVo2Day>,
  start: string,
  end: string,
): Record<string, GarminVo2Day> {
  const merged: Record<string, GarminVo2Day> = {}
  for (const [date, value] of Object.entries(previous))
    if (date < start || date > end) merged[date] = value
  for (const [date, value] of Object.entries(fetched)) merged[date] = value
  return Object.fromEntries(
    Object.entries(merged).sort(([left], [right]) => left.localeCompare(right)),
  )
}

export function mergeGarminWeightRange(
  previous: GarminWeightSample[],
  fetched: GarminWeightSample[],
  start: string,
  end: string,
): GarminWeightSample[] {
  const merged = new Map<number, GarminWeightSample>()
  for (const sample of previous)
    if (sample.date < start || sample.date > end) merged.set(sample.ts, sample)
  for (const sample of fetched) merged.set(sample.ts, sample)
  return [...merged.values()].sort((left, right) => left.ts - right.ts)
}

export async function fetchRunningLactateThreshold(
  session: GarminConnectSession,
  base: string,
  previous?: GarminRunningLactateThreshold | null,
  refreshWindowDays = syncRefreshDays(),
  stale = false,
): Promise<GarminRunningLactateThreshold | null> {
  const latest = garminConnectRunningLactateThreshold(
    await fetchGarminJson(session, base, '/biometric-service/biometric/latestLactateThreshold'),
  )
  const explicit =
    cleanDay(process.env.GARMIN_CONNECT_START_DATE) ?? cleanDay(process.env.GARMIN_CONNECT_SINCE)
  const configured = (await readTriathlonStart()) ?? localIsoDayOffset(-90)
  const end = endDate()
  const start =
    explicit ??
    garminLactateThresholdRefreshStart(previous, configured, end, refreshWindowDays, stale)
  const history = {
    speedMps: previous?.history?.speedMps ?? [],
    heartRateBpm: previous?.history?.heartRateBpm ?? [],
  }
  const metrics = [
    { key: 'speedMps', endpoint: 'lactateThresholdSpeed' },
    { key: 'heartRateBpm', endpoint: 'lactateThresholdHeartRate' },
  ] as const
  for (const { key, endpoint } of metrics) {
    try {
      const raw = await fetchGarminJson(
        session,
        base,
        `/biometric-service/stats/${endpoint}/range/${start}/${end}`,
        new URLSearchParams({
          sport: 'RUNNING',
          aggregation: 'daily',
          aggregationStrategy: 'LATEST',
        }),
      )
      history[key] = mergeGarminLactateThresholdHistory(
        history[key],
        garminConnectLactateThresholdHistory(raw, key),
        start,
        end,
      )
    } catch (error) {
      console.warn(
        `[garmin] lactate threshold ${key} history failed: ${error instanceof Error ? error.message : error}`,
      )
    }
  }
  return latest || history.speedMps.length || history.heartRateBpm.length
    ? { speedMps: latest?.speedMps ?? null, heartRateBpm: latest?.heartRateBpm ?? null, history }
    : null
}

export function mergeGarminLactateThresholdHistory(
  previous: GarminLactateThresholdValue[],
  fetched: GarminLactateThresholdValue[],
  start: string,
  end: string,
): GarminLactateThresholdValue[] {
  const days = new Map(
    previous
      .filter(point => point.date < start || point.date > end)
      .map(point => [point.date, point]),
  )
  for (const point of fetched) days.set(point.date, point)
  return [...days.values()].sort((left, right) => left.date.localeCompare(right.date))
}

interface GarminSleepFetchResult {
  sleep: Record<string, GarminSleepSummary>
  fetchedDays: number
  failedDays: string[]
}

export async function fetchGarminSleepRange(
  session: GarminConnectSession,
  base: string,
  previous: Readonly<Record<string, GarminSleepSummary>>,
  start: string,
  end: string,
  delayMs: number,
): Promise<GarminSleepFetchResult> {
  const profile = await fetchGarminJson(session, base, '/userprofile-service/socialProfile')
  const displayName = isRecord(profile) ? readString(profile, 'displayName') : undefined
  if (!displayName) throw new Error('Garmin profile omitted displayName')
  const summaries: Record<string, GarminSleepSummary> = { ...previous }
  let fetchedDays = 0
  const failedDays: string[] = []
  for (let day = start; day <= end; day = shiftIsoDay(day, 1)) {
    try {
      const raw = await fetchGarminJson(
        session,
        base,
        `/wellness-service/wellness/dailySleepData/${encodeURIComponent(displayName)}`,
        new URLSearchParams({ date: day, nonSleepBufferMinutes: '0' }),
      )
      if (!isRecord(raw) || (raw.dailySleepDTO !== null && !isRecord(raw.dailySleepDTO)))
        throw new Error('Garmin sleep response omitted a valid dailySleepDTO')
      const summary = garminConnectSleep(raw, day)
      if (summary) summaries[day] = summary
      else delete summaries[day]
      fetchedDays++
    } catch (error) {
      failedDays.push(day)
      console.warn(
        `[garmin] sleep ${day} failed: ${error instanceof Error ? error.message : error}`,
      )
    }
    if (delayMs > 0 && day < end) await sleep(delayMs)
  }
  return {
    sleep: Object.fromEntries(
      Object.entries(summaries).sort(([left], [right]) => left.localeCompare(right)),
    ),
    fetchedDays,
    failedDays,
  }
}

export async function syncGarminBiometrics(
  session: GarminConnectSession,
  base: string,
  previous: GarminCache | null,
  start: string,
  end: string,
  refreshWindowDays: number,
  stale: boolean,
  delayMs: number,
): Promise<{
  lactateThresholdOutcome: GarminFetchOutcome<GarminRunningLactateThreshold | null>
  vo2max: Record<string, GarminVo2Day>
  weight: GarminWeightSample[]
}> {
  let lactateThresholdOutcome: GarminFetchOutcome<GarminRunningLactateThreshold | null> = {
    ok: false,
  }
  try {
    lactateThresholdOutcome = {
      ok: true,
      value: await fetchRunningLactateThreshold(
        session,
        base,
        previous?.runningLactateThreshold,
        refreshWindowDays,
        stale,
      ),
    }
  } catch (err) {
    console.warn(
      `[garmin] lactate threshold fetch failed: ${err instanceof Error ? err.message : err}`,
    )
  }

  let vo2Outcome: GarminFetchOutcome<Record<string, GarminVo2Day>> = { ok: false }
  try {
    const raw = await fetchGarminJson(
      session,
      base,
      `/metrics-service/metrics/maxmet/daily/${encodeURIComponent(start)}/${encodeURIComponent(end)}`,
    )
    const fetched: Record<string, GarminVo2Day> = {}
    for (const day of garminConnectVo2(raw)) fetched[day.date] = day
    vo2Outcome = { ok: true, value: fetched }
    console.log(`[garmin] vo2max days: ${Object.keys(fetched).length}`)
  } catch (err) {
    console.warn(`[garmin] vo2max fetch failed: ${err instanceof Error ? err.message : err}`)
  }
  const vo2max = vo2Outcome.ok
    ? mergeGarminVo2Range(previous?.vo2max ?? {}, vo2Outcome.value ?? {}, start, end)
    : (previous?.vo2max ?? {})

  let weightOutcome: GarminFetchOutcome<GarminWeightSample[]> = { ok: false }
  try {
    const rangeRaw = await fetchGarminJson(
      session,
      base,
      '/weight-service/weight/dateRange',
      new URLSearchParams({ startDate: start, endDate: end }),
    )
    const byDay = new Map<string, GarminWeightSample>()
    for (const s of garminConnectWeightSamples(rangeRaw)) byDay.set(s.date, s)
    const collected: GarminWeightSample[] = []
    for (const day of [...byDay.keys()].sort()) {
      let dayOutcome: GarminFetchOutcome<GarminWeightSample[]> = { ok: false }
      try {
        const dv = await fetchGarminJson(
          session,
          base,
          `/weight-service/weight/dayview/${encodeURIComponent(day)}`,
          new URLSearchParams({ includeAll: 'true' }),
        )
        dayOutcome = { ok: true, value: garminConnectWeightSamples(dv) }
      } catch (err) {
        console.warn(
          `[garmin] weight dayview ${day} failed: ${err instanceof Error ? err.message : err}`,
        )
      }
      collected.push(
        ...resolveGarminWeightDay(day, dayOutcome, byDay.get(day)!, previous?.weight ?? []),
      )
      if (delayMs > 0) await sleep(delayMs)
    }
    const deduped = new Map<number, GarminWeightSample>()
    for (const s of collected) deduped.set(s.ts, s)
    const weight = [...deduped.values()].sort((a, b) => a.ts - b.ts)
    weightOutcome = { ok: true, value: weight }
    const days = new Set(weight.map(s => s.date)).size
    console.log(`[garmin] weight samples: ${weight.length} over ${days} days`)
  } catch (err) {
    console.warn(`[garmin] weight fetch failed: ${err instanceof Error ? err.message : err}`)
  }
  const weight = weightOutcome.ok
    ? mergeGarminWeightRange(previous?.weight ?? [], weightOutcome.value ?? [], start, end)
    : (previous?.weight ?? [])

  return { lactateThresholdOutcome, vo2max, weight }
}
