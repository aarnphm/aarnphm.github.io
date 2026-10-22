import { selectActivityAnalysisSummary } from '../../../util/activity-analysis-selection'
import { latestProviderSync } from '../../../util/provider-sync'
import { matchGarminHeartRateActivity, type GarminWeightSample } from '../garmin'
import { resolveActivityHeartRate, round, type StravaRawCache } from '../strava'
import { TrackEntry } from '../tracking'
import { fatFreeMassIndex, fatFreeMassKg, katchMcArdleBmr } from './body-and-labs'
import {
  ATHLETE,
  DAY_MS,
  dayMs,
  decouplingOf,
  efOf,
  goalWeightKg,
  gradeAdjSpeed,
  oneHz,
  PEAK_WINDOWS,
  peakDistRate,
  peakMean,
  TANAKA_A,
  TANAKA_B,
} from './primitives'
import { activityWithProviderMetrics } from './provider-metrics'
import {
  ACTIVITY_FIELDS,
  DAY_FIELDS,
  WEEK_FIELDS,
  type Analytics,
  type DataFeedInputs,
} from './schema'

const pickLine = (row: Record<string, unknown>, fields: readonly string[]): string =>
  JSON.stringify(Object.fromEntries(fields.map(k => [k, row[k] ?? null])))

const rd = (x: number | null, dp: number): number | null => (x == null ? null : round(x, dp))

const nextIso = (date: string): string => new Date(dayMs(date) + DAY_MS).toISOString().slice(0, 10)

export function buildDataFeed(
  cache: StravaRawCache | null,
  analytics: Analytics,
  inputs: DataFeedInputs = {},
): string {
  const generatedAt = latestProviderSync(
    inputs.generatedAt,
    cache,
    inputs.oura,
    inputs.apple,
    inputs.weather,
    inputs.garmin,
  )
  const ouraDays = inputs.oura?.days ?? {}
  const appleDays = inputs.apple?.days ?? {}
  const vThrBySport = new Map(analytics.thresholds.map(t => [t.sport, t.vThr]))
  const skipTrainingActivityIds = new Set(
    (inputs.trainingExclusions ?? []).map(entry => entry.activityId),
  )
  const sorted = [...analytics.activities].sort(
    (p, q) => p.date.localeCompare(q.date) || p.id - q.id,
  )

  const byDay = new Map<string, { sessions: number; meters: number; seconds: number }>()
  for (const s of sorted) {
    const raw = cache?.activities[String(s.id)]
    const detail = inputs.activityDetails?.[String(s.id)]
    const b = byDay.get(s.date) ?? { sessions: 0, meters: 0, seconds: 0 }
    b.sessions += 1
    b.meters += s.sport === 'sauna' ? 0 : (raw?.distance ?? (detail?.distanceKm ?? 0) * 1_000)
    b.seconds += s.movingTimeS
    byDay.set(s.date, b)
  }

  const windByDate = new Map<string, TrackEntry>()
  for (const w of inputs.weights ?? [])
    if (w.windKph != null || w.windDir != null) windByDate.set(w.date, w)
  const weatherDays = inputs.weather?.days ?? {}
  const garminWeightByDay = new Map<string, GarminWeightSample>()
  const feedWeightRaw = inputs.garmin?.weight
  for (const s of Array.isArray(feedWeightRaw) ? feedWeightRaw : [])
    garminWeightByDay.set(s.date, s)

  const dayLines = analytics.daily.map(d => {
    const o = ouraDays[d.date]
    const next = ouraDays[nextIso(d.date)]
    const ap = appleDays[d.date]
    const gw = garminWeightByDay.get(d.date)
    const agg = byDay.get(d.date)
    const wind = windByDate.get(d.date)
    const weather = weatherDays[d.date]
    return pickLine(
      {
        kind: 'day',
        date: d.date,
        sessions: agg?.sessions ?? 0,
        km: round((agg?.meters ?? 0) / 1000, 2),
        hours: round((agg?.seconds ?? 0) / 3600, 2),
        load: d.load,
        swimLoad: d.swimLoad,
        bikeLoad: d.bikeLoad,
        runLoad: d.runLoad,
        ctl: d.ctl,
        atl: d.atl,
        tsb: d.tsb,
        swimCtl: d.swimCtl,
        bikeCtl: d.bikeCtl,
        runCtl: d.runCtl,
        warmup: d.warmup,
        readiness: o?.readiness ?? null,
        sleepScore: o?.sleepScore ?? null,
        hrv: o?.hrv ?? null,
        rhr: o?.rhr ?? null,
        sleepDurationS: o?.sleepDurationS ?? null,
        napDurationS: d.napDurationS ?? null,
        napCount: d.napCount ?? null,
        garminHealth: d.garminHealth ?? null,
        ouraHealth: d.ouraHealth ?? null,
        sleepMetrics: d.sleepMetrics,
        tempDeviationC: o?.tempDeviationC ?? null,
        totalCalories: d.totalCalories ?? o?.totalCalories ?? ap?.burnKcal ?? null,
        activeCalories: o?.activeCalories ?? ap?.activeKcal ?? null,
        intakeKcal: d.intakeKcal ?? ap?.intakeKcal ?? null,
        weightKg: d.weightKg ?? ap?.weightKg ?? null,
        bmi: gw?.bmi ?? null,
        ffmi:
          gw?.weightKg != null && gw?.bodyFatPct != null
            ? fatFreeMassIndex(fatFreeMassKg(gw.weightKg, gw.bodyFatPct), ATHLETE.heightCm)
            : null,
        bodyFatPct: gw?.bodyFatPct ?? null,
        bodyWaterPct: gw?.bodyWaterPct ?? null,
        muscleMassKg: gw?.muscleMassKg ?? null,
        boneMassKg: gw?.boneMassKg ?? null,
        bmr:
          gw?.weightKg != null && gw?.bodyFatPct != null
            ? katchMcArdleBmr(gw.weightKg, gw.bodyFatPct)
            : null,
        windKph: wind?.windKph ?? weather?.windKph ?? null,
        windDir: wind?.windDir ?? weather?.windDir ?? null,
        windGustKph: weather?.windGustKph ?? null,
        readinessNext: next?.readiness ?? null,
        hrvNext: next?.hrv ?? null,
      },
      DAY_FIELDS,
    )
  })

  const activityLines: string[] = []
  for (const s of sorted) {
    const detail = inputs.activityDetails?.[String(s.id)]
    const source = cache?.activities[String(s.id)]
    const raw = source ? activityWithProviderMetrics(source, detail) : undefined
    const selectedAnalysis = detail ? selectActivityAnalysisSummary(detail.analyses) : null
    const environment = detail?.analyses.derived.environment ?? null
    const apparentWind = detail?.analyses.derived.apparentWind ?? null
    const selectedUv = selectedAnalysis?.uvExposure ?? null
    const selectedUvScore =
      selectedUv?.kind === 'pelotan-score' || selectedUv?.kind === 'garden-score'
        ? selectedUv.score
        : null
    const selectedUvSeverity =
      selectedUv?.kind === 'pelotan-score'
        ? selectedUv.severity
        : selectedUv?.kind === 'garden-score'
          ? selectedUv.severity
          : null
    const selectedAverageUvIndex =
      selectedUv?.kind === 'pelotan-score'
        ? (detail?.analyses.native.pelotan?.averageUvIndex ?? null)
        : (environment?.summary.averageUvIndex ?? null)
    if (s.sport === 'sauna' && detail?.sauna) {
      activityLines.push(
        pickLine(
          {
            kind: 'activity',
            id: s.id,
            date: s.date,
            sport: s.sport,
            name: s.name,
            distanceKm: 0,
            movingTimeS: detail.movingTimeS,
            elapsedTimeS: detail.movingTimeS,
            elevationM: 0,
            avgHr: detail.avgHr,
            maxHr: detail.maxHr,
            avgWatts: null,
            weightedWatts: null,
            deviceWatts: false,
            cadence: null,
            strokes: null,
            calories: null,
            sufferScore: null,
            deviceTemperatureC: detail.deviceTemperatureC,
            ambientTemperatureC: detail.ambientTemperatureC,
            weatherImpactPct: selectedAnalysis?.weatherImpact?.valuePct ?? null,
            weatherImpactSource: selectedAnalysis?.weatherImpact ? 'mywindsock' : null,
            uvExposureKind: selectedUv?.kind ?? null,
            uvScore: selectedUvScore,
            uvSeverity: selectedUvSeverity,
            ambientSed: environment?.summary.ambientSed ?? null,
            avgUvIndex: selectedAverageUvIndex,
            peakUvIndex: environment?.summary.peakUvIndex ?? null,
            avgCloudCoverPct:
              selectedUv?.kind === 'pelotan-score'
                ? (detail.analyses.native.pelotan?.averageCloudCoverPct ?? null)
                : (environment?.summary.averageCloudCoverPct ?? null),
            gardenHeadwindSharePct: apparentWind?.summary.headwindSharePct ?? null,
            gardenApparentAirRatio: apparentWind?.summary.apparentAirRatio ?? null,
            gardenWindCoveragePct: apparentWind?.summary.coveragePct ?? null,
            windKph: detail.windKph,
            windDir: detail.windDir,
            windGustKph: detail.windGustKph,
            sauna: detail.sauna,
            skipTraining: true,
            vGap: 0,
            intensity: null,
            load: 0,
            pp30: null,
            pp60: null,
            pp300: null,
            pp1200: null,
            ps30: null,
            ps60: null,
            ps300: null,
            ps1200: null,
            ef: null,
            decoupling: null,
          },
          ACTIVITY_FIELDS,
        ),
      )
      continue
    }
    if (!raw) continue
    if (s.sport !== 'swim' && s.sport !== 'bike' && s.sport !== 'run') continue
    const stream = cache?.streams?.[String(s.id)]
    const heartRate = resolveActivityHeartRate(
      raw,
      s.sport,
      stream,
      matchGarminHeartRateActivity(raw, s.sport, inputs.garmin ?? null),
      inputs.garmin ?? null,
    )
    const vThr = vThrBySport.get(s.sport) ?? 0
    const vGap = gradeAdjSpeed(raw, s.sport, stream)
    const watts = stream?.watts ?? []
    const distArr = stream?.distance ?? []
    const wattsOk = oneHz(watts.length, raw.movingTime)
    const distOk = s.sport !== 'swim' && oneHz(distArr.length, raw.movingTime)
    const peaks: Record<string, number | null> = {}
    for (const w of PEAK_WINDOWS) {
      peaks[`pp${w}`] = wattsOk ? rd(peakMean(watts, w), 0) : null
      peaks[`ps${w}`] = distOk ? rd(peakDistRate(distArr, w), 2) : null
    }
    activityLines.push(
      pickLine(
        {
          kind: 'activity',
          id: s.id,
          date: s.date,
          sport: s.sport,
          name: s.name,
          distanceKm: round(raw.distance / 1000, s.sport === 'swim' ? 3 : 2),
          movingTimeS: raw.movingTime,
          elapsedTimeS: raw.elapsedTime,
          elevationM: round(raw.totalElevationGain, 0),
          avgHr: heartRate.avgHr,
          maxHr: heartRate.maxHr,
          avgWatts: raw.averageWatts ?? null,
          weightedWatts: raw.weightedAverageWatts ?? null,
          deviceWatts: raw.deviceWatts ?? null,
          cadence: s.cadence,
          strokes: s.strokes,
          calories: raw.calories ?? null,
          sufferScore: raw.sufferScore ?? null,
          deviceTemperatureC: raw.averageTemp ?? null,
          ambientTemperatureC:
            detail?.ambientTemperatureC ??
            inputs.weather?.activities[String(s.id)]?.temperatureC ??
            null,
          weatherImpactPct: selectedAnalysis?.weatherImpact?.valuePct ?? null,
          weatherImpactSource: selectedAnalysis?.weatherImpact ? 'mywindsock' : null,
          uvExposureKind: selectedUv?.kind ?? null,
          uvScore: selectedUvScore,
          uvSeverity: selectedUvSeverity,
          ambientSed: environment?.summary.ambientSed ?? null,
          avgUvIndex: selectedAverageUvIndex,
          peakUvIndex: environment?.summary.peakUvIndex ?? null,
          avgCloudCoverPct:
            selectedUv?.kind === 'pelotan-score'
              ? (detail?.analyses.native.pelotan?.averageCloudCoverPct ?? null)
              : (environment?.summary.averageCloudCoverPct ?? null),
          gardenHeadwindSharePct: apparentWind?.summary.headwindSharePct ?? null,
          gardenApparentAirRatio: apparentWind?.summary.apparentAirRatio ?? null,
          gardenWindCoveragePct: apparentWind?.summary.coveragePct ?? null,
          windKph: s.windKph,
          windDir: s.windDir,
          windGustKph: s.windGustKph,
          sauna: null,
          skipTraining: skipTrainingActivityIds.has(s.id),
          vGap: round(vGap, 3),
          intensity: vThr > 0 ? round(vGap / vThr, 3) : null,
          load: s.load,
          ...peaks,
          ef: efOf(s.sport, raw, vGap, heartRate.avgHr),
          decoupling: decouplingOf(s.sport, stream, raw.movingTime),
        },
        ACTIVITY_FIELDS,
      ),
    )
  }

  const weekLines = analytics.weekly.map(w => pickLine({ kind: 'week', ...w }, WEEK_FIELDS))

  const todayMsV = dayMs(analytics.meta.today)
  const bornMs = dayMs(ATHLETE.bornAnchor)
  const ageYears =
    Number.isFinite(todayMsV) && todayMsV > bornMs
      ? Math.floor((todayMsV - bornMs) / (365.25 * DAY_MS))
      : null
  const meta = {
    kind: 'meta',
    v: 6,
    generatedAt,
    athleteId: analytics.meta.athleteId,
    today: analytics.meta.today,
    windowFrom: analytics.meta.windowFrom,
    windowTo: analytics.meta.windowTo,
    athlete: {
      sex: ATHLETE.sex,
      born: ATHLETE.born,
      ageYears,
      heightCm: ATHLETE.heightCm,
      hrMaxEst:
        ATHLETE.hrMax ?? (ageYears != null ? round(TANAKA_A - TANAKA_B * ageYears, 1) : null),
      weightGoalKg: goalWeightKg,
      ftp: ATHLETE.ftp,
      goalFtp: ATHLETE.goalFTP,
    },
    zones: inputs.zones
      ? { hr: inputs.zones.hr, power: inputs.zones.power, ftp: inputs.zones.ftp }
      : null,
    criticalPower: analytics.powerCurve.criticalPower,
    criticalPowerYear: analytics.powerCurve.criticalPowerYear,
    thresholds: analytics.thresholds.map(t => ({
      sport: t.sport,
      vThr: t.vThr,
      conf: t.conf,
      sampleSize: t.sampleSize,
    })),
    counts: { day: dayLines.length, activity: activityLines.length, week: weekLines.length },
    labels: [
      { key: 'readinessNext', source: 'oura readiness', lagDays: 1 },
      { key: 'hrvNext', source: 'oura hrv', lagDays: 1 },
    ],
    fields: { day: DAY_FIELDS, activity: ACTIVITY_FIELDS, week: WEEK_FIELDS },
    defs: {
      load: 'clamp(vGap/vThr,0,1.15)^2 * movingHours * 100',
      intensity: 'vGap/vThr, unclamped, current threshold applied retroactively',
      vGap: 'grade-adjusted speed m/s',
      pp: 'peak mean watts over 30/60/300/1200s windows, 1hz power streams only',
      ps: 'peak mean m/s over 30/60/300/1200s from distance stream, bike+run only',
      ef: 'bike np/avgHr w/bpm; run+swim 60*vGap/avgHr m/min/bpm',
      decoupling: '(ef first half - ef second half)/ef first half * 100, streams, >=1200s only',
      hrMaxEst: 'declared max hr; 208 - 0.7*age (tanaka 2001) when unset',
      weightKg: 'garmin scale primary, tracking forward-filled, apple fallback',
      bodyComposition:
        'bmi/bodyFatPct/bodyWaterPct/muscleMassKg/boneMassKg from garmin index scale; ffmi derived from scale weight and body fat; measurement days only',
      ffmi: 'fat-free mass kg / height m^2',
      avgWatts: 'strava estimate unless deviceWatts true',
      criticalPower: 'P(t)=CP+Wprime/t, OLS in power space at 180/420/720s',
      windDir: 'tracking override, WeatherKit compass fallback',
    },
  }

  return [JSON.stringify(meta), ...dayLines, ...activityLines, ...weekLines].join('\n') + '\n'
}
