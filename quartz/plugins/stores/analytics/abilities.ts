import type { BodyBlock } from './body-and-labs'
import type { PowerCurveBlock } from './power-to-weight'
import { WORLD_TOUR_POWER_REFERENCE } from '../../../util/cycling-ability-reference'
import { SWIM_PACE_MAX_S_PER_100M, SWIM_PACE_MIN_S_PER_100M } from '../../../util/swim-metrics'
import { round, SPORT_ORDER, type Sport, type StravaRawCache } from '../strava'
import {
  bikePowerScore,
  CAD_UNIT,
  cadScoreOf,
  cadValOf,
  COGGAN_FTP_WKG,
  CTL_ANCHOR,
  plausibleVgap,
  PROJ_HORIZON_D,
  PROJ_MIN_POINTS,
  PROJ_WINDOW_D,
  rollingMetricMean,
  RUN_SPRINT_MS,
  RUN_THR_MS,
  sportCtlOf,
  SWIM_SPEED_MS,
  swimPaceScoreOf,
  swimSpeedScoreOf,
} from './ability-scoring'
import {
  clamp,
  DAY_MS,
  dayMs,
  LOAD_SHARE_TARGET,
  mean,
  olsSlope,
  oneHz,
  peakDistRate,
  RUN_SPRINT_WIN_S,
  SPRINT_CAP_MS,
  winValues,
  type Act,
  type DailyPoint,
  type RunningDynamicsMetrics,
  type SwimActivityMetrics,
  type ThresholdEstimate,
} from './primitives'
import {
  metricBounds,
  norm01,
  personalMetricScore,
  type AbilitiesBlock,
  type AbilityTrendPoint,
  type RadarAxis,
  type SportAbilities,
} from './schema'

/** Engine state the ability radars read. */
interface AbilityScope {
  cache: StravaRawCache
  acts: Act[]
  daily: DailyPoint[]
  body: BodyBlock
  thresholds: Map<Sport, ThresholdEstimate>
  today: string
  swimMetrics: ReadonlyMap<number, SwimActivityMetrics>
  runningDynamics: ReadonlyMap<number, RunningDynamicsMetrics>
  powerCurve: PowerCurveBlock
  ftp: number | null
}

export function buildAbilities(scope: AbilityScope): AbilitiesBlock {
  const {
    cache,
    acts,
    daily,
    body,
    thresholds,
    today,
    swimMetrics,
    runningDynamics,
    powerCurve,
    ftp,
  } = scope
  const kgNow = body.latestKg
  const engineTodayMs = dayMs(today)
  const bikePowerHistory = new Map(
    powerCurve.powerToWeight.points.map(point => [point.date, point]),
  )
  const bikePowerAxis = (key: 'sprint' | 'climb'): RadarAxis => {
    const reference = WORLD_TOUR_POWER_REFERENCE[key]
    const effort = bikePowerHistory.get(today)?.efforts[reference.durationS]
    const wattsPerKg = effort?.wattsPerKg ?? null
    return {
      key,
      label: key,
      score: bikePowerScore(key, wattsPerKg),
      proj: null,
      rawValue: wattsPerKg != null ? round(wattsPerKg, 2) : null,
      rawUnit: 'w/kg',
      lo: 0,
      hi: reference.wattsPerKg,
    }
  }
  const ftpWkg = ftp != null && kgNow ? ftp / kgNow : null
  const rdy14 = winValues(
    daily.map(d => d.readiness),
    daily.length - 1,
    14,
  )
  const hrv14 = winValues(
    daily.map(d => d.hrv),
    daily.length - 1,
    14,
  )
  const recScore = rdy14.length
    ? Math.round(mean(rdy14))
    : hrv14.length
      ? Math.round(norm01(mean(hrv14), 20, 120) * 100)
      : null

  const distOf = (id: number): number[] => cache.streams?.[String(id)]?.distance ?? []
  const peakRunSpeedOf = (x: Act): number | null => {
    const d = distOf(x.a.id)
    if (!oneHz(d.length, x.a.movingTime)) return null
    const v = peakDistRate(d, RUN_SPRINT_WIN_S)
    return v != null && v <= SPRINT_CAP_MS.run ? v : null
  }

  const buildSportAbilities = (sport: Sport): SportAbilities => {
    const mine = acts.filter(x => x.sport === sport)
    const ctlHi = round(CTL_ANCHOR[1] * LOAD_SHARE_TARGET[sport], 0)
    const ctlSport = daily.length ? sportCtlOf(daily[daily.length - 1], sport) : 0
    const th = thresholds.get(sport)
    const vThrSport = th && th.conf !== 'prior' ? th.vThr : null
    const swimPaces =
      sport === 'swim'
        ? mine.map(x => swimMetrics.get(x.a.id)?.paceSPer100m).filter((v): v is number => v != null)
        : []
    const bestSwimPace = swimPaces.length ? Math.min(...swimPaces) : null
    const nativeStride =
      sport === 'run'
        ? mine
            .map(x => ({ day: x.day, value: runningDynamics.get(x.a.id)?.strideLengthM ?? null }))
            .filter((point): point is { day: string; value: number } => point.value != null)
        : []
    const estimatedStride =
      sport === 'run'
        ? mine
            .map(x => {
              const cadence = cadValOf('run', x.a)
              const value =
                cadence != null && cadence > 0 ? (x.a.averageSpeed * 60) / cadence : null
              return {
                day: x.day,
                value: value != null && value >= 0.2 && value <= 3 ? value : null,
              }
            })
            .filter((point): point is { day: string; value: number } => point.value != null)
        : []
    const stride = nativeStride.length > 0 ? nativeStride : estimatedStride
    const oscillation =
      sport === 'run'
        ? mine
            .map(x => ({
              day: x.day,
              value: runningDynamics.get(x.a.id)?.verticalOscillationCm ?? null,
            }))
            .filter((point): point is { day: string; value: number } => point.value != null)
        : []
    const strideBounds = metricBounds(stride.map(point => point.value))
    const oscillationBounds = metricBounds(oscillation.map(point => point.value))
    const currentStride = rollingMetricMean(stride, today)
    const currentOscillation = rollingMetricMean(oscillation, today)

    let sprint: RadarAxis
    let threshold: RadarAxis
    if (sport === 'bike') {
      sprint = bikePowerAxis('sprint')
      threshold = {
        key: 'threshold',
        label: 'threshold',
        proj: null,
        score: ftpWkg != null ? Math.round(norm01(ftpWkg, ...COGGAN_FTP_WKG) * 100) : null,
        rawValue: ftpWkg != null ? round(ftpWkg, 2) : null,
        rawUnit: 'w/kg',
        lo: COGGAN_FTP_WKG[0],
        hi: COGGAN_FTP_WKG[1],
      }
    } else {
      let vs: number | null = null
      if (sport === 'run')
        for (const x of mine) {
          const v = peakRunSpeedOf(x)
          if (v != null && (vs == null || v > vs)) vs = v
        }
      if (vs == null)
        for (const x of mine) {
          const v = plausibleVgap(sport, x)
          if (v != null && (vs == null || v > vs)) vs = v
        }
      const sAnchor = sport === 'swim' ? SWIM_SPEED_MS : RUN_SPRINT_MS
      sprint = {
        key: 'sprint',
        label: 'sprint',
        proj: null,
        score:
          vs != null
            ? sport === 'swim'
              ? swimSpeedScoreOf(vs)
              : Math.round(norm01(vs, ...sAnchor) * 100)
            : null,
        rawValue: vs != null ? round(vs, 2) : null,
        rawUnit: 'm/s',
        lo: sAnchor[0],
        hi: sAnchor[1],
      }
      const tAnchor = sport === 'swim' ? SWIM_SPEED_MS : RUN_THR_MS
      threshold = {
        key: 'threshold',
        label: 'threshold',
        proj: null,
        score:
          vThrSport != null
            ? sport === 'swim'
              ? swimSpeedScoreOf(vThrSport)
              : Math.round(norm01(vThrSport, ...tAnchor) * 100)
            : null,
        rawValue: vThrSport != null ? round(vThrSport, 2) : null,
        rawUnit: 'm/s',
        lo: tAnchor[0],
        hi: tAnchor[1],
      }
    }

    let fourth: RadarAxis
    if (sport === 'swim') {
      fourth = {
        key: 'climb',
        label: 'pace',
        proj: null,
        score: bestSwimPace != null ? swimPaceScoreOf(bestSwimPace) : null,
        rawValue: bestSwimPace != null ? round(bestSwimPace, 1) : null,
        rawUnit: 's/100m',
        lo: SWIM_PACE_MIN_S_PER_100M,
        hi: SWIM_PACE_MAX_S_PER_100M,
      }
    } else if (sport === 'bike') {
      fourth = bikePowerAxis('climb')
    } else {
      fourth = {
        key: 'stride',
        label: nativeStride.length > 0 ? 'stride length' : 'estimated stride length',
        proj: null,
        score: personalMetricScore(currentStride, strideBounds, false),
        rawValue: currentStride != null ? round(currentStride, 2) : null,
        rawUnit: 'm',
        lo: strideBounds?.[0] ?? 0,
        hi: strideBounds?.[1] ?? 0,
      }
    }

    const cads = mine
      .map(x => (sport === 'swim' ? swimMetrics.get(x.a.id)?.strokeRateSpm : cadValOf(sport, x.a)))
      .filter((v): v is number => v != null)
    const sixth: RadarAxis =
      sport === 'run'
        ? {
            key: 'oscillation',
            label: 'vertical oscillation',
            proj: null,
            score: personalMetricScore(currentOscillation, oscillationBounds, true),
            rawValue: currentOscillation != null ? round(currentOscillation, 1) : null,
            rawUnit: 'cm',
            lo: oscillationBounds?.[0] ?? 0,
            hi: oscillationBounds?.[1] ?? 0,
          }
        : {
            key: 'recovery',
            label: 'recovery',
            proj: null,
            score: recScore,
            rawValue: rdy14.length
              ? round(mean(rdy14), 0)
              : hrv14.length
                ? round(mean(hrv14), 0)
                : null,
            rawUnit: rdy14.length ? 'readiness' : 'ms',
            lo: 0,
            hi: 100,
          }
    const axes: RadarAxis[] = [
      sprint,
      threshold,
      {
        key: 'endurance',
        label: 'endurance',
        proj: null,
        score: Math.round(norm01(ctlSport, CTL_ANCHOR[0], ctlHi) * 100),
        rawValue: round(ctlSport, 0),
        rawUnit: 'ctl',
        lo: CTL_ANCHOR[0],
        hi: ctlHi,
      },
      fourth,
      {
        key: 'cadence',
        label: sport === 'swim' ? 'stroke rate' : 'cadence',
        proj: null,
        score: cads.length ? cadScoreOf(sport, mean(cads)) : null,
        rawValue: cads.length ? round(mean(cads), sport === 'swim' ? 1 : 0) : null,
        rawUnit: CAD_UNIT[sport],
        lo: 0,
        hi: 100,
      },
      sixth,
    ]
    const scored = axes.filter(a => a.score != null).map(a => a.score as number)
    const area = scored.length ? Math.round(mean(scored)) : null

    const stats = mine
      .map(x => ({
        day: x.day,
        start: x.a.startDate,
        id: x.a.id,
        sprint:
          sport === 'bike'
            ? null
            : sport === 'run'
              ? (peakRunSpeedOf(x) ?? plausibleVgap(sport, x))
              : plausibleVgap(sport, x),
        fourth: sport === 'swim' ? swimMetrics.get(x.a.id)?.paceSPer100m : null,
        fifth: sport === 'swim' ? swimMetrics.get(x.a.id)?.strokeRateSpm : cadValOf(sport, x.a),
      }))
      .sort((p, q) => p.start.localeCompare(q.start) || p.id - q.id)
    const sprintScoreOf = (v: number): number | null =>
      sport === 'swim' ? swimSpeedScoreOf(v) : Math.round(norm01(v, ...RUN_SPRINT_MS) * 100)
    const rdyArr = daily.map(d => d.readiness)
    const history: AbilityTrendPoint[] = []
    let si = 0
    let cumSprint: number | null = null
    let cumFourth: number | null = null
    const fifthAcc: number[] = []
    for (let i = 0; i < daily.length; i++) {
      const cutoff = daily[i].date
      while (si < stats.length && stats[si].day <= cutoff) {
        const st = stats[si]
        if (st.sprint != null && (cumSprint == null || st.sprint > cumSprint)) cumSprint = st.sprint
        if (
          st.fourth != null &&
          (cumFourth == null || (sport === 'swim' ? st.fourth < cumFourth : st.fourth > cumFourth))
        )
          cumFourth = st.fourth
        if (st.fifth != null) fifthAcc.push(st.fifth)
        si++
      }
      const rdyH = winValues(rdyArr, i, 14)
      const point: AbilityTrendPoint = {
        date: cutoff,
        sprint:
          sport === 'bike'
            ? bikePowerScore('sprint', bikePowerHistory.get(cutoff)?.efforts[5]?.wattsPerKg ?? null)
            : cumSprint != null
              ? sprintScoreOf(cumSprint)
              : null,
        threshold: threshold.score,
        endurance: Math.round(norm01(sportCtlOf(daily[i], sport), CTL_ANCHOR[0], ctlHi) * 100),
        cadence: fifthAcc.length ? cadScoreOf(sport, mean(fifthAcc)) : null,
      }
      if (sport === 'run') {
        point.stride = personalMetricScore(rollingMetricMean(stride, cutoff), strideBounds, false)
        point.oscillation = personalMetricScore(
          rollingMetricMean(oscillation, cutoff),
          oscillationBounds,
          true,
        )
      } else {
        point.climb =
          sport === 'bike'
            ? bikePowerScore(
                'climb',
                bikePowerHistory.get(cutoff)?.efforts[1200]?.wattsPerKg ?? null,
              )
            : cumFourth != null
              ? swimPaceScoreOf(cumFourth)
              : null
        point.recovery = rdyH.length ? Math.round(mean(rdyH)) : null
      }
      history.push(point)
    }
    const recent = history.slice(-PROJ_WINDOW_D)
    for (const a of axes) {
      if (a.score == null) continue
      if (a.key === 'sprint' || a.key === 'climb') {
        a.proj = a.score
        if (sport === 'bike') {
          const durationS = WORLD_TOUR_POWER_REFERENCE[a.key].durationS
          const current = bikePowerHistory.get(today)?.efforts[durationS]
          const observations = recent.flatMap(day => {
            const effort = bikePowerHistory.get(day.date)?.efforts[durationS]
            return effort == null ? [] : [{ date: day.date, effort }]
          })
          // Carrying one record across days does not supply independent evidence of improvement.
          const efforts = new Set(observations.map(point => point.effort.activityId))
          if (current != null && observations.length >= PROJ_MIN_POINTS && efforts.size >= 3) {
            const slope =
              olsSlope(
                observations.map(point => (dayMs(point.date) - engineTodayMs) / DAY_MS),
                observations.map(point => point.effort.wattsPerKg),
              ) ?? 0
            const projected = Math.max(0, current.wattsPerKg + slope * PROJ_HORIZON_D)
            a.proj = bikePowerScore(a.key, projected)
            a.projRawValue = round(projected, 2)
          }
        }
        continue
      }
      const xs: number[] = []
      const ys: number[] = []
      recent.forEach((h, i) => {
        const v = h[a.key]
        if (v != null) {
          xs.push(i)
          ys.push(v)
        }
      })
      if (ys.length < PROJ_MIN_POINTS) {
        a.proj = a.score
        continue
      }
      const slope = olsSlope(xs, ys) ?? 0
      a.proj = Math.round(clamp(a.score + slope * PROJ_HORIZON_D, 0, 100))
    }
    return { sport, axes, area, history }
  }
  return { sports: SPORT_ORDER.map(buildSportAbilities) }
}
