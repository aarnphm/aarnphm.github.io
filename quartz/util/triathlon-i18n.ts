import type { SwimChartMetric } from './swim-metrics'
import type { Gloss } from './triathlon-i18n/types'
import type { Locale } from './triathlon-presentation'
import * as activityCharts from './triathlon-i18n/activity-charts'
import * as activityGraphNotes from './triathlon-i18n/activity-graph-notes'
import * as activityGraphs from './triathlon-i18n/activity-graphs'
import * as dailyAnalytics from './triathlon-i18n/daily-analytics'
import * as estimatesNavigation from './triathlon-i18n/estimates-navigation'
import * as feedGear from './triathlon-i18n/feed-gear'
import * as glossBodyRecovery from './triathlon-i18n/gloss-body-recovery'
import * as glossPerformance from './triathlon-i18n/gloss-performance'
import * as glossTrainingLoad from './triathlon-i18n/gloss-training-load'
import * as healthStatus from './triathlon-i18n/health-status'
import * as matchedRoutes from './triathlon-i18n/matched-routes'
import * as metrics from './triathlon-i18n/metrics'
import * as powerCurveCompare from './triathlon-i18n/power-curve-compare'
import * as raceCalendar from './triathlon-i18n/race-calendar'
import * as radarBestEfforts from './triathlon-i18n/radar-best-efforts'
import * as sleep from './triathlon-i18n/sleep'
import * as trainingCalendar from './triathlon-i18n/training-calendar'
import * as weather from './triathlon-i18n/weather'
import * as workoutNotation from './triathlon-i18n/workout-notation'

interface TriDict {
  ui: Record<string, string>
  gloss: Record<string, Gloss>
}

export type SwimActivityTextPoint = {
  elapsed: string
  cumulativeDistanceM: number
  windowStartDistanceM?: number
}

const swimTextNumber = (target: Locale, value: number, maximumFractionDigits = 0): string =>
  value.toLocaleString(target === 'fr' ? 'fr-CA' : 'en-US', {
    minimumFractionDigits: 0,
    maximumFractionDigits,
  })

export const swimActivityDistanceText = (target: Locale, distanceM: number): string =>
  `${swimTextNumber(target, distanceM)} m`

export const swimActivityHeaderValue = (
  target: Locale,
  kind: SwimChartMetric,
  value: number,
  pace: string,
): string =>
  kind === 'pace'
    ? pace
    : swimTextNumber(target, value, kind === 'cadence' || kind === 'rate' ? 1 : 0)

export const swimActivityPointText = (target: Locale, point: SwimActivityTextPoint): string => {
  const end = swimTextNumber(target, point.cumulativeDistanceM)
  const distance =
    point.windowStartDistanceM == null
      ? `${end} m`
      : `${swimTextNumber(target, point.windowStartDistanceM)}–${end} m`
  return `${distance} · ${point.elapsed} ${target === 'fr' ? 'écoulé' : 'elapsed'}`
}

export const swimActivityDisplayValue = (
  target: Locale,
  kind: SwimChartMetric,
  value: number,
  pace: string,
): string =>
  kind === 'pace'
    ? `${pace} /100m`
    : kind === 'cadence'
      ? `${swimTextNumber(target, value, 1)} ${target === 'fr' ? 'coups/longueur' : 'str/length'}`
      : kind === 'rate'
        ? `${swimTextNumber(target, value, 1)} ${target === 'fr' ? 'coups/min' : 'spm'}`
        : `${swimTextNumber(target, value)} SWOLF`

export const swimActivityValueText = (
  target: Locale,
  kind: SwimChartMetric,
  point: SwimActivityTextPoint,
  value: number,
  pace: string,
): string => {
  const end = swimTextNumber(target, point.cumulativeDistanceM)
  const position =
    point.windowStartDistanceM == null
      ? target === 'fr'
        ? `${end} mètres, temps écoulé ${point.elapsed}`
        : `${end} metres, ${point.elapsed} elapsed`
      : target === 'fr'
        ? `bloc de ${swimTextNumber(target, point.cumulativeDistanceM - point.windowStartDistanceM)} mètres, de ${swimTextNumber(target, point.windowStartDistanceM)} à ${end} mètres, temps écoulé ${point.elapsed}`
        : `${swimTextNumber(target, point.cumulativeDistanceM - point.windowStartDistanceM)} metre block from ${swimTextNumber(target, point.windowStartDistanceM)} to ${end} metres, ${point.elapsed} elapsed`
  if (kind === 'pace')
    return target === 'fr'
      ? `${position}, allure de nage ${pace} par 100 mètres`
      : `${position}, swim pace ${pace} per 100 metres`
  if (kind === 'cadence') {
    const cadence = swimTextNumber(target, value, 1)
    return target === 'fr'
      ? `${position}, cadence de nage ${cadence} coups par longueur`
      : `${position}, swim cadence ${cadence} strokes per length`
  }
  if (kind === 'rate') {
    const rate = swimTextNumber(target, value, 1)
    return target === 'fr'
      ? `${position}, fréquence de nage ${rate} coups par minute`
      : `${position}, stroke rate ${rate} strokes per minute`
  }
  const swolf = swimTextNumber(target, value)
  return target === 'fr' ? `${position}, score SWOLF ${swolf}` : `${position}, SWOLF score ${swolf}`
}

export const detectLocale = (): Locale => {
  try {
    return navigator.language.toLowerCase().startsWith('fr') ? 'fr' : 'en'
  } catch {
    return 'en'
  }
}

// Spread order is key order, which triLabelKeys depends on when two keys share a label.
const en: TriDict = {
  ui: {
    ...activityGraphs.en,
    ...activityGraphNotes.en,
    ...metrics.en,
    ...dailyAnalytics.en,
    ...healthStatus.en,
    ...activityCharts.en,
    ...powerCurveCompare.en,
    ...matchedRoutes.en,
    ...estimatesNavigation.en,
    ...trainingCalendar.en,
    ...raceCalendar.en,
    ...feedGear.en,
    ...weather.en,
    ...sleep.en,
    ...radarBestEfforts.en,
  },
  gloss: { ...glossTrainingLoad.en, ...glossBodyRecovery.en, ...glossPerformance.en },
}

const fr: TriDict = {
  ui: {
    ...workoutNotation.fr,
    ...activityGraphs.fr,
    ...activityGraphNotes.fr,
    ...metrics.fr,
    ...dailyAnalytics.fr,
    ...healthStatus.fr,
    ...activityCharts.fr,
    ...powerCurveCompare.fr,
    ...matchedRoutes.fr,
    ...estimatesNavigation.fr,
    ...trainingCalendar.fr,
    ...raceCalendar.fr,
    ...feedGear.fr,
    ...weather.fr,
    ...sleep.fr,
    ...radarBestEfforts.fr,
  },
  gloss: { ...glossTrainingLoad.fr, ...glossBodyRecovery.fr, ...glossPerformance.fr },
}

const TRI_I18N: Record<Locale, TriDict> = { en, fr }

export const triText = (target: Locale, key: string): string => TRI_I18N[target].ui[key] ?? key

const triLabelKeys = new Map<string, string>()
for (const dictionary of Object.values(TRI_I18N))
  for (const [key, label] of Object.entries(dictionary.ui)) {
    triLabelKeys.set(key, key)
    triLabelKeys.set(label, key)
  }

/** A share label can come from either the serialized payload or an already-localized table. */
export const triLabel = (target: Locale, label: string): string => {
  const lap = /^(?:lap|tour) (\d+)$/.exec(label)
  if (lap) return `${triText(target, 'lap')} ${lap[1]}`
  return label
    .split(' · ')
    .map(part => triText(target, triLabelKeys.get(part) ?? part))
    .join(' · ')
}

type Vo2SourceMethod = 'garmin' | 'apple' | 'bike' | 'run' | 'hrratio' | 'lab' | 'none'

type Vo2BikeSourceText = {
  ftpW: number
  ftpSource: 'athlete' | 'strava' | 'derived'
  mapW: number
  weightKg: number
  labBaseline: { date: string; vo2max: number; ftpW: number; weightKg: number } | null
}

export const vo2SourceText = (
  target: Locale,
  method: Vo2SourceMethod,
  bike: Vo2BikeSourceText | null,
): string => {
  if (method === 'garmin')
    return target === 'fr'
      ? "Garmin Connect ou d'une saisie manuelle."
      : 'Garmin Connect or manual entry.'
  if (method === 'apple')
    return target === 'fr' ? "Cette mesure vient de l'Apple Watch." : 'Apple Watch measurement.'
  if (method === 'run')
    return target === 'fr'
      ? 'Cette estimation utilise la vitesse de course et la fréquence cardiaque.'
      : 'This estimate uses running speed and heart rate.'
  if (method === 'hrratio')
    return target === 'fr'
      ? 'Cette estimation utilise les fréquences cardiaques maximale et au repos.'
      : 'This estimate uses maximum and resting heart rate.'
  if (method === 'lab')
    return target === 'fr'
      ? "Cette valeur vient d'un test d'effort progressif."
      : 'This value comes from a graded exercise test.'
  if (method === 'bike' && bike != null) {
    const weight = bike.weightKg.toLocaleString(target === 'fr' ? 'fr-CA' : 'en-US', {
      maximumFractionDigits: 1,
    })
    const source =
      bike.ftpSource === 'athlete'
        ? target === 'fr'
          ? 'athlète'
          : 'athlete'
        : bike.ftpSource === 'strava'
          ? 'Strava'
          : target === 'fr'
            ? 'estimée'
            : 'estimated'
    const calculation =
      target === 'fr'
        ? `FTP ${bike.ftpW} W (${source}). La puissance aérobie maximale estimée est de ${bike.mapW} W. Le poids est de ${weight} kg.`
        : `FTP ${bike.ftpW} W (${source}). Estimated maximum aerobic power is ${bike.mapW} W. Body weight is ${weight} kg.`
    if (bike.labBaseline == null) return calculation
    const baselineVo2max = bike.labBaseline.vo2max.toLocaleString(
      target === 'fr' ? 'fr-CA' : 'en-US',
      { maximumFractionDigits: 1 },
    )
    const baselineWeight = bike.labBaseline.weightKg.toLocaleString(
      target === 'fr' ? 'fr-CA' : 'en-US',
      { maximumFractionDigits: 1 },
    )
    return target === 'fr'
      ? `${calculation} La référence du laboratoire est une VO₂max de ${baselineVo2max}, une FTP de ${bike.labBaseline.ftpW} W et un poids de ${baselineWeight} kg.`
      : `${calculation} The lab baseline is VO₂max ${baselineVo2max}, FTP ${bike.labBaseline.ftpW} W, and body weight ${baselineWeight} kg.`
  }
  return target === 'fr'
    ? 'Il manque les données de puissance ou de fréquence cardiaque.'
    : 'There is no power or heart rate data.'
}

export const trendUnavailableText = (
  target: Locale,
  sampleSize: number | null,
  daysSinceLastEffort: number | null,
): string => {
  if (sampleSize === 0)
    return target === 'fr' ? "Aucun effort n'a été enregistré." : 'No efforts were recorded.'
  if (daysSinceLastEffort === 0)
    return target === 'fr'
      ? "Le dernier effort date d'aujourd'hui."
      : 'The latest effort was today.'
  if (daysSinceLastEffort != null) {
    const unit =
      target === 'fr'
        ? daysSinceLastEffort === 1
          ? 'jour'
          : 'jours'
        : daysSinceLastEffort === 1
          ? 'day'
          : 'days'
    return target === 'fr'
      ? `Le dernier effort remonte à ${daysSinceLastEffort} ${unit}.`
      : `The latest effort was ${daysSinceLastEffort} ${unit} ago.`
  }
  return target === 'fr' ? 'Données insuffisantes.' : 'Not enough data.'
}

export const powerCurveReferenceLabel = (target: Locale, year: number | null): string =>
  year == null
    ? triText(target, '6-week best')
    : target === 'fr'
      ? `meilleur de ${year}`
      : `${year} best`

type CriticalPowerSummary = {
  criticalPowerWatts: number
  wPrimeJoules: number
  independentEffortCount: number
  confidence: 'medium' | 'provisional'
}

export const criticalPowerSummaryParts = (
  target: Locale,
  estimate: CriticalPowerSummary,
): { criticalPower: string; wPrime: string; evidence: string } => {
  const cp = estimate.criticalPowerWatts.toLocaleString(target === 'fr' ? 'fr-CA' : 'en-US', {
    maximumFractionDigits: 1,
  })
  const wPrime = (estimate.wPrimeJoules / 1_000).toLocaleString(
    target === 'fr' ? 'fr-CA' : 'en-US',
    { minimumFractionDigits: 1, maximumFractionDigits: 1 },
  )
  const efforts = estimate.independentEffortCount
  if (target === 'fr')
    return {
      criticalPower: `eCP ${cp} W`,
      wPrime: `eW′ ${wPrime} kJ`,
      evidence: `${efforts} effort${efforts === 1 ? '' : 's'} indépendant${efforts === 1 ? '' : 's'} · ${estimate.confidence === 'medium' ? 'confiance moyenne' : 'provisoire'}`,
    }
  return {
    criticalPower: `eCP ${cp} W`,
    wPrime: `eW′ ${wPrime} kJ`,
    evidence: `${efforts} independent effort${efforts === 1 ? '' : 's'} · ${estimate.confidence}`,
  }
}

export const criticalPowerSummaryText = (
  target: Locale,
  estimate: CriticalPowerSummary,
): string => {
  const summary = criticalPowerSummaryParts(target, estimate)
  return `${summary.criticalPower} · ${summary.wPrime}`
}

export const criticalPowerEvidenceText = (target: Locale, estimate: CriticalPowerSummary): string =>
  criticalPowerSummaryParts(target, estimate).evidence

export const glossFor = (target: Locale, key: string): Gloss | undefined =>
  TRI_I18N[target].gloss[key] ?? en.gloss[key]

export const glossKeys = (): string[] => Object.keys(en.gloss)
