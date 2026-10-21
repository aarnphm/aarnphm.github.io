import type { StravaActivityDetail } from '../../plugins/stores/strava'

const TRAINING_EFFECT_LABELS: Record<string, string> = {
  RECOVERY: 'recovery',
  AEROBIC_BASE: 'base',
  TEMPO: 'tempo',
  LACTATE_THRESHOLD: 'threshold',
  THRESHOLD: 'threshold',
  VO2_MAX: 'VO2max',
  VO2MAX: 'VO2max',
  ANAEROBIC_CAPACITY: 'anaerobic capacity',
  ANAEROBIC: 'anaerobic',
  SPEED: 'sprint',
  SPRINT: 'sprint',
}

export const formatTrainingEffectLabel = (value: string | null): string | null => {
  const key = value?.trim().toUpperCase()
  if (!key) return null
  return TRAINING_EFFECT_LABELS[key] ?? key.replaceAll('_', ' ').toLowerCase()
}

export const activityTrainingEffectLabel = (d: StravaActivityDetail): string => {
  if (d.sport === 'sauna') return 'recovery'
  const garminLabel = formatTrainingEffectLabel(d.garmin?.trainingEffectLabel ?? null)
  if (garminLabel) return garminLabel
  if (d.sport === 'strength' || d.sport === 'yoga' || d.sport === 'treatment') return 'recovery'

  const calculated = d.calculatedTrainingEffect
  if (
    !calculated ||
    d.garmin?.aerobicTrainingEffect != null ||
    d.garmin?.anaerobicTrainingEffect != null ||
    !Number.isFinite(calculated.aerobic) ||
    !Number.isFinite(calculated.anaerobic)
  )
    return 'base'

  const anaerobicEvidence = calculated.evidence.anaerobic
  if (
    anaerobicEvidence.source !== 'heart-rate' &&
    calculated.anaerobic >= 3 &&
    calculated.anaerobic >= calculated.aerobic - 0.5
  ) {
    if (
      anaerobicEvidence.source === 'power' &&
      anaerobicEvidence.sprintEffortCount != null &&
      anaerobicEvidence.sprintEffortCount >= 3 &&
      anaerobicEvidence.sprintStimulus != null &&
      anaerobicEvidence.stimulus > 0 &&
      anaerobicEvidence.sprintStimulus / anaerobicEvidence.stimulus >= 0.5
    )
      return 'sprint'
    return 'anaerobic capacity'
  }

  const powerIntensity = d.calculatedIntensityFactor
  const ftpWatts =
    powerIntensity?.source === 'power' &&
    Number.isFinite(powerIntensity.value) &&
    powerIntensity.value > 0 &&
    d.npWatts != null
      ? d.npWatts / powerIntensity.value
      : null
  if (
    calculated.aerobic >= 3 &&
    d.sport === 'bike' &&
    d.powerZones?.length === 7 &&
    ftpWatts != null &&
    Number.isFinite(ftpWatts) &&
    ftpWatts > 0
  ) {
    const zoneSeconds = d.powerZones
    const movingTimeS = d.movingTimeS
    const hasSustainedEffort = (durationS: number, ftpFraction: number): boolean =>
      d.bestEfforts?.power.some(
        effort => effort.durationS >= durationS && effort.averageWatts >= ftpWatts * ftpFraction,
      ) ?? false
    if (
      zoneSeconds[4] >= 480 &&
      zoneSeconds[4] >= movingTimeS * 0.1 &&
      hasSustainedEffort(300, 1.05)
    )
      return 'VO2max'
    if (
      zoneSeconds[3] >= 900 &&
      zoneSeconds[3] >= movingTimeS * 0.2 &&
      hasSustainedEffort(1_200, 0.9)
    )
      return 'threshold'
    if (
      zoneSeconds[2] >= 1_200 &&
      zoneSeconds[2] >= movingTimeS * 0.25 &&
      hasSustainedEffort(1_800, 0.75)
    )
      return 'tempo'
  }

  const heartRateSeconds = d.hrZones?.reduce((sum, seconds) => sum + seconds, 0) ?? 0
  const recoveryHeartRate =
    d.movingTimeS > 0 &&
    d.hrZones?.length === 5 &&
    d.hrZones.every(seconds => Number.isFinite(seconds) && seconds >= 0) &&
    heartRateSeconds >= d.movingTimeS * 0.8 &&
    d.hrZones[0] >= heartRateSeconds * 0.9
  const recoveryIntensity =
    d.calculatedIntensityFactor?.source !== 'heart-rate' &&
    d.calculatedIntensityFactor?.value != null &&
    d.calculatedIntensityFactor.value < 0.65
  if (
    calculated.aerobic >= 1 &&
    calculated.aerobic < 2 &&
    calculated.anaerobic < 1 &&
    (recoveryIntensity || recoveryHeartRate)
  )
    return 'recovery'
  return 'base'
}

export const formatTrainingEffectNote = (value: string | null): string | null => {
  const key = value?.trim()
  if (!key) return null
  return key
    .replace(/_\d+$/, '')
    .replaceAll('_', ' ')
    .toLowerCase()
    .replace(/\bvo2 max\b/g, 'VO2max')
}

type TrainingEffectGroup = 'low-aerobic' | 'high-aerobic' | 'anaerobic'

export const dominantTrainingEffectGroup = (dominant: string | null): TrainingEffectGroup => {
  const key = dominant?.trim().toUpperCase() ?? ''
  if (/ANAEROBIC|SPEED|SPRINT/.test(key)) return 'anaerobic'
  return /TEMPO|THRESHOLD|VO2/.test(key) ? 'high-aerobic' : 'low-aerobic'
}

export const trainingEffectGroup = (
  effect: 'aerobic' | 'anaerobic',
  message: string | null,
  dominant: string | null,
): TrainingEffectGroup => {
  if (effect === 'anaerobic') return 'anaerobic'
  const key = `${message ?? ''} ${dominant ?? ''}`.toUpperCase()
  return /TEMPO|THRESHOLD|VO2/.test(key) ? 'high-aerobic' : 'low-aerobic'
}

export const trainingEffectScore = (score: number | null): number | null =>
  score == null || !Number.isFinite(score) ? null : Math.min(5, Math.max(0, score))

const fallbackTrainingEffectNote = (
  effect: 'aerobic' | 'anaerobic',
  score: number | null,
): string | null => {
  const value = trainingEffectScore(score)
  if (value == null) return null
  if (value < 1) return `no ${effect} benefit`
  if (value < 2) return `minor ${effect} benefit`
  if (value < 3) return `maintaining ${effect} fitness`
  if (value < 4) return `improving ${effect} fitness`
  if (value < 5) return `highly improving ${effect} fitness`
  return `overreaching ${effect} effect`
}

export const trainingEffectNote = (
  effect: 'aerobic' | 'anaerobic',
  score: number | null,
  message: string | null,
): string | null => formatTrainingEffectNote(message) ?? fallbackTrainingEffectNote(effect, score)
