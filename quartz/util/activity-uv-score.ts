import type { GardenEnvironmentEstimate, GardenUvScore } from './activity-environment'
import type { UvSeverity } from './activity-provider-reports'
import { isRecord, readNumber, readString } from './type-guards'

export type GardenUvDoseClock = 'elapsed' | 'moving-telemetry'

export interface GardenUvCalibrationPair {
  activityId: number
  date: string
  score: number
  severity: UvSeverity
  elapsedSed: number
  movingTelemetrySed: number
}

export interface GardenUvCalibrationMetrics {
  pairCount: number
  representedBands: number
  mae: number
  meanBias: number
  bandAgreementPct: number
}

export interface GardenUvCalibrationArtifact {
  formulaId: 'garden-uv-score-v1'
  formulaVersion: 1
  calibrationVersion: 1
  inputVersion: 'pelotan-description-v1+garden-environment-v1'
  status: 'insufficient' | 'rejected' | 'active' | 'suspended'
  doseClock: GardenUvDoseClock | null
  coefficientSed: number | null
  training: GardenUvCalibrationMetrics | null
  holdout: GardenUvCalibrationMetrics | null
  activityIds: number[]
  auditedActivityIds: number[]
}

const calibrationMetrics = (value: unknown): GardenUvCalibrationMetrics | null => {
  if (!isRecord(value)) return null
  const pairCount = readNumber(value, 'pairCount')
  const bands = readNumber(value, 'representedBands')
  const mae = readNumber(value, 'mae')
  const meanBias = readNumber(value, 'meanBias')
  const bandAgreementPct = readNumber(value, 'bandAgreementPct')
  if (
    pairCount == null ||
    !Number.isInteger(pairCount) ||
    pairCount < 0 ||
    bands == null ||
    !Number.isInteger(bands) ||
    bands < 0 ||
    mae == null ||
    mae < 0 ||
    meanBias == null ||
    bandAgreementPct == null ||
    bandAgreementPct < 0 ||
    bandAgreementPct > 100
  )
    return null
  return { pairCount, representedBands: bands, mae, meanBias, bandAgreementPct }
}

const activityIds = (value: unknown): number[] | null => {
  if (!Array.isArray(value)) return null
  const ids = value.filter(item => typeof item === 'number' && Number.isInteger(item))
  return ids.length === value.length ? ids : null
}

export function parseGardenUvCalibrationArtifact(
  value: unknown,
): GardenUvCalibrationArtifact | null {
  if (!isRecord(value)) return null
  const status = readString(value, 'status')
  const doseClock = value.doseClock
  const coefficientSed = value.coefficientSed
  const calibratedIds = activityIds(value.activityIds)
  const auditedActivityIds = activityIds(value.auditedActivityIds)
  if (
    value.formulaId !== 'garden-uv-score-v1' ||
    value.formulaVersion !== 1 ||
    value.calibrationVersion !== 1 ||
    value.inputVersion !== 'pelotan-description-v1+garden-environment-v1' ||
    (status !== 'insufficient' &&
      status !== 'rejected' &&
      status !== 'active' &&
      status !== 'suspended') ||
    (doseClock !== null && doseClock !== 'elapsed' && doseClock !== 'moving-telemetry') ||
    (coefficientSed !== null &&
      (typeof coefficientSed !== 'number' ||
        !Number.isFinite(coefficientSed) ||
        coefficientSed <= 0)) ||
    !calibratedIds ||
    !auditedActivityIds
  )
    return null
  const training = value.training === null ? null : calibrationMetrics(value.training)
  const holdout = value.holdout === null ? null : calibrationMetrics(value.holdout)
  if ((value.training !== null && !training) || (value.holdout !== null && !holdout)) return null
  const fitted = doseClock != null && coefficientSed != null && training != null && holdout != null
  const uniqueCalibratedIds = new Set(calibratedIds)
  const uniqueAuditedIds = new Set(auditedActivityIds)
  if (
    uniqueCalibratedIds.size !== calibratedIds.length ||
    uniqueAuditedIds.size !== auditedActivityIds.length ||
    auditedActivityIds.some(id => uniqueCalibratedIds.has(id)) ||
    (status === 'insufficient' &&
      (doseClock != null || coefficientSed != null || training != null || holdout != null)) ||
    (status !== 'insufficient' && !fitted) ||
    (fitted && (training.pairCount < 20 || holdout.pairCount !== 10)) ||
    (status === 'active' && fitted && !passesGate(holdout, HOLDOUT_MINIMUM_BANDS)) ||
    ((status === 'rejected' || status === 'suspended') &&
      fitted &&
      passesGate(holdout, HOLDOUT_MINIMUM_BANDS))
  )
    return null
  return {
    formulaId: 'garden-uv-score-v1',
    formulaVersion: 1,
    calibrationVersion: 1,
    inputVersion: 'pelotan-description-v1+garden-environment-v1',
    status,
    doseClock,
    coefficientSed,
    training,
    holdout,
    activityIds: calibratedIds,
    auditedActivityIds,
  }
}

const SCORE_MINIMUM_PAIRS = 30
const SCORE_HOLDOUT_PAIRS = 10
const SCORE_MINIMUM_TRAINING_PAIRS = 20
const SCORE_MINIMUM_BANDS = 4
const HOLDOUT_MINIMUM_BANDS = 3
const MAXIMUM_MAE = 5
const MAXIMUM_ABSOLUTE_BIAS = 3
const MINIMUM_BAND_AGREEMENT_PCT = 90

const round = (value: number, digits = 3): number => {
  const factor = 10 ** digits
  return Math.round(value * factor) / factor
}

export function gardenUvScoreFromDose(doseSed: number, coefficientSed: number): number {
  if (!Number.isFinite(doseSed) || doseSed < 0) return 0
  if (!Number.isFinite(coefficientSed) || coefficientSed <= 0) return 0
  return Math.min(100, Math.max(0, Math.round(100 * (1 - Math.exp(-doseSed / coefficientSed)))))
}

export function gardenUvSeverity(score: number): UvSeverity {
  if (score <= 10) return 'negligible'
  if (score <= 30) return 'low'
  if (score <= 60) return 'moderate'
  if (score <= 85) return 'high'
  if (score <= 95) return 'serious'
  return 'extreme'
}

const dose = (pair: GardenUvCalibrationPair, clock: GardenUvDoseClock): number =>
  clock === 'elapsed' ? pair.elapsedSed : pair.movingTelemetrySed

const representedBands = (pairs: readonly GardenUvCalibrationPair[]): number =>
  new Set(pairs.map(pair => pair.severity)).size

const validPair = (pair: GardenUvCalibrationPair): boolean =>
  Number.isInteger(pair.activityId) &&
  /^\d{4}-\d{2}-\d{2}$/.test(pair.date) &&
  Number.isFinite(pair.score) &&
  pair.score >= 0 &&
  pair.score <= 100 &&
  Number.isFinite(pair.elapsedSed) &&
  pair.elapsedSed >= 0 &&
  Number.isFinite(pair.movingTelemetrySed) &&
  pair.movingTelemetrySed >= 0

const sortedPairs = (pairs: readonly GardenUvCalibrationPair[]): GardenUvCalibrationPair[] => {
  const unique = new Map<number, GardenUvCalibrationPair>()
  for (const pair of pairs.filter(validPair)) unique.set(pair.activityId, pair)
  return [...unique.values()].sort(
    (left, right) => left.date.localeCompare(right.date) || left.activityId - right.activityId,
  )
}

const candidateCoefficients = (
  pairs: readonly GardenUvCalibrationPair[],
  clock: GardenUvDoseClock,
): number[] => {
  const candidates = new Set<number>()
  for (let index = 0; index <= 2_000; index += 1) candidates.add(10 ** (-2 + (index / 2_000) * 5))
  for (const pair of pairs) {
    if (pair.score <= 0 || pair.score >= 100) continue
    const coefficient = -dose(pair, clock) / Math.log(1 - pair.score / 100)
    if (!Number.isFinite(coefficient) || coefficient <= 0) continue
    candidates.add(coefficient)
    candidates.add(coefficient * 0.999)
    candidates.add(coefficient * 1.001)
  }
  return [...candidates].sort((left, right) => left - right)
}

const metrics = (
  pairs: readonly GardenUvCalibrationPair[],
  clock: GardenUvDoseClock,
  coefficientSed: number,
): GardenUvCalibrationMetrics => {
  let absoluteError = 0
  let bias = 0
  let bandMatches = 0
  for (const pair of pairs) {
    const predicted = gardenUvScoreFromDose(dose(pair, clock), coefficientSed)
    absoluteError += Math.abs(predicted - pair.score)
    bias += predicted - pair.score
    if (gardenUvSeverity(predicted) === pair.severity) bandMatches += 1
  }
  return {
    pairCount: pairs.length,
    representedBands: representedBands(pairs),
    mae: pairs.length > 0 ? round(absoluteError / pairs.length) : 0,
    meanBias: pairs.length > 0 ? round(bias / pairs.length) : 0,
    bandAgreementPct: pairs.length > 0 ? round((bandMatches / pairs.length) * 100, 1) : 0,
  }
}

const continuousMae = (
  pairs: readonly GardenUvCalibrationPair[],
  clock: GardenUvDoseClock,
  coefficientSed: number,
): number =>
  pairs.reduce(
    (total, pair) =>
      total + Math.abs(100 * (1 - Math.exp(-dose(pair, clock) / coefficientSed)) - pair.score),
    0,
  ) / pairs.length

const chooseModel = (
  training: readonly GardenUvCalibrationPair[],
): { clock: GardenUvDoseClock; coefficientSed: number; metrics: GardenUvCalibrationMetrics } => {
  const clocks: readonly GardenUvDoseClock[] = ['elapsed', 'moving-telemetry']
  let best: {
    clock: GardenUvDoseClock
    coefficientSed: number
    metrics: GardenUvCalibrationMetrics
    continuousMae: number
  } | null = null
  for (const clock of clocks)
    for (const coefficientSed of candidateCoefficients(training, clock)) {
      const candidateMetrics = metrics(training, clock, coefficientSed)
      const candidateContinuousMae = continuousMae(training, clock, coefficientSed)
      if (
        !best ||
        candidateMetrics.mae < best.metrics.mae ||
        (candidateMetrics.mae === best.metrics.mae &&
          Math.abs(candidateMetrics.meanBias) < Math.abs(best.metrics.meanBias)) ||
        (candidateMetrics.mae === best.metrics.mae &&
          Math.abs(candidateMetrics.meanBias) === Math.abs(best.metrics.meanBias) &&
          candidateContinuousMae < best.continuousMae) ||
        (candidateMetrics.mae === best.metrics.mae &&
          Math.abs(candidateMetrics.meanBias) === Math.abs(best.metrics.meanBias) &&
          candidateContinuousMae === best.continuousMae &&
          clock === best.clock &&
          coefficientSed < best.coefficientSed)
      )
        best = {
          clock,
          coefficientSed,
          metrics: candidateMetrics,
          continuousMae: candidateContinuousMae,
        }
    }
  if (!best) throw new Error('UV calibration requires at least one coefficient candidate')
  return best
}

const passesGate = (result: GardenUvCalibrationMetrics, minimumBands: number): boolean =>
  result.representedBands >= minimumBands &&
  result.mae <= MAXIMUM_MAE &&
  Math.abs(result.meanBias) <= MAXIMUM_ABSOLUTE_BIAS &&
  result.bandAgreementPct >= MINIMUM_BAND_AGREEMENT_PCT

const emptyArtifact = (pairs: readonly GardenUvCalibrationPair[]): GardenUvCalibrationArtifact => ({
  formulaId: 'garden-uv-score-v1',
  formulaVersion: 1,
  calibrationVersion: 1,
  inputVersion: 'pelotan-description-v1+garden-environment-v1',
  status: 'insufficient',
  doseClock: null,
  coefficientSed: null,
  training: null,
  holdout: null,
  activityIds: pairs.map(pair => pair.activityId),
  auditedActivityIds: [],
})

export function calibrateGardenUvScore(
  inputPairs: readonly GardenUvCalibrationPair[],
): GardenUvCalibrationArtifact {
  const pairs = sortedPairs(inputPairs)
  if (
    pairs.length < SCORE_MINIMUM_PAIRS ||
    representedBands(pairs) < SCORE_MINIMUM_BANDS ||
    pairs.length - SCORE_HOLDOUT_PAIRS < SCORE_MINIMUM_TRAINING_PAIRS
  )
    return emptyArtifact(pairs)
  const trainingPairs = pairs.slice(0, -SCORE_HOLDOUT_PAIRS)
  const holdoutPairs = pairs.slice(-SCORE_HOLDOUT_PAIRS)
  const model = chooseModel(trainingPairs)
  const holdout = metrics(holdoutPairs, model.clock, model.coefficientSed)
  return {
    formulaId: 'garden-uv-score-v1',
    formulaVersion: 1,
    calibrationVersion: 1,
    inputVersion: 'pelotan-description-v1+garden-environment-v1',
    status: passesGate(holdout, HOLDOUT_MINIMUM_BANDS) ? 'active' : 'rejected',
    doseClock: model.clock,
    coefficientSed: round(model.coefficientSed, 6),
    training: model.metrics,
    holdout,
    activityIds: pairs.map(pair => pair.activityId),
    auditedActivityIds: [],
  }
}

export function auditGardenUvCalibration(
  artifact: GardenUvCalibrationArtifact,
  inputPairs: readonly GardenUvCalibrationPair[],
): GardenUvCalibrationArtifact {
  if (
    artifact.status !== 'active' ||
    artifact.doseClock == null ||
    artifact.coefficientSed == null ||
    artifact.formulaVersion !== 1 ||
    artifact.calibrationVersion !== 1
  )
    return artifact
  const seen = new Set([...artifact.activityIds, ...artifact.auditedActivityIds])
  const newPairs = sortedPairs(inputPairs)
    .filter(pair => !seen.has(pair.activityId))
    .slice(0, 10)
  if (newPairs.length < 10) return artifact
  const audit = metrics(newPairs, artifact.doseClock, artifact.coefficientSed)
  return {
    ...artifact,
    status: passesGate(audit, HOLDOUT_MINIMUM_BANDS) ? 'active' : 'suspended',
    holdout: audit,
    auditedActivityIds: [...artifact.auditedActivityIds, ...newPairs.map(pair => pair.activityId)],
  }
}

export function applyGardenUvCalibration(
  environment: GardenEnvironmentEstimate,
  artifact: GardenUvCalibrationArtifact | null | undefined,
  computedAt: number,
): GardenUvScore | null {
  if (
    artifact?.status !== 'active' ||
    artifact.formulaId !== 'garden-uv-score-v1' ||
    artifact.formulaVersion !== 1 ||
    artifact.calibrationVersion !== 1 ||
    artifact.doseClock == null ||
    artifact.coefficientSed == null
  )
    return null
  const doseSed =
    artifact.doseClock === 'elapsed'
      ? environment.doseClocks.elapsedSed
      : environment.doseClocks.movingTelemetrySed
  if (doseSed == null) return null
  const score = gardenUvScoreFromDose(doseSed, artifact.coefficientSed)
  return {
    source: 'garden-estimate',
    formulaId: 'garden-uv-score-v1',
    formulaVersion: 1,
    inputVersion: environment.inputVersion,
    normalizationVersion: environment.normalizationVersion,
    computedAt,
    inputAsOf: environment.inputAsOf,
    temporalSamplingModel: environment.temporalSamplingModel,
    spatialSamplingModel: environment.spatialSamplingModel,
    score,
    severity: gardenUvSeverity(score),
    doseClock: artifact.doseClock,
    doseSed,
    coefficientSed: artifact.coefficientSed,
    calibrationVersion: 1,
  }
}
