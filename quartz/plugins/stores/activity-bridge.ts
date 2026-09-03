import type { Sport } from './strava'

const START_TOLERANCE_MS = 20 * 60 * 1000
const DISTANCE_TOLERANCE_RATIO = 0.08
const DISTANCE_TOLERANCE_M = 1500
const DURATION_TOLERANCE_RATIO = 0.12
const DURATION_TOLERANCE_S = 10 * 60

export type ActivityBridgeProvider = 'garmin' | 'wahoo'
export type ActivityBridgeDirection = 'garmin-to-wahoo' | 'wahoo-to-garmin'
export type ActivityBridgeUploadStatus =
  | 'pending'
  | 'in_progress'
  | 'complete'
  | 'duplicate'
  | 'error'

export interface ActivityBridgeStravaActivity {
  id: string
  name: string
  sportType: string
  startDate: string
  startDateLocal: string
  distanceM: number
  movingTimeS: number
  elapsedTimeS: number
}

interface ActivityBridgeProviderActivity {
  id: string
  sport: Sport | null
  startDate: string
  startDateLocal: string
  distanceM: number | null
  movingTimeS: number | null
  elapsedTimeS: number | null
}

export interface ActivityBridgeGarminActivity extends ActivityBridgeProviderActivity {
  name: string
}

export interface ActivityBridgeWahooActivity extends ActivityBridgeProviderActivity {
  name: string
  workoutId: number
  fitUrl: string
  fitSha256: string
}

export interface ActivityBridgeInputs {
  strava: readonly ActivityBridgeStravaActivity[]
  garmin: readonly ActivityBridgeGarminActivity[]
  wahoo: readonly ActivityBridgeWahooActivity[]
}

export interface ActivityBridgeReceipt {
  direction: ActivityBridgeDirection
  sourceProvider: ActivityBridgeProvider
  sourceActivityId: string
  sourceFitSha256: string
  destinationProvider: ActivityBridgeProvider
  destinationActivityId: string | null
  stravaActivityId: string
  uploadToken: string | null
  uploadStatus: ActivityBridgeUploadStatus
  createdAt: number
  updatedAt: number
}

export interface ActivityBridgeLedger {
  version: 1
  updatedAt: number
  receipts: Record<string, ActivityBridgeReceipt>
}

interface ActivityBridgePlanBase {
  stravaActivityId: string
  title: string
}

export interface WahooToGarminBridgePlan extends ActivityBridgePlanBase {
  direction: 'wahoo-to-garmin'
  source: ActivityBridgeWahooActivity
}

export interface GarminToWahooBridgePlan extends ActivityBridgePlanBase {
  direction: 'garmin-to-wahoo'
  source: ActivityBridgeGarminActivity
}

export type ActivityBridgePlan = WahooToGarminBridgePlan | GarminToWahooBridgePlan

export type TrainingPeaksBackfillSource = 'garmin' | 'strava' | 'wahoo'

interface TrainingPeaksBackfillPlanBase {
  title: string
  localDate: string
  sport: Sport
}

export interface StravaTrainingPeaksBackfillPlan extends TrainingPeaksBackfillPlanBase {
  sourceProvider: 'strava'
  source: ActivityBridgeStravaActivity
}

export interface GarminTrainingPeaksBackfillPlan extends TrainingPeaksBackfillPlanBase {
  sourceProvider: 'garmin'
  source: ActivityBridgeGarminActivity
}

export interface WahooTrainingPeaksBackfillPlan extends TrainingPeaksBackfillPlanBase {
  sourceProvider: 'wahoo'
  source: ActivityBridgeWahooActivity
}

export type TrainingPeaksBackfillPlan =
  | StravaTrainingPeaksBackfillPlan
  | GarminTrainingPeaksBackfillPlan
  | WahooTrainingPeaksBackfillPlan

interface MatchEdge<T extends ActivityBridgeProviderActivity> {
  strava: ActivityBridgeStravaActivity
  provider: T
  score: number
}

export function emptyActivityBridgeLedger(): ActivityBridgeLedger {
  return { version: 1, updatedAt: 0, receipts: {} }
}

export function activityBridgeReceiptKey(
  sourceProvider: ActivityBridgeProvider,
  sourceActivityId: string,
  sourceFitSha256: string,
  destinationProvider: ActivityBridgeProvider,
): string {
  const sha256 = sourceFitSha256.trim().toLowerCase()
  if (!/^[a-f0-9]{64}$/.test(sha256)) throw new Error('activity bridge receipt needs a SHA-256')
  return [sourceProvider, encodeURIComponent(sourceActivityId), sha256, destinationProvider].join(
    ':',
  )
}

export function upsertActivityBridgeReceipt(
  ledger: ActivityBridgeLedger,
  receipt: ActivityBridgeReceipt,
): ActivityBridgeLedger {
  const key = activityBridgeReceiptKey(
    receipt.sourceProvider,
    receipt.sourceActivityId,
    receipt.sourceFitSha256,
    receipt.destinationProvider,
  )
  const receipts = { ...ledger.receipts, [key]: receipt }
  return {
    version: 1,
    updatedAt: receipt.updatedAt,
    receipts: Object.fromEntries(
      Object.entries(receipts).sort(([left], [right]) => left.localeCompare(right)),
    ),
  }
}

export function isTerminalActivityBridgeReceipt(receipt: ActivityBridgeReceipt): boolean {
  return receipt.uploadStatus === 'complete' || receipt.uploadStatus === 'duplicate'
}

export function activityBridgeCreatedDestinations(
  ledger: ActivityBridgeLedger,
  provider: ActivityBridgeProvider,
): ReadonlySet<string> {
  return new Set(
    Object.values(ledger.receipts)
      .filter(receipt => receipt.destinationProvider === provider)
      .flatMap(receipt =>
        receipt.destinationActivityId == null ? [] : [receipt.destinationActivityId],
      ),
  )
}

export function activityBridgeReceiptForSource(
  ledger: ActivityBridgeLedger,
  sourceProvider: ActivityBridgeProvider,
  sourceActivityId: string,
  destinationProvider: ActivityBridgeProvider,
): ActivityBridgeReceipt | null {
  const receipts = Object.values(ledger.receipts).filter(
    receipt =>
      receipt.sourceProvider === sourceProvider &&
      receipt.sourceActivityId === sourceActivityId &&
      receipt.destinationProvider === destinationProvider,
  )
  return receipts.find(isTerminalActivityBridgeReceipt) ?? receipts[0] ?? null
}

function positive(value: number | null): number | null {
  return value != null && Number.isFinite(value) && value > 0 ? value : null
}

function stravaSport(value: string): Sport | null {
  const sport = value.toLowerCase()
  if (sport.includes('ride') || sport.includes('cycling') || sport.includes('bike')) return 'bike'
  if (sport.includes('run')) return 'run'
  if (sport.includes('swim')) return 'swim'
  return null
}

function localDate(value: string): string | null {
  const match = /^(\d{4}-\d{2}-\d{2})T/.exec(value)
  return match?.[1] ?? null
}

function matchScore<T extends ActivityBridgeProviderActivity>(
  strava: ActivityBridgeStravaActivity,
  provider: T,
): number | null {
  if (provider.sport !== stravaSport(strava.sportType)) return null
  const stravaStart = Date.parse(strava.startDate)
  const providerStart = Date.parse(provider.startDate)
  if (!Number.isFinite(stravaStart) || !Number.isFinite(providerStart)) return null
  const startDiffMs = Math.abs(stravaStart - providerStart)
  if (startDiffMs > START_TOLERANCE_MS) return null
  const stravaDistance = positive(strava.distanceM)
  const providerDistance = positive(provider.distanceM)
  if (stravaDistance == null || providerDistance == null) return null
  const distanceDiffM = Math.abs(stravaDistance - providerDistance)
  const distanceRatio = distanceDiffM / stravaDistance
  if (distanceDiffM > DISTANCE_TOLERANCE_M && distanceRatio > DISTANCE_TOLERANCE_RATIO) return null
  const stravaDurations = [positive(strava.movingTimeS), positive(strava.elapsedTimeS)].filter(
    (value): value is number => value != null,
  )
  const providerDurations = [
    positive(provider.movingTimeS),
    positive(provider.elapsedTimeS),
  ].filter((value): value is number => value != null)
  if (stravaDurations.length === 0 || providerDurations.length === 0) return null
  const durationDiffS = Math.min(
    ...stravaDurations.flatMap(left => providerDurations.map(right => Math.abs(left - right))),
  )
  const durationToleranceS = Math.max(
    DURATION_TOLERANCE_S,
    strava.elapsedTimeS * DURATION_TOLERANCE_RATIO,
  )
  if (durationDiffS > durationToleranceS) return null
  return startDiffMs / 60_000 + distanceRatio * 100 + durationDiffS / 60
}

function oneToOneMatches<T extends ActivityBridgeProviderActivity>(
  strava: readonly ActivityBridgeStravaActivity[],
  provider: readonly T[],
  excludedProviderIds: ReadonlySet<string>,
): ReadonlyMap<string, T> {
  const edges: MatchEdge<T>[] = []
  for (const canonical of strava) {
    if (stravaSport(canonical.sportType) == null) continue
    for (const candidate of provider) {
      if (excludedProviderIds.has(candidate.id)) continue
      const score = matchScore(canonical, candidate)
      if (score != null) edges.push({ strava: canonical, provider: candidate, score })
    }
  }
  edges.sort(
    (left, right) =>
      left.score - right.score ||
      left.strava.id.localeCompare(right.strava.id) ||
      left.provider.id.localeCompare(right.provider.id),
  )
  const matchedStrava = new Set<string>()
  const matchedProvider = new Set<string>()
  const matches = new Map<string, T>()
  for (const edge of edges) {
    if (matchedStrava.has(edge.strava.id) || matchedProvider.has(edge.provider.id)) continue
    matchedStrava.add(edge.strava.id)
    matchedProvider.add(edge.provider.id)
    matches.set(edge.strava.id, edge.provider)
  }
  return matches
}

export function planActivityBridge(
  inputs: ActivityBridgeInputs,
  ledger: ActivityBridgeLedger,
): ActivityBridgePlan[] {
  const canonical = inputs.strava.filter(activity => stravaSport(activity.sportType) != null)
  const garminMatches = oneToOneMatches(
    canonical,
    inputs.garmin,
    activityBridgeCreatedDestinations(ledger, 'garmin'),
  )
  const wahooMatches = oneToOneMatches(
    canonical,
    inputs.wahoo,
    activityBridgeCreatedDestinations(ledger, 'wahoo'),
  )
  const plans: ActivityBridgePlan[] = []
  for (const strava of canonical) {
    const garmin = garminMatches.get(strava.id)
    const wahoo = wahooMatches.get(strava.id)
    if (wahoo && !garmin) {
      const receipt = activityBridgeReceiptForSource(ledger, 'wahoo', wahoo.id, 'garmin')
      if (!receipt || !isTerminalActivityBridgeReceipt(receipt))
        plans.push({
          direction: 'wahoo-to-garmin',
          stravaActivityId: strava.id,
          title: strava.name,
          source: wahoo,
        })
    }
    if (garmin && !wahoo) {
      const receipt = activityBridgeReceiptForSource(ledger, 'garmin', garmin.id, 'wahoo')
      if (!receipt || !isTerminalActivityBridgeReceipt(receipt))
        plans.push({
          direction: 'garmin-to-wahoo',
          stravaActivityId: strava.id,
          title: strava.name,
          source: garmin,
        })
    }
  }
  const directionOrder: Record<ActivityBridgeDirection, number> = {
    'wahoo-to-garmin': 0,
    'garmin-to-wahoo': 1,
  }
  const receiptOrder = (plan: ActivityBridgePlan): number => {
    const sourceProvider = plan.direction === 'wahoo-to-garmin' ? 'wahoo' : 'garmin'
    const destinationProvider = plan.direction === 'wahoo-to-garmin' ? 'garmin' : 'wahoo'
    return activityBridgeReceiptForSource(
      ledger,
      sourceProvider,
      plan.source.id,
      destinationProvider,
    )
      ? 0
      : 1
  }
  return plans.sort(
    (left, right) =>
      directionOrder[left.direction] - directionOrder[right.direction] ||
      receiptOrder(left) - receiptOrder(right) ||
      right.stravaActivityId.localeCompare(left.stravaActivityId),
  )
}

export function planTrainingPeaksBackfill(
  inputs: ActivityBridgeInputs,
  ledger: ActivityBridgeLedger,
  sourceProvider: TrainingPeaksBackfillSource,
): TrainingPeaksBackfillPlan[] {
  const plans: TrainingPeaksBackfillPlan[] = []
  if (sourceProvider === 'strava') {
    for (const strava of inputs.strava) {
      const sport = stravaSport(strava.sportType)
      const day = localDate(strava.startDateLocal)
      if (sport == null || day == null) continue
      plans.push({ sourceProvider, title: strava.name, localDate: day, sport, source: strava })
    }
  } else if (sourceProvider === 'garmin') {
    const mirrored = activityBridgeCreatedDestinations(ledger, 'garmin')
    for (const garmin of inputs.garmin) {
      const day = localDate(garmin.startDateLocal)
      if (garmin.sport == null || day == null || mirrored.has(garmin.id)) continue
      plans.push({
        sourceProvider,
        title: garmin.name,
        localDate: day,
        sport: garmin.sport,
        source: garmin,
      })
    }
  } else {
    const mirrored = activityBridgeCreatedDestinations(ledger, 'wahoo')
    for (const wahoo of inputs.wahoo) {
      const day = localDate(wahoo.startDateLocal)
      if (wahoo.sport == null || day == null || mirrored.has(wahoo.id)) continue
      plans.push({
        sourceProvider,
        title: wahoo.name,
        localDate: day,
        sport: wahoo.sport,
        source: wahoo,
      })
    }
  }
  return plans.sort(
    (left, right) =>
      right.localDate.localeCompare(left.localDate) ||
      right.source.id.localeCompare(left.source.id),
  )
}
