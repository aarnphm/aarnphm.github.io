import type { Analytics } from '../plugins/stores/analytics'
import type {
  TrainingPeaksCalendarLocalZones,
  TrainingPeaksCalendarZoneBounds,
} from './trainingpeaks-calendar'

const zoneBounds = (
  threshold: number | null | undefined,
  bounds: readonly number[],
): TrainingPeaksCalendarZoneBounds | null =>
  threshold != null &&
  Number.isFinite(threshold) &&
  threshold > 0 &&
  bounds.length > 0 &&
  bounds.every((bound, index) => bound > 0 && (index === 0 || bound > bounds[index - 1]))
    ? { threshold, bounds: [...bounds] }
    : null

/** The zone bounds the activity pages use, anchored to the garden thresholds. */
export function trainingPeaksLocalZones(
  analytics: Analytics,
  ftp: number | null,
): TrainingPeaksCalendarLocalZones {
  const { distributions } = analytics
  const { lactateThreshold } = analytics.engine
  const run = lactateThreshold.sports.find(item => item.sport === 'run' && item.unit === 's/km')
  return {
    power: zoneBounds(ftp, distributions.powerZoneBounds),
    // Pace bounds run from slow to fast in s/km, so the speeds ascend.
    runSpeed: zoneBounds(
      run && 1000 / run.value,
      distributions.paceZoneBoundsSPerKm.map(seconds => 1000 / seconds),
    ),
    heartRate: zoneBounds(lactateThreshold.heartRate?.value, distributions.heartRateZoneBounds),
  }
}
