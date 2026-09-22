import type { PowerCurvePoint } from '../plugins/stores/strava'

export const FTP_FROM_P20 = 0.95

export interface PowerCurveFtpEstimate {
  watts: number
  method: '95%-20-minute-power'
  confidence: 'provisional'
  anchor: PowerCurvePoint
}

export const estimateFtpFromPowerCurve = (
  curve: readonly PowerCurvePoint[],
): PowerCurveFtpEstimate | null => {
  let anchor: PowerCurvePoint | null = null
  for (const point of curve) {
    if (point.s !== 1200 || !Number.isFinite(point.w) || point.w <= 0) continue
    if (!anchor || point.w > anchor.w) anchor = point
  }
  if (!anchor) return null
  const watts = Math.round(anchor.w * FTP_FROM_P20)
  if (watts <= 0) return null

  // A recorded best effort does not establish that the rider completed a maximal FTP test.
  return { watts, method: '95%-20-minute-power', confidence: 'provisional', anchor }
}
