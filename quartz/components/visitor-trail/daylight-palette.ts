import type { VisitorDaylight } from '../../util/visitor-daylight'

export type DaylightTint = [number, number, number]

// Opus 5.5 supplied these restrained RGB gains for the existing paper, foliage and fire palette.
const dawn: DaylightTint = [1.04, 1, 0.95]
const dusk: DaylightTint = [1.05, 0.98, 0.91]
const night: DaylightTint = [0.93, 0.96, 1.04]

export function daylightTint(light: VisitorDaylight | null, automatic: boolean): DaylightTint {
  if (!light || !automatic) return [1, 1, 1]
  const warm = light.phase === 'dawn' || (light.phase === 'day' && light.bearing < 0) ? dawn : dusk
  const nightShare = 1 - light.daylight
  const clear = light.daylight > 0 ? light.sunlight / light.daylight : 0
  const warmShare = light.twilight * clear
  const channel = (index: number) =>
    Math.round((1 + (night[index] - 1) * nightShare + (warm[index] - 1) * warmShare) * 255) / 255
  return [channel(0), channel(1), channel(2)]
}
