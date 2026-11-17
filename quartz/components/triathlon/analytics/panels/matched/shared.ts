import type { MatchedRidesBlock } from '../../../../../plugins/stores/matched-rides'
import type { MatchedRunsBlock } from '../../../../../plugins/stores/matched-runs'
import type { TriathlonPresentation } from '../../../../../util/triathlon-presentation'
import { clock, KM_TO_MI, M_TO_FT } from '../../../../../util/triathlon-card'
import { createTriathlonFormatter } from '../../../runtime/formatter'
import { hms } from '../../shared'

export type MatchedRunGroup = MatchedRunsBlock['groups'][number]

export type MatchedRideGroup = MatchedRidesBlock['groups'][number]

export const matchedRunUnitScale = (presentation: TriathlonPresentation): number =>
  presentation.distance === 'imperial' ? KM_TO_MI : 1

export const matchedRunDisplayPace = (
  presentation: TriathlonPresentation,
  paceSPerKm: number,
): number => paceSPerKm / matchedRunUnitScale(presentation)

export // Older payloads predate elapsedTimeS; show the gap as missing data.
const matchedElapsed = (effort: { elapsedTimeS?: number }): string =>
  effort.elapsedTimeS != null && Number.isFinite(effort.elapsedTimeS) && effort.elapsedTimeS > 0
    ? hms(effort.elapsedTimeS)
    : '–'

export const matchedRunPace = (presentation: TriathlonPresentation, paceSPerKm: number): string =>
  `${clock(matchedRunDisplayPace(presentation, paceSPerKm))}${presentation.distance === 'imperial' ? '/mi' : '/km'}`

export const matchedRunDelta = (
  presentation: TriathlonPresentation,
  paceSPerKm: number,
  averagePaceSPerKm: number,
): string => {
  const delta = Math.round((paceSPerKm - averagePaceSPerKm) / matchedRunUnitScale(presentation))
  const sign = delta > 0 ? '+' : delta < 0 ? '-' : ''
  return `${sign}${Math.abs(delta)}s${presentation.distance === 'imperial' ? '/mi' : '/km'}`
}

export const matchedRunDirection = (
  paceSPerKm: number,
  averagePaceSPerKm: number,
): 'faster' | 'slower' | 'equal' =>
  paceSPerKm < averagePaceSPerKm ? 'faster' : paceSPerKm > averagePaceSPerKm ? 'slower' : 'equal'

export const matchedActivityDate = (presentation: TriathlonPresentation, iso: string): string => {
  const date = new Date(`${iso}T12:00:00Z`)
  return Number.isNaN(date.getTime())
    ? iso
    : date.toLocaleDateString(presentation.locale === 'fr' ? 'fr-CA' : 'en-US', {
        year: '2-digit',
        month: 'numeric',
        day: 'numeric',
        timeZone: 'UTC',
      })
}

export const matchedRidePower = (
  effort: MatchedRideGroup['efforts'][number],
  metric: MatchedRideGroup['powerMetric'],
): number =>
  metric === 'normalized' ? (effort.normalizedWatts ?? effort.averageWatts) : effort.averageWatts

export const matchedRidePowerText = (watts: number): string => `${Math.round(watts)} W`

export const matchedRidePowerDelta = (watts: number, averageWatts: number): string => {
  const delta = Math.round(watts - averageWatts)
  return `${delta > 0 ? '+' : delta < 0 ? '-' : ''}${Math.abs(delta)} W`
}

export const matchedRideClimbing = (
  presentation: TriathlonPresentation,
  metersPerKm: number,
): string =>
  presentation.distance === 'imperial'
    ? `${Math.round((metersPerKm * M_TO_FT) / KM_TO_MI)} ft/mi`
    : `${Math.round(metersPerKm)} m/km`

export const matchedRideEffortPower = (
  presentation: TriathlonPresentation,
  effort: MatchedRideGroup['efforts'][number],
  value: number | null,
): string =>
  value == null
    ? '—'
    : `${matchedRidePowerText(value)}${effort.powerSource === 'estimate' ? ` ${createTriathlonFormatter(presentation).text('estimated')}` : ''}`

export const matchedRunTrendingAverage = (group: MatchedRunGroup): number[] =>
  group.efforts.map((_, index) => {
    const window = group.efforts.slice(0, index + 1)
    return window.reduce((total, effort) => total + effort.paceSPerKm, 0) / window.length
  })

export const matchedSmoothPath = (
  values: number[],
  xOf: (index: number) => number,
  yOf: (value: number) => number,
): string => {
  const points = values.map((value, index) => ({ x: xOf(index), y: yOf(value) }))
  if (points.length < 2) return ''
  if (points.length === 2)
    return `M ${points[0].x.toFixed(2)} ${points[0].y.toFixed(2)} L ${points[1].x.toFixed(2)} ${points[1].y.toFixed(2)}`
  let path = `M ${points[0].x.toFixed(2)} ${points[0].y.toFixed(2)}`
  for (let index = 0; index < points.length - 1; index++) {
    const previous = points[Math.max(0, index - 1)]
    const start = points[index]
    const end = points[index + 1]
    const next = points[Math.min(points.length - 1, index + 2)]
    const firstX = start.x + (end.x - previous.x) * 0.15
    const firstY = start.y + (end.y - previous.y) * 0.15
    const secondX = end.x - (next.x - start.x) * 0.15
    const secondY = end.y - (next.y - start.y) * 0.15
    path += ` C ${firstX.toFixed(2)} ${firstY.toFixed(2)} ${secondX.toFixed(2)} ${secondY.toFixed(2)} ${end.x.toFixed(2)} ${end.y.toFixed(2)}`
  }
  return path
}
