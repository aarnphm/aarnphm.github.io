import type { BestEffortCategory, BestEffortEntry } from '../../../../util/best-efforts'
import type { TriathlonFormatter } from '../../runtime/formatter'
import { KM_TO_MI } from '../../../../util/triathlon-card'
import { el } from '../../runtime/dom'

export interface BestEffortDisplay {
  primary: string
  secondary: string | null
}

export const bestEffortTime = (seconds: number): string => {
  const total = Math.round(seconds)
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const s = (total % 60).toString().padStart(2, '0')
  return h > 0 ? `${h}:${m.toString().padStart(2, '0')}:${s}` : `${m}:${s}`
}

const unitLabel = (
  value: number,
  unit: 'second' | 'minute' | 'hour' | 'kilometer' | 'meter' | 'mile',
  formatter: TriathlonFormatter,
): string =>
  new Intl.NumberFormat(formatter.presentation.locale === 'fr' ? 'fr-CA' : 'en-US', {
    style: 'unit',
    unit,
    unitDisplay: 'long',
    maximumFractionDigits: 1,
  }).format(value)

export const bestEffortCategoryLabel = (
  category: BestEffortCategory,
  formatter: TriathlonFormatter,
): string => {
  if (category.group === 'longest') return formatter.text('longest ride')
  if (category.group === 'elevation') return formatter.text('elevation gain')
  if (category.group === 'power' && category.durationS != null) {
    const seconds = category.durationS
    return seconds < 60
      ? unitLabel(seconds, 'second', formatter)
      : seconds < 3600
        ? unitLabel(seconds / 60, 'minute', formatter)
        : unitLabel(seconds / 3600, 'hour', formatter)
  }
  const label = category.label ?? category.key
  if (label === '1/2 mile') return unitLabel(0.5, 'mile', formatter)
  const distance = /^(\d+)(K|m| mile)$/.exec(label)
  if (distance)
    return unitLabel(
      Number(distance[1]),
      distance[2] === 'K' ? 'kilometer' : distance[2] === 'm' ? 'meter' : 'mile',
      formatter,
    )
  return formatter.text(label)
}

/** Pace for runs and speed for rides over a fixed distance; the time itself is the primary value. */
const distanceRate = (
  category: BestEffortCategory,
  seconds: number,
  formatter: TriathlonFormatter,
): string | null => {
  if (category.distanceM == null || seconds <= 0) return null
  const km = category.distanceM / 1000
  if (category.sport === 'run') return formatter.pace(seconds / km)
  const imperial = formatter.presentation.distance === 'imperial'
  const speed = (km / (seconds / 3600)) * (imperial ? KM_TO_MI : 1)
  return `${formatter.number(speed, 1)} ${imperial ? 'mph' : 'km/h'}`
}

export const bestEffortDisplay = (
  category: BestEffortCategory,
  entry: BestEffortEntry,
  formatter: TriathlonFormatter,
): BestEffortDisplay => {
  switch (category.unit) {
    case 'seconds':
      return {
        primary: bestEffortTime(entry.value),
        secondary: distanceRate(category, entry.value, formatter),
      }
    case 'km':
      return {
        primary: formatter.distance(entry.value, 'bike'),
        secondary: entry.timeS == null ? null : bestEffortTime(entry.timeS),
      }
    case 'm':
      return {
        primary: formatter.elevation(entry.value),
        secondary: entry.distanceKm == null ? null : formatter.distance(entry.distanceKm, 'bike'),
      }
    case 'watts':
      return {
        primary: `${formatter.number(Math.round(entry.value))} W`,
        secondary:
          entry.wattsPerKg == null ? null : `${formatter.number(entry.wattsPerKg, 2)} W/kg`,
      }
  }
}

/** Axis tick text for a category value: pace or speed for fixed-distance times, else the primary unit. */
export const bestEffortAxisLabel = (
  category: BestEffortCategory,
  value: number,
  formatter: TriathlonFormatter,
): string => {
  switch (category.unit) {
    case 'seconds':
      return distanceRate(category, value, formatter) ?? bestEffortTime(value)
    case 'km':
      return formatter.distance(value, 'bike')
    case 'm':
      return formatter.elevation(value)
    case 'watts':
      return `${formatter.number(Math.round(value))} W`
  }
}

export const isPodiumRank = (rank: number): rank is 1 | 2 | 3 => rank >= 1 && rank <= 3

/** Rank 1 reads PR; every other rank is its number. */
export const bestEffortRankText = (rank: number, formatter: TriathlonFormatter): string =>
  rank === 1 ? formatter.text('PR') : `#${rank}`

/** Boxed rank shared by the hero, the rows and the chart labels; the podium boxes carry a colour each. */
export const buildBestEffortRank = (rank: number, formatter: TriathlonFormatter): HTMLElement =>
  el(
    'span',
    `tri-be-rank${isPodiumRank(rank) ? ` tri-be-rank--${rank === 1 ? 'pr' : rank}` : ''}`,
    bestEffortRankText(rank, formatter),
    { title: `${formatter.text('all-time rank')} #${rank}` },
  )

/** Collapsed tab text in the map metric idiom: distances as written, durations and groups as codes. */
export const bestEffortShortLabel = (
  category: BestEffortCategory,
  formatter: TriathlonFormatter,
): string => {
  if (category.group === 'longest') return 'L'
  if (category.group === 'elevation') return 'E'
  if (category.group === 'power' && category.durationS != null) {
    const seconds = category.durationS
    if (seconds < 60) return `${formatter.number(seconds)}s`
    if (seconds < 3600) return `${formatter.number(seconds / 60, 0, 1)}m`
    return `${formatter.number(seconds / 3600, 0, 1)}h`
  }
  const label = category.label ?? category.key
  if (label === '1/2 mile') return `${formatter.number(0.5, 1)}mi`
  if (label === 'Half marathon') return formatter.text('HM')
  if (label === 'Marathon') return 'M'
  return label.replace(/ mile$/, 'mi')
}
