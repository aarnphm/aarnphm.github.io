import type {
  BestEffortEntry,
  BestEffortGroup,
  BestEffortSport,
} from '../../../../../util/best-efforts'
import { ELEV_RAMP, HEAT_RAMP, SPD_RAMP, STRIDE_RAMP } from '../../../maps/palette'
export const TRI_BEST_EFFORTS_KEY = 'tri-best-efforts'

export type BestEffortsView = 'analysis' | 'top'

export type BestEffortsSort = 'newest' | 'oldest' | 'high' | 'low'

export interface BestEffortsState {
  sport: BestEffortSport
  view: BestEffortsView
  runCategory: string
  bikeGroup: BestEffortGroup
  bikeCategory: Partial<Record<BestEffortGroup, string>>
  year: number | null
  sort: BestEffortsSort
}

export interface CategoryTabOption {
  value: string
  label: string
  short: string
}

export interface CategoryTablist {
  name: string
  element: HTMLElement
  tabs: HTMLButtonElement[]
  values: string[]
}

export const SPORTS: readonly BestEffortSport[] = ['run', 'bike']

export const VIEWS: readonly { key: BestEffortsView; label: string }[] = [
  { key: 'analysis', label: 'analysis' },
  { key: 'top', label: 'top 10' },
]

export const SORTS: readonly { key: BestEffortsSort; label: string }[] = [
  { key: 'newest', label: 'newest first' },
  { key: 'oldest', label: 'oldest first' },
  { key: 'high', label: 'high → low' },
  { key: 'low', label: 'low → high' },
]

export const BIKE_GROUPS: readonly BestEffortGroup[] = ['longest', 'elevation', 'distance', 'power']

export const BIKE_GROUP_LABEL: Record<BestEffortGroup, string> = {
  longest: 'longest ride',
  elevation: 'elevation gain',
  distance: 'distance',
  power: 'power',
}

export const BIKE_GROUP_SHORT: Record<BestEffortGroup, string> = {
  longest: 'L',
  elevation: 'E',
  distance: 'D',
  power: 'W',
}

export const BIKE_GROUP_COLOR: Record<BestEffortGroup, string> = {
  longest: STRIDE_RAMP[6],
  elevation: ELEV_RAMP[6],
  distance: SPD_RAMP[6],
  power: HEAT_RAMP[6],
}

export const BIKE_GROUP_DEFAULT: Partial<Record<BestEffortGroup, string>> = {
  distance: 'bike:distance:40K',
  power: 'bike:power:1200',
}

export const entryYear = (entry: BestEffortEntry): number => Number(entry.date.slice(0, 4))

export const byRank = (efforts: readonly BestEffortEntry[]): BestEffortEntry[] =>
  [...efforts].sort((a, b) => a.rank - b.rank)
