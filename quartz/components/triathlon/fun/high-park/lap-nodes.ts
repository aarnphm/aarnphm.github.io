import type { HighParkLap } from '../../../../util/high-park-loops'
import { PICK_SLOTS, type LapPick } from '../../../../util/high-park-lap-compare'
import {
  lapChartScale,
  lapMetrics,
  lapViewModel,
  profileFields,
  profileModel,
  READ_WHO,
  type HighParkUnits,
  type LapSport,
  type ProfileField,
  type ProfileModel,
} from '../../../pages/triathlon-panels/high-park-laps'

export interface LapRide {
  laps: HighParkLap[]
  altitude: number[] | null
  sport: LapSport
}

export type Who = (typeof READ_WHO)[number]
export interface ReadNodes {
  el: HTMLElement
  lap: Text
  value: Text
  delta: Text
  dot: HTMLElement
}
export interface ProfileNodes {
  field: ProfileField
  model: ProfileModel
  svg: SVGSVGElement
  at: Text
  read: Map<Who, ReadNodes>
  pickLines: SVGPathElement[]
  hoverLine: SVGPathElement
  cursor: SVGPathElement
}
export interface ChartNodes {
  scale: NonNullable<ReturnType<typeof lapChartScale>>
  range: Text
  idle: string
  cursor: SVGPathElement
  /** Per pick slot, one pin per series. */
  pins: HTMLElement[][]
}
/** One rendered drop-down: its ride, what is picked and hovered, and the pooled nodes to write. */
export interface View {
  el: HTMLElement
  ride: LapRide
  units: HighParkUnits
  best: number
  lapKm: number
  picks: LapPick[]
  /** The baseline is on: picks read against the average lap. */
  even: boolean
  /** The lap under the pointer, and the bin when the pointer is on a profile; -1 for neither. */
  hoverLap: number
  bin: number
  lapName: Text
  charts: ChartNodes[]
  profiles: ProfileNodes[]
  chips: { el: HTMLElement; lap: Text }[]
  clear: HTMLElement | null
  mode: HTMLElement | null
  rows: HTMLElement[]
  marked: Set<number>
}

const textIn = (node: Element | null | undefined): Text => {
  const first = node?.firstChild
  if (first instanceof Text) return first
  const text = new Text('')
  node?.prepend(text)
  return text
}
const all = <T extends Element>(root: Element, selector: string): T[] => [
  ...root.querySelectorAll<T>(selector),
]

/** Find a rendered drop-down's nodes; `previous` hands over its picks and baseline switch. */
export const collectView = (
  el: HTMLElement,
  ride: LapRide,
  units: HighParkUnits,
  previous: View | undefined,
): View => {
  const { best, lapKm } = lapViewModel({ lapDetail: ride.laps })
  const one = <T extends Element>(root: Element, selector: string) =>
    root.querySelector<T>(selector)!
  const charts = lapMetrics(units, ride.sport).flatMap((metric): ChartNodes[] => {
    const figure = el.querySelector(`.tri-hp-lapchart[data-metric="${metric.key}"]`)
    const scale = lapChartScale(metric, ride.laps)
    if (!figure || !scale) return []
    const range = one<HTMLElement>(figure, '.tri-hp-range')
    return [
      {
        scale,
        range: textIn(range),
        idle: range.dataset.idle ?? '',
        cursor: one(figure, '.tri-hp-cursor'),
        pins: Array.from({ length: PICK_SLOTS }, (_, slot) =>
          all<HTMLElement>(figure, `.tri-hp-pin[data-who="${slot}"]`),
        ),
      },
    ]
  })
  const profiles = profileFields(units, ride.sport).flatMap((field): ProfileNodes[] => {
    const figure = el.querySelector(`.tri-hp-profile[data-metric="${field.key}"]`)
    const model = profileModel(field, ride.laps, ride.altitude, lapKm, ride.sport)
    if (!figure || !model) return []
    const read = new Map<Who, ReadNodes>()
    for (const who of READ_WHO) {
      const value = one<HTMLElement>(figure, `.tri-hp-val[data-who="${who}"]`)
      read.set(who, {
        el: value,
        lap: textIn(value.querySelector('b')),
        value: textIn(value.querySelector('span')),
        delta: textIn(value.querySelector('em')),
        dot: one(figure, `.tri-hp-dot[data-who="${who}"]`),
      })
    }
    return [
      {
        field,
        model,
        svg: one(figure, 'svg'),
        at: textIn(figure.querySelector('.tri-hp-at')),
        read,
        pickLines: all(figure, '.tri-hp-pickline'),
        hoverLine: one(figure, '.tri-hp-hoverline'),
        cursor: one(figure, '.tri-hp-cursor'),
      },
    ]
  })
  return {
    el,
    ride,
    units,
    best,
    lapKm,
    picks: previous?.picks ?? [],
    even: previous?.even ?? false,
    hoverLap: -1,
    bin: -1,
    lapName: textIn(el.querySelector('.tri-hp-lapname')),
    charts,
    profiles,
    chips: all<HTMLElement>(el, 'button.tri-hp-pick').map(chip => ({
      el: chip,
      lap: textIn(chip.querySelector('span')),
    })),
    clear: el.querySelector('.tri-hp-clear'),
    mode: el.querySelector('.tri-hp-mode'),
    rows: all(el, '.tri-hp-lapgrid tr[data-lap]'),
    marked: previous?.marked ?? new Set(),
  }
}
