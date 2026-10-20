import type { Element, Root } from 'hast'
import { h } from 'hastscript'
import { visit } from 'unist-util-visit'
import { localIsoDay } from '../util/local-date'
import { latestStravaRides, type StravaRide } from '../util/strava-latest-rides'
import { formatDurationClock } from '../util/triathlon-calculator'

const MONTHS = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

// YYYY-MM-DD read as a calendar day, so the build timezone cannot shift it.
const dayLabel = (date: string, withYear: boolean): string => {
  const [year, month, day] = date.split('-')
  const label = `${MONTHS[Number(month) - 1]} ${Number(day)}`
  return withYear ? `${label} ${year}` : label
}

const km = (meters: number, digits = 1): string =>
  `${(meters / 1000).toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits })} km`

const climb = (meters: number): string => `${Math.round(meters).toLocaleString('en-US')} m`

const rideRow = (ride: StravaRide, currentYear: string): Element => {
  const withYear = !ride.date.startsWith(currentYear)
  const date = dayLabel(ride.date, withYear)
  const distance = km(ride.distanceM)
  const moving = formatDurationClock(ride.movingTimeS)
  const elevation = climb(ride.elevationM)
  return h('li', [
    h(
      'a.strava-ride',
      {
        href: `https://www.strava.com/activities/${ride.id}`,
        ariaLabel: `${ride.name}, ${dayLabel(ride.date, true)}, ${distance}, ${moving} moving, ${elevation} climbed`,
      },
      [
        h('span.strava-ride-name', { title: ride.name }, ride.name),
        h('time.strava-ride-date', { dateTime: ride.date }, date),
        h('span.strava-ride-num', distance),
        h('span.strava-ride-num', moving),
        h('span.strava-ride-num', elevation),
      ],
    ),
  ])
}

// Fills the placeholders that ofm leaves for Strava's latest-rides iframe. A placeholder keeps its
// profile link when the cache is missing or belongs to another athlete.
export function renderStravaRides(root: Root): void {
  visit(root, { tagName: 'div' }, (node: Element) => {
    const athleteId = Number(node.properties?.dataStravaAthlete)
    if (!athleteId) return
    const summary = latestStravaRides()
    if (!summary || summary.athleteId !== athleteId || summary.rides.length === 0) return

    const profile = `https://www.strava.com/athletes/${athleteId}`
    const currentYear = `${summary.totals.year}-`
    const synced = localIsoDay(summary.syncedAt)
    const { totals } = summary
    node.children = [
      h('div.strava-rides-head', [
        h('span.strava-rides-title', 'latest rides'),
        h('a.strava-rides-source', { href: profile }, 'strava'),
      ]),
      h('div.strava-rides-grid', [
        h('div.strava-rides-cols', { ariaHidden: 'true' }, [
          h('span', 'ride'),
          h('span', 'date'),
          h('span', 'distance'),
          h('span', 'moving'),
          h('span', 'climb'),
        ]),
        h(
          'ol.strava-rides-list',
          summary.rides.map(ride => rideRow(ride, currentYear)),
        ),
      ]),
      h('div.strava-rides-foot', [
        h(
          'span',
          `${totals.year}: ${totals.count} rides, ${km(totals.distanceM, 0)}, ${Math.round(totals.movingTimeS / 3600)} h, ${climb(totals.elevationM)}`,
        ),
        h('span', ['synced ', h('time', { dateTime: synced }, dayLabel(synced, false))]),
      ]),
    ]
  })
}
