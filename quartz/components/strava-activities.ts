import type { Element, Root } from 'hast'
import { h } from 'hastscript'
import { SKIP, visit } from 'unist-util-visit'
import { localIsoDay } from '../util/local-date'
import { joinSegments, pathToRoot, type FullSlug } from '../util/path'
import {
  latestStravaActivities,
  type StravaCardActivity,
  type StravaCardKind,
} from '../util/strava-latest-activities'
import { formatDurationClock } from '../util/triathlon-calculator'
import { triathlonActivityAnchor, triathlonDaySlug } from '../util/triathlon-date-route'

const MONTHS = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

const KIND_LABEL: Record<StravaCardKind, string> = { swim: 'swim', bike: 'ride', run: 'run' }

const TOTAL_KINDS: readonly StravaCardKind[] = ['swim', 'bike', 'run']

const MISSING = '–'

// YYYY-MM-DD read as a calendar day, so the build timezone cannot shift it.
const dayLabel = (date: string, withYear: boolean): string => {
  const [year, month, day] = date.split('-')
  const label = `${MONTHS[Number(month) - 1]} ${Number(day)}`
  return withYear ? `${label} ${year}` : label
}

const km = (meters: number, digits = 1): string =>
  `${(meters / 1000).toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits })} km`

const metres = (meters: number): string => `${Math.round(meters).toLocaleString('en-US')} m`

const distanceLabel = (activity: StravaCardActivity): string => {
  if (activity.distanceM <= 0) return MISSING
  return activity.kind === 'swim' ? metres(activity.distanceM) : km(activity.distanceM)
}

const climbLabel = (activity: StravaCardActivity): string =>
  activity.kind === 'swim' ? MISSING : metres(activity.elevationM)

const activityRow = (activity: StravaCardActivity, currentYear: string, root: string): Element => {
  const kind = KIND_LABEL[activity.kind]
  const date = dayLabel(activity.date, !activity.date.startsWith(currentYear))
  const distance = distanceLabel(activity)
  const moving = formatDurationClock(activity.movingTimeS)
  const climb = climbLabel(activity)
  const spoken = [
    `${kind}: ${activity.name}`,
    dayLabel(activity.date, true),
    distance === MISSING ? null : distance,
    `${moving} moving`,
    climb === MISSING ? null : `${climb} climbed`,
  ]
  return h('li', [
    h(
      'a.internal.strava-activity',
      {
        href: `${joinSegments(root, triathlonDaySlug(activity.date) ?? 'triathlon')}#${triathlonActivityAnchor(activity.id)}`,
        ariaLabel: spoken.filter(Boolean).join(', '),
      },
      [
        h('span.strava-activity-name', { title: activity.name }, [
          h('span.strava-activity-kind', kind),
          activity.name,
        ]),
        h('time.strava-activity-date', { dateTime: activity.date }, date),
        h('span.strava-activity-num', distance),
        h('span.strava-activity-num', moving),
        h('span.strava-activity-num', climb),
      ],
    ),
  ])
}

// Fills each `<div data-triathlon-latest>` marker in a note. HTML has no self-closing div, so a
// marker written as `<div data-triathlon-latest />` swallows the content after it; that content
// moves back out beside the card. Without synced data the marker links to the training log.
export function renderStravaActivities(root: Root, slug: FullSlug): void {
  visit(root, { tagName: 'div' }, (node: Element, index, parent) => {
    if (node.properties?.dataTriathlonLatest === undefined) return
    if (parent && index !== undefined && node.children.length > 0) {
      parent.children.splice(index + 1, 0, ...node.children)
    }
    delete node.properties.dataTriathlonLatest

    const summary = latestStravaActivities()
    if (!summary || summary.activities.length === 0) {
      node.children = [
        h(
          'a.internal',
          { href: joinSegments(pathToRoot(slug), 'triathlon') },
          'latest activities in the training log',
        ),
      ]
      return SKIP
    }

    node.properties.className = ['strava-activities']
    const athleteId = summary.athleteId
    const profile = `https://www.strava.com/athletes/${athleteId}`
    const currentYear = `${summary.year}-`
    const root = pathToRoot(slug)
    const synced = localIsoDay(summary.syncedAt)
    const kinds = TOTAL_KINDS.flatMap(kind => {
      const sum = summary.totals[kind]
      return sum ? [`${KIND_LABEL[kind]} ${km(sum.distanceM, 0)}`] : []
    })
    const hours = Object.values(summary.totals).reduce((s, sum) => s + sum.movingTimeS, 0) / 3600
    node.children = [
      h('div.strava-activities-head', [
        h('a.internal.strava-activities-title', { href: joinSegments(root, 'triathlon') }, [
          'latest activities',
        ]),
        h('span.strava-activities-meta', [
          h('span', ['synced ', h('time', { dateTime: synced }, dayLabel(synced, false))]),
          h('a.strava-activities-source', { href: profile }, 'strava'),
        ]),
      ]),
      h('div.strava-activities-grid', [
        h('div.strava-activities-cols', { ariaHidden: 'true' }, [
          h('span', 'activity'),
          h('span', 'date'),
          h('span', 'distance'),
          h('span', 'moving'),
          h('span', 'climb'),
        ]),
        h(
          'ol.strava-activities-list',
          summary.activities.map(activity => activityRow(activity, currentYear, root)),
        ),
      ]),
      h('div.strava-activities-foot', [
        h('span', `${summary.year}: ${[...kinds, `${Math.round(hours)} h`].join(', ')}`),
      ]),
    ]
    return SKIP
  })
}
