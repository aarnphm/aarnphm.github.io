import TurndownService from 'turndown'
import type { Analytics } from '../plugins/stores/analytics'
import type { StravaActivityDetail, StravaPayload } from '../plugins/stores/strava'
import type { TrainingPlan } from '../plugins/stores/training'
import type { FullSlug } from './path'
import type { TriathlonEquipmentUsage } from './triathlon-equipment'
import { TRI_RACE_DISTANCES } from './triathlon-calculator'
import {
  calendarScheduleDates,
  calendarScheduleEdition,
  TRIATHLON_CALENDAR_LEGS,
  TRIATHLON_CALENDAR_RESULT_SEGMENTS,
  type TriathlonCalendar,
  type TriathlonCalendarEvent,
} from './triathlon-calendar'
import { buildFeedMarkdown } from './triathlon-feed'
import { type TriathlonMaintenance, type TriathlonMaintenanceRange } from './triathlon-maintenance'
import {
  escapeMarkdownHeading,
  renderGfmTable,
  renderTitledSections,
} from './triathlon-markdown-data'

export type TriathlonMarkdownView =
  | 'tools'
  | 'calc'
  | 'analytics'
  | 'maps'
  | 'training'
  | 'calendar'
  | 'feed'
  | 'on'
  | 'day'

export interface TriathlonMarkdownTools {
  conversions: ReadonlyArray<readonly [string, string]>
  gear: ReadonlyArray<readonly [string, readonly string[]]>
  maintenance: TriathlonMaintenance | null
  equipment?: Record<string, TriathlonEquipmentUsage>
}

export interface TriathlonMarkdownOptions {
  view: TriathlonMarkdownView
  slug: FullSlug
  title: string
  description: string
  baseUrl?: string
  scopePrefix?: string
  dataFeed: string
  analytics: Analytics
  payload: StravaPayload
  plans: TrainingPlan[]
  calendar?: TriathlonCalendar | null
  tools: TriathlonMarkdownTools
}

const turndown = new TurndownService({
  headingStyle: 'atx',
  bulletListMarker: '-',
  codeBlockStyle: 'fenced',
})

const tableCellMarkdown = (cell: Element): string =>
  turndown
    .turndown(cell.innerHTML)
    .trim()
    .replace(/\n+/g, '<br>')
    .replace(/\s*<br>\s*/g, '<br>')
    .replace(/\|/g, '\\|')

turndown.addRule('table', {
  filter: 'table',
  replacement: (_content, node) => {
    const rows = Array.from(node.querySelectorAll('tr'))
      .map(row =>
        Array.from(row.children).filter(cell => cell.tagName === 'TH' || cell.tagName === 'TD'),
      )
      .filter(row => row.length > 0)
    if (rows.length === 0) return ''
    const lines = rows.map(row => `| ${row.map(cell => tableCellMarkdown(cell)).join(' | ')} |`)
    const separator = `| ${rows[0].map(() => '---').join(' | ')} |`
    return `\n\n${[lines[0], separator, ...lines.slice(1)].join('\n')}\n\n`
  },
})

const generatedAt = (payload: StravaPayload): string => new Date(payload.generatedAt).toISOString()

const origin = (baseUrl?: string): string => (baseUrl ? `https://${baseUrl}` : '')

const document = (
  opts: TriathlonMarkdownOptions,
  body: string,
  units = 'distance km, time seconds, elevation m, heart rate bpm, power W, temperature C',
): string => {
  const pageUrl = `${origin(opts.baseUrl)}/${opts.slug}`
  return [
    '---',
    `title: ${opts.title}`,
    `source: ${pageUrl}`,
    `permalink: ${pageUrl}.md`,
    `generated: ${generatedAt(opts.payload)}`,
    `units: ${units}`,
    `description: ${opts.description}`,
    '---',
    '',
    `# ${opts.title}`,
    '',
    opts.description,
    '',
    body,
    '',
  ].join('\n')
}

const analyticsMarkdown = (opts: TriathlonMarkdownOptions): string => {
  const related = {
    activityData: `${origin(opts.baseUrl)}/static/strava-detail.json`,
    activityFeed: `${origin(opts.baseUrl)}/triathlon/feed.md`,
  }
  return document(
    opts,
    [
      renderTitledSections(related, { title: 'relatedData' }),
      renderTitledSections(opts.analytics, { title: 'analytics' }),
    ].join('\n\n'),
  )
}

const mapRouteSummary = (detail: StravaActivityDetail) => {
  const segments = detail.mapRoute.length > 0 ? detail.mapRoute : [detail.route]
  const points = segments.flat()
  if (points.length === 0) return null
  const first = points[0]
  const last = points[points.length - 1]
  let south = first.lat
  let west = first.lng
  let north = first.lat
  let east = first.lng
  for (const point of points) {
    south = Math.min(south, point.lat)
    west = Math.min(west, point.lng)
    north = Math.max(north, point.lat)
    east = Math.max(east, point.lng)
  }
  return {
    segmentCount: segments.length,
    pointCount: points.length,
    start: { lat: first.lat, lng: first.lng, distanceKm: first.d },
    finish: { lat: last.lat, lng: last.lng, distanceKm: last.d },
    bounds: { south, west, north, east },
  }
}

const mapActivity = (detail: StravaActivityDetail) => ({
  id: detail.id,
  date: detail.date,
  start: detail.start,
  sport: detail.sport,
  name: detail.name,
  distanceKm: detail.distanceKm,
  movingTimeS: detail.movingTimeS,
  elapsedTimeS: detail.elapsedTimeS,
  maxSpeedKph: detail.maxSpeedKph,
  elevationM: detail.elevationM,
  avgHr: detail.avgHr,
  avgWatts: detail.avgWatts,
  avgCadence: detail.avgCadence,
  deviceTemperatureC: detail.deviceTemperatureC,
  ambientTemperatureC: detail.ambientTemperatureC,
  windKph: detail.windKph,
  windDir: detail.windDir,
  averageRelativeHumidityPct: detail.averageRelativeHumidityPct,
  relativeHumidityProvenance: detail.relativeHumidityProvenance,
  location: detail.location,
  route: mapRouteSummary(detail),
})

const mapsMarkdown = (opts: TriathlonMarkdownOptions): string => {
  const activities = Object.values(opts.payload.details)
    .filter(
      detail => detail.mapRoute.some(segment => segment.length >= 2) || detail.route.length >= 2,
    )
    .sort((left, right) => right.start.localeCompare(left.start))
    .map(mapActivity)
  const data = {
    activityCount: activities.length,
    fullActivityData: `${origin(opts.baseUrl)}/static/strava-detail.json`,
    activities,
  }
  return document(opts, renderTitledSections(data, { title: 'mappedActivities' }))
}

const trainingMarkdown = (opts: TriathlonMarkdownOptions): string => {
  const plans = opts.plans
    .map(plan => {
      const metadata = renderGfmTable([
        {
          id: plan.id,
          distance: plan.distance || 'unspecified',
          date: plan.date || 'unspecified',
          target: plan.target || 'unspecified',
          author: plan.author || 'unspecified',
        },
      ])
      return [
        `## ${escapeMarkdownHeading(plan.meta || plan.id)}`,
        '',
        metadata,
        '',
        turndown.turndown(plan.html),
      ].join('\n')
    })
    .join('\n\n')
  return document(opts, plans || 'No generated training plans are available.')
}

const celsiusText = (range: [number, number] | null): string =>
  range ? (range[0] === range[1] ? `${range[0]} °C` : `${range[0]}–${range[1]} °C`) : ''

const calendarDetailsMarkdown = (event: TriathlonCalendarEvent, year: number): string[] => {
  const details = event.details
  if (!details) return []
  const sections: string[] = []
  const legs = TRIATHLON_CALENDAR_LEGS.filter(
    leg => details.terrain[leg] || details.course.some(course => course.leg === leg),
  )
  if (legs.length > 0) {
    const rows = legs.map(leg => {
      const course = details.course.find(entry => entry.leg === leg)
      return {
        leg,
        distanceKm: course ? Math.round((course.raceDistanceM ?? course.distanceM) / 100) / 10 : '',
        mappedDistanceKm: course?.raceDistanceM ? Math.round(course.distanceM / 100) / 10 : '',
        edition: course?.edition ?? '',
        laps: course?.laps ?? '',
        elevationGainM: course?.elevationGainM ?? '',
        aidStations: course?.aidM.length || '',
        aidStationsPerLap: course?.aidStationsPerLap ?? '',
        terrain: details.terrain[leg] ?? '',
        route: course?.source ?? '',
        map: course?.mapUrl ?? '',
        gpx: course?.gpxUrl ?? '',
      }
    })
    sections.push(
      '### Course',
      '',
      renderGfmTable(rows, {
        plainTextColumns: [
          'leg',
          'distanceKm',
          'mappedDistanceKm',
          'edition',
          'laps',
          'elevationGainM',
          'aidStations',
          'aidStationsPerLap',
          'terrain',
          'route',
          'map',
          'gpx',
        ],
      }),
    )
  }
  if (details.airC || details.waterC)
    sections.push(
      '',
      `Air ${celsiusText(details.airC) || 'unknown'} · water ${celsiusText(details.waterC) || 'unknown'}`,
    )
  const schedule = details.schedule
  const days = calendarScheduleDates(event)
  if (schedule && days.length > 0) {
    const edition = calendarScheduleEdition(schedule)
    const rows = days.flatMap(day =>
      day.items.map(item => ({
        date: day.date,
        start: item.start,
        end: item.end ?? '',
        activity: item.activity,
        location: item.location ?? '',
        race: item.race ? 'race start' : '',
      })),
    )
    sections.push(
      '',
      '### Schedule',
      '',
      edition === year
        ? `Source: <${schedule.source}>`
        : `Times come from the ${edition} schedule, placed on ${year} dates by days from race day. Source: <${schedule.source}>`,
      '',
      renderGfmTable(rows, {
        plainTextColumns: ['date', 'start', 'end', 'activity', 'location', 'race'],
      }),
    )
  } else if (details.schedulePending) sections.push('', 'Schedule not published.')
  if (details.venueMap) sections.push('', `Venue map: <${details.venueMap}>`)
  if (details.athleteGuide) sections.push('', `Athlete guide: <${details.athleteGuide}>`)
  if (details.fetched) sections.push('', `Details fetched: ${details.fetched}`)
  const kept = Object.entries(details.kept)
  if (kept.length > 0)
    sections.push(
      '',
      `Kept from earlier syncs after extraction failed: ${kept.map(([field, date]) => `${field} (${date})`).join(', ')}.`,
    )
  return sections.length > 0 ? ['', ...sections] : []
}

const calendarMarkdown = (opts: TriathlonMarkdownOptions): string => {
  const calendar = opts.calendar
  if (!calendar) return document(opts, 'No events planned.', 'local calendar dates')
  const events = calendar.events.map(event =>
    [
      `## ${escapeMarkdownHeading(event.name)}`,
      '',
      renderGfmTable(
        [
          {
            date: event.date ?? 'date pending',
            endDate: event.endDate ?? event.date ?? 'date pending',
            location: event.location,
            series: event.series,
            kind: event.kind,
            format: event.format,
          },
        ],
        { plainTextColumns: ['date', 'endDate', 'location', 'series', 'kind', 'format'] },
      ),
      '',
      `Official event: <${event.url}>`,
      ...(event.participated ? ['', 'Participated.'] : []),
      ...(event.note ? ['', escapeMarkdownHeading(event.note)] : []),
      ...(event.results
        ? [
            '',
            '### Race results',
            '',
            renderGfmTable(
              TRIATHLON_CALENDAR_RESULT_SEGMENTS.flatMap(segment => {
                const time = event.results?.[segment]
                return time ? [{ segment, time }] : []
              }),
              { plainTextColumns: ['segment', 'time'] },
            ),
          ]
        : []),
      ...calendarDetailsMarkdown(event, calendar.year),
    ].join('\n'),
  )
  return document(
    opts,
    [
      `${calendar.year} · ${calendar.events.length} ${calendar.events.length === 1 ? 'event' : 'events'}`,
      '',
      `${calendar.checked ? `Dates checked: ${calendar.checked}. ` : ''}End dates are inclusive.${calendar.fetched ? ` Courses and schedules fetched: ${calendar.fetched}.` : ''}`,
      '',
      `[Download calendar](${origin(opts.baseUrl)}/triathlon/calendar/${calendar.year}.ics)`,
      '',
      events.join('\n\n') || 'No events planned.',
    ].join('\n'),
    'local calendar dates',
  )
}

const maintenanceRangeText = (ranges: TriathlonMaintenanceRange[]): string =>
  ranges.map(range => `${range.start} to ${range.end ?? 'current'}`).join(', ')

const toolsMarkdown = (opts: TriathlonMarkdownOptions): string => {
  const conversions = renderTitledSections(
    opts.tools.conversions.map(([kind, conversion]) => ({ kind, conversion })),
    { title: 'conversions' },
  )
  const distances = renderTitledSections(
    TRI_RACE_DISTANCES.map(([distance, swimKm, bikeKm, runKm]) => ({
      distance,
      swimKm,
      bikeKm,
      runKm,
    })),
    { title: 'raceDistances' },
  )
  const gear = renderTitledSections(
    opts.tools.gear.flatMap(([category, items]) => items.map(item => ({ category, item }))),
    { title: 'gearAndFuel' },
  )
  const maintenanceData = opts.tools.maintenance
  const equipment = renderTitledSections(Object.values(opts.tools.equipment ?? {}), {
    title: 'equipment',
  })
  const maintenance = maintenanceData
    ? [
        '## maintenance',
        '',
        renderTitledSections(
          maintenanceData.services.map(entry => ({
            bike: entry.bike,
            date: entry.date,
            place: entry.place,
            distanceMiles: entry.distanceMiles,
          })),
          { title: 'maintenance.services', headingDepth: 3 },
        ),
        '',
        renderTitledSections(
          maintenanceData.components.map(entry => ({
            component: entry.component,
            type: entry.type,
            ranges: maintenanceRangeText(entry.ranges),
            distanceMiles: entry.distanceMiles,
            reason: entry.reason,
          })),
          { title: 'maintenance.components', headingDepth: 3 },
        ),
        '',
        renderTitledSections(
          maintenanceData.chains.map(entry => ({
            id: entry.id,
            lubricant: entry.lubricant,
            since: entry.since,
            distanceMiles: entry.distanceMiles,
            waxed: entry.waxed,
          })),
          { title: 'maintenance.chains', headingDepth: 3 },
        ),
        '',
        renderTitledSections(
          maintenanceData.wheels.map(entry => ({
            position: entry.position,
            part: entry.part,
            type: entry.type,
            ranges: maintenanceRangeText(entry.ranges),
            distanceMiles: entry.distanceMiles,
            repaired: entry.repaired,
            reason: entry.reason,
          })),
          { title: 'maintenance.tires', headingDepth: 3 },
        ),
        '',
      ]
    : []
  return document(
    opts,
    [conversions, '', distances, '', ...maintenance, equipment, '', gear].join('\n'),
    'race distance km, maintenance distance mi, equipment lifetime distance m',
  )
}

const calculatorMarkdown = (opts: TriathlonMarkdownOptions): string => {
  const data = {
    presets: TRI_RACE_DISTANCES.map(([label, swimKm, bikeKm, runKm]) => ({
      label,
      swimKm,
      bikeKm,
      runKm,
    })),
    calibration: opts.analytics.calibration,
    thresholds: opts.analytics.thresholds,
    raceReadiness: opts.analytics.races,
    events: opts.analytics.events,
    ftpHypothesis: opts.analytics.engine.ftpHypothesis,
    zones: opts.payload.zones,
  }
  return document(opts, renderTitledSections(data, { title: 'calculatorInputs' }))
}

const activityMarkdown = (opts: TriathlonMarkdownOptions): string =>
  buildFeedMarkdown(opts.dataFeed, opts.analytics, {
    details: opts.payload.details,
    baseUrl: opts.baseUrl,
    generatedAt: generatedAt(opts.payload),
    title: opts.title,
    sourcePath: `/${opts.slug}`,
    scopePrefix: opts.scopePrefix,
    includeActivityDetails: opts.view === 'day',
    includeRestDays: opts.view !== 'feed',
  })

export function buildTriathlonMarkdown(opts: TriathlonMarkdownOptions): string {
  switch (opts.view) {
    case 'analytics':
      return analyticsMarkdown(opts)
    case 'maps':
      return mapsMarkdown(opts)
    case 'training':
      return trainingMarkdown(opts)
    case 'calendar':
      return calendarMarkdown(opts)
    case 'tools':
      return toolsMarkdown(opts)
    case 'calc':
      return calculatorMarkdown(opts)
    case 'feed':
    case 'on':
    case 'day':
      return activityMarkdown(opts)
  }
}
