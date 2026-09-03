import TurndownService from 'turndown'
import type { Analytics } from '../plugins/stores/analytics'
import type { StravaActivityDetail, StravaPayload } from '../plugins/stores/strava'
import type { TrainingPlan } from '../plugins/stores/training'
import type { FullSlug } from './path'
import { TRI_RACE_DISTANCES } from './triathlon-calculator'
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
  | 'feed'
  | 'on'
  | 'day'

export interface TriathlonMarkdownTools {
  conversions: ReadonlyArray<readonly [string, string]>
  gear: ReadonlyArray<readonly [string, readonly string[]]>
  maintenance: TriathlonMaintenance | null
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
    [conversions, '', distances, '', ...maintenance, gear].join('\n'),
    'race distance km, maintenance distance mi',
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
