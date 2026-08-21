import TurndownService from 'turndown'
import type { Analytics } from '../plugins/stores/analytics'
import type { StravaActivityDetail, StravaPayload } from '../plugins/stores/strava'
import type { TrainingPlan } from '../plugins/stores/training'
import type { FullSlug } from './path'
import { TRI_RACE_DISTANCES } from './triathlon-calculator'
import { buildFeedMarkdown } from './triathlon-feed'
import {
  formatTriathlonMaintenanceDistance,
  type TriathlonMaintenance,
  type TriathlonMaintenanceRange,
} from './triathlon-maintenance'

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

const jsonBlock = (value: unknown): string =>
  `\`\`\`json\n${JSON.stringify(value, null, 2)}\n\`\`\``

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
      '## related data',
      '',
      jsonBlock(related),
      '',
      '## analytics',
      '',
      jsonBlock(opts.analytics),
    ].join('\n'),
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
  maxSpeedKph: detail.maxSpeedKph,
  elevationM: detail.elevationM,
  avgHr: detail.avgHr,
  avgWatts: detail.avgWatts,
  avgCadence: detail.avgCadence,
  avgTemp: detail.avgTemp,
  windKph: detail.windKph,
  windDir: detail.windDir,
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
  return document(opts, ['## mapped activities', '', jsonBlock(data)].join('\n'))
}

const trainingMarkdown = (opts: TriathlonMarkdownOptions): string => {
  const plans = opts.plans
    .map(plan => {
      const metadata = [
        `- id: ${plan.id}`,
        `- distance: ${plan.distance || 'unspecified'}`,
        `- date: ${plan.date || 'unspecified'}`,
        `- target: ${plan.target || 'unspecified'}`,
        `- author: ${plan.author || 'unspecified'}`,
      ].join('\n')
      return [`## ${plan.meta || plan.id}`, '', metadata, '', turndown.turndown(plan.html)].join(
        '\n',
      )
    })
    .join('\n\n')
  return document(opts, plans || 'No generated training plans are available.')
}

const maintenanceRangeText = (ranges: TriathlonMaintenanceRange[]): string =>
  ranges.map(range => `${range.start} to ${range.end ?? 'current'}`).join(', ')

const maintenanceSection = (title: string, entries: string[]): string[] =>
  entries.length > 0 ? [`### ${title}`, '', ...entries, ''] : []

const maintenanceDistanceText = (distanceMiles: number | null): string =>
  distanceMiles === null ? '' : `; ${formatTriathlonMaintenanceDistance(distanceMiles, 'imperial')}`

const toolsMarkdown = (opts: TriathlonMarkdownOptions): string => {
  const conversions = [
    '| kind | conversion |',
    '| --- | --- |',
    ...opts.tools.conversions.map(([kind, conversion]) => `| ${kind} | ${conversion} |`),
  ].join('\n')
  const distances = [
    '| distance | swim km | bike km | run km |',
    '| --- | ---: | ---: | ---: |',
    ...TRI_RACE_DISTANCES.map(
      ([label, swim, bike, run]) => `| ${label} | ${swim} | ${bike} | ${run} |`,
    ),
  ].join('\n')
  const gear = opts.tools.gear
    .map(([label, items]) => [`### ${label}`, '', ...items.map(item => `- ${item}`)].join('\n'))
    .join('\n\n')
  const maintenanceData = opts.tools.maintenance
  const maintenance = maintenanceData
    ? [
        '## maintenance',
        '',
        ...maintenanceSection(
          'service',
          maintenanceData.services.map(
            entry =>
              `- ${entry.bike}: ${entry.date}; ${entry.place}${maintenanceDistanceText(entry.distanceMiles)}`,
          ),
        ),
        ...maintenanceSection(
          'components',
          maintenanceData.components.map(
            entry =>
              `- ${entry.component}: ${entry.type}; ${maintenanceRangeText(entry.ranges)}${maintenanceDistanceText(entry.distanceMiles)}${entry.reason ? `; reason: ${entry.reason}` : ''}`,
          ),
        ),
        ...maintenanceSection(
          'chains',
          maintenanceData.chains.map(
            entry =>
              `- chain ${entry.id}: ${entry.lubricant}; since ${entry.since}${maintenanceDistanceText(entry.distanceMiles)}; waxed ${entry.waxed ? 'yes' : 'no'}`,
          ),
        ),
        ...maintenanceSection(
          'tires',
          maintenanceData.wheels.map(entry => {
            const repaired =
              entry.repaired === null ? '' : `; repaired ${entry.repaired ? 'yes' : 'no'}`
            return `- ${entry.position} ${entry.part}: ${entry.type}; ${maintenanceRangeText(entry.ranges)}${maintenanceDistanceText(entry.distanceMiles)}${repaired}${entry.reason ? `; reason: ${entry.reason}` : ''}`
          }),
        ),
      ]
    : []
  return document(
    opts,
    [
      '## conversions',
      '',
      conversions,
      '',
      '## race distances',
      '',
      distances,
      '',
      ...maintenance,
      '## gear and fuel',
      '',
      gear,
    ].join('\n'),
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
  return document(opts, ['## calculator inputs', '', jsonBlock(data)].join('\n'))
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
