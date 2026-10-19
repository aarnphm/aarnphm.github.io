import type { StravaActivityDetail } from '../plugins/stores/strava'
import {
  ATHLETE,
  type Analytics,
  type FeedActivityRow,
  type FeedDayRow,
  type FeedWeekRow,
} from '../plugins/stores/analytics'
import { escapeMarkdownHeading, renderTitledSections } from './triathlon-markdown-data'

const pad = (n: number): string => String(n).padStart(2, '0')

const clock = (s: number): string => {
  const t = Math.max(0, Math.round(s))
  return `${Math.floor(t / 60)}:${pad(t % 60)}`
}

const hms = (s: number): string => {
  const t = Math.max(0, Math.round(s))
  const h = Math.floor(t / 3600)
  const m = Math.floor((t % 3600) / 60)
  return h ? `${h}h${pad(m)}'` : `${Math.floor(t / 60)}:${pad(t % 60)}`
}

const distGloss = (sport: string, km: number): string =>
  sport === 'swim' ? `${Math.round(km * 1000).toLocaleString('en-US')} m` : `${km.toFixed(1)} km`

const paceGloss = (sport: string, km: number, s: number): string => {
  if (km <= 0 || s <= 0) return '—'
  if (sport === 'swim') return `${clock(s / (km * 10))}/100m`
  if (sport === 'bike') return `${(km / (s / 3600)).toFixed(1)} km/h`
  return `${clock(s / km)}/km`
}

const ageOn = (iso: string): number =>
  Math.floor(
    (new Date(`${iso}T00:00:00Z`).getTime() -
      new Date(`${ATHLETE.bornAnchor}T00:00:00Z`).getTime()) /
      (365.25 * 86400000),
  )

export interface FeedMarkdownOpts {
  details?: Record<string, StravaActivityDetail>
  baseUrl?: string
  generatedAt?: string
  title?: string
  sourcePath?: string
  scopePrefix?: string
  includeActivityDetails?: boolean
  includeRestDays?: boolean
}

interface ParsedDataFeed {
  source: string
  activitiesReversed: FeedActivityRow[]
  daysReversed: FeedDayRow[]
  weeksReversed: FeedWeekRow[]
}

let cachedParsedDataFeed: ParsedDataFeed | undefined
const staticSummaryTailCache = new WeakMap<Analytics, string>()
const activityMarkdownCache = new WeakMap<FeedActivityRow, WeakMap<object, Map<boolean, string>>>()
const activityMarkdownNoDetailCache = new WeakMap<FeedActivityRow, Map<boolean, string>>()

function parseDataFeed(dataFeed: string): ParsedDataFeed {
  if (cachedParsedDataFeed?.source === dataFeed) {
    return cachedParsedDataFeed
  }
  const rows = dataFeed
    .split('\n')
    .filter(Boolean)
    .map(line => JSON.parse(line) as { kind: string } & Record<string, unknown>)
  const activitiesReversed = (
    rows.filter(r => r.kind === 'activity') as unknown as FeedActivityRow[]
  )
    .slice()
    .reverse()
  const daysReversed = (rows.filter(r => r.kind === 'day') as unknown as FeedDayRow[])
    .slice()
    .reverse()
  const weeksReversed = (rows.filter(r => r.kind === 'week') as unknown as FeedWeekRow[])
    .slice()
    .reverse()
  cachedParsedDataFeed = { source: dataFeed, activitiesReversed, daysReversed, weeksReversed }
  return cachedParsedDataFeed
}

function renderStaticSummaryTail(analytics: Analytics): string {
  const cached = staticSummaryTailCache.get(analytics)
  if (cached !== undefined) return cached
  const m = analytics.meta
  const staticEntries: Array<[string, unknown]> = [
    ['loadShare', analytics.loadShare],
    ['form', analytics.risk],
    ['vo2max', analytics.engine.vo2max],
    ['body', analytics.body],
    ['thresholds', analytics.thresholds],
    ['bests', analytics.bests],
    ['criticalPower', analytics.powerCurve.criticalPower],
    ['criticalPowerYear', analytics.powerCurve.criticalPowerYear],
    ['upcoming', (analytics.events ?? []).filter(e => e.date >= m.today)],
    ['races', analytics.races],
    ['tests', analytics.tests],
  ]
  const rendered = staticEntries
    .map(([key, value]) =>
      renderTitledSections(value, { title: `summary.${key}`, headingDepth: 3 }),
    )
    .join('\n\n')
  staticSummaryTailCache.set(analytics, rendered)
  return rendered
}

function renderActivityMarkdown(
  a: FeedActivityRow,
  det: StravaActivityDetail | undefined,
  includeActivityDetails: boolean,
): string {
  if (det && typeof det === 'object') {
    const byDetail = activityMarkdownCache.get(a)?.get(det)?.get(includeActivityDetails)
    if (byDetail !== undefined) return byDetail
  } else {
    const byFlag = activityMarkdownNoDetailCache.get(a)?.get(includeActivityDetails)
    if (byFlag !== undefined) return byFlag
  }

  const full = includeActivityDetails
    ? { ...a, detail: det ?? null }
    : {
        ...a,
        hrZones: det?.hrZones ?? null,
        powerZones: det?.powerZones ?? null,
        strokes: det?.strokes ?? null,
      }
  const sauna = a.sport === 'sauna' ? a.sauna : null
  const gloss = (
    sauna
      ? [
          hms(a.movingTimeS),
          `${Math.round(sauna.temperatureC)} °C`,
          `${sauna.humidityPct}% RH`,
          sauna.cooldown,
          sauna.heatTrainingLoad != null ? `HTL ${sauna.heatTrainingLoad}` : null,
          a.avgHr != null ? `hr ${a.avgHr}` : null,
        ]
      : [
          distGloss(a.sport, a.distanceKm),
          hms(a.movingTimeS),
          paceGloss(a.sport, a.distanceKm, a.movingTimeS),
          a.avgHr != null ? `hr ${a.avgHr}` : null,
          a.avgWatts != null ? `${a.avgWatts}w` : null,
          `load ${a.load}`,
        ]
  )
    .filter(Boolean)
    .join(' · ')
  const title = `${a.date} · ${a.sport} · ${a.id}`
  const data = renderTitledSections(full, { title: `activities.${a.id}`, headingDepth: 4 })
  const block = `### ${escapeMarkdownHeading(title)}\n\n${gloss}\n\n${data}`

  if (det && typeof det === 'object') {
    let byDetail = activityMarkdownCache.get(a)
    if (!byDetail) {
      byDetail = new WeakMap()
      activityMarkdownCache.set(a, byDetail)
    }
    let byFlag = byDetail.get(det)
    if (!byFlag) {
      byFlag = new Map()
      byDetail.set(det, byFlag)
    }
    byFlag.set(includeActivityDetails, block)
  } else {
    let byFlag = activityMarkdownNoDetailCache.get(a)
    if (!byFlag) {
      byFlag = new Map()
      activityMarkdownNoDetailCache.set(a, byFlag)
    }
    byFlag.set(includeActivityDetails, block)
  }

  return block
}

export function buildFeedMarkdown(
  dataFeed: string,
  analytics: Analytics,
  opts: FeedMarkdownOpts = {},
): string {
  const parsed = parseDataFeed(dataFeed)
  const scopePrefix = opts.scopePrefix ?? ''
  const activities = scopePrefix
    ? parsed.activitiesReversed.filter(row => row.date.startsWith(scopePrefix))
    : parsed.activitiesReversed
  const days = parsed.daysReversed.filter(
    row =>
      (!scopePrefix || row.date.startsWith(scopePrefix)) &&
      (opts.includeRestDays !== false || row.sessions > 0),
  )
  const weeks = scopePrefix
    ? parsed.weeksReversed.filter(row => row.weekStart.startsWith(scopePrefix))
    : parsed.weeksReversed
  const details = opts.details ?? {}
  const m = analytics.meta
  const title = opts.title ?? 'triathlon activity feed'
  const sourcePath = opts.sourcePath ?? '/triathlon/on'
  const baseUrl = opts.baseUrl ? `https://${opts.baseUrl}` : ''
  const dates = activities.map(activity => activity.date).sort()

  const totals: Record<string, { count: number; km: number; sec: number }> = {}
  for (const a of activities) {
    const t = (totals[a.sport] ??= { count: 0, km: 0, sec: 0 })
    t.count += 1
    t.km += a.distanceKm
    t.sec += a.movingTimeS
  }
  const bySport = Object.fromEntries(
    Object.entries(totals).map(([sport, t]) => [
      sport,
      { count: t.count, km: Math.round(t.km * 10) / 10, time: hms(t.sec) },
    ]),
  )

  const dynamicSummary = {
    window: {
      from: dates[0] ?? m.windowFrom,
      to: dates[dates.length - 1] ?? m.windowTo,
      today: m.today,
      activities: activities.length,
    },
    athlete: {
      sex: ATHLETE.sex,
      born: ATHLETE.born,
      ageYears: ageOn(m.today),
      hrMax: ATHLETE.hrMax,
      vo2max: ATHLETE.vo2max,
      ftp: ATHLETE.ftp,
      goalWeightLb: ATHLETE.goalWeightLb,
      goalFtp: ATHLETE.goalFTP,
    },
    bySport,
  }

  const summaryMarkdown = `${renderTitledSections(dynamicSummary, { title: 'summary' })}\n\n${renderStaticSummaryTail(analytics)}`

  const header = [
    '---',
    `title: ${title}`,
    `source: ${baseUrl}${sourcePath}`,
    `permalink: ${baseUrl}${sourcePath}.md`,
    `generated: ${opts.generatedAt ?? m.today}`,
    'units: distance km (swim m), pace min/km, swim min/100m, speed km/h, time h:mm, weight kg, hr bpm, power w, sauna temperature °C, humidity %, HTL heat-training load',
    `description: Generated triathlon training data for ${scopePrefix || 'the full activity window'}, with exact measured values in hierarchical GFM tables.`,
    '---',
    '',
    `# ${title}`,
    '',
    `Training log for ${ATHLETE.sex === 'M' ? 'a male' : 'a'} athlete (age ${ageOn(m.today)}), ${activities.length} activities from ${dates[0] ?? m.windowFrom} to ${dates[dates.length - 1] ?? m.windowTo}. Each activity below carries hierarchical GFM tables of its measured numbers; the summary holds season totals, current form, thresholds, lab tests, and bests. All distances are kilometres, paces minutes per kilometre (swim per 100 m), speeds km/h, unless noted.`,
    '',
    summaryMarkdown,
    '',
    '## activities',
    '',
    'Newest first. Each block is one activity; fields are exact measured values (null when unrecorded).',
  ].join('\n')

  const includeActivityDetails = Boolean(opts.includeActivityDetails)
  const acts = activities
    .map(a => renderActivityMarkdown(a, details[String(a.id)], includeActivityDetails))
    .join('\n\n')

  return [
    header,
    '',
    acts,
    '',
    renderTitledSections(days, { title: 'daily' }),
    '',
    renderTitledSections(weeks, { title: 'weekly' }),
    '',
  ].join('\n')
}
