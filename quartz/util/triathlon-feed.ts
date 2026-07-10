import type { StravaActivityDetail } from '../plugins/stores/strava'
import {
  ATHLETE,
  type Analytics,
  type FeedActivityRow,
  type FeedDayRow,
  type FeedWeekRow,
} from '../plugins/stores/analytics'

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

const block = (label: string, value: unknown): string =>
  `## ${label}\n\n\`\`\`json\n${JSON.stringify(value, null, 2)}\n\`\`\``

export interface FeedMarkdownOpts {
  details?: Record<string, StravaActivityDetail>
  baseUrl?: string
  generatedAt?: string
}

export function buildFeedMarkdown(
  dataFeed: string,
  analytics: Analytics,
  opts: FeedMarkdownOpts = {},
): string {
  const rows = dataFeed
    .split('\n')
    .filter(Boolean)
    .map(line => JSON.parse(line) as { kind: string } & Record<string, unknown>)
  const activities = (rows.filter(r => r.kind === 'activity') as unknown as FeedActivityRow[])
    .slice()
    .reverse()
  const days = (rows.filter(r => r.kind === 'day') as unknown as FeedDayRow[]).slice().reverse()
  const weeks = (rows.filter(r => r.kind === 'week') as unknown as FeedWeekRow[]).slice().reverse()
  const details = opts.details ?? {}
  const m = analytics.meta

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

  const summary = {
    window: { from: m.windowFrom, to: m.windowTo, today: m.today, activities: m.activityCount },
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
    loadShare: analytics.loadShare,
    form: analytics.risk,
    vo2max: analytics.engine.vo2max,
    body: analytics.body,
    thresholds: analytics.thresholds,
    bests: analytics.bests,
    upcoming: (analytics.events ?? []).filter(e => e.date >= m.today),
    races: analytics.races,
    tests: analytics.tests,
  }

  const header = [
    '---',
    'title: triathlon activity feed',
    `source: ${opts.baseUrl ? `https://${opts.baseUrl}` : ''}/triathlon/on`,
    `generated: ${opts.generatedAt ?? m.today}`,
    'units: distance km (swim m), pace min/km, swim min/100m, speed km/h, time h:mm, weight kg, hr bpm, power w',
    'description: Full triathlon training feed — season summary plus every activity with its numbers as JSON, for machine reading.',
    '---',
    '',
    '# triathlon activity feed',
    '',
    `Training log for ${ATHLETE.sex === 'M' ? 'a male' : 'a'} athlete (age ${ageOn(m.today)}), ${m.activityCount} activities from ${m.windowFrom} to ${m.windowTo}. Each activity below carries a JSON block of its measured numbers; the summary holds season totals, current form, thresholds, lab tests, and bests. All distances are kilometres, paces minutes per kilometre (swim per 100 m), speeds km/h, unless noted.`,
    '',
    block('summary', summary),
    '',
    '## activities',
    '',
    'Newest first. Each block is one activity; fields are exact measured values (null when unrecorded).',
  ].join('\n')

  const acts = activities
    .map(a => {
      const det = details[String(a.id)]
      const full = {
        ...a,
        hrZones: det?.hrZones ?? null,
        powerZones: det?.powerZones ?? null,
        strokes: det?.strokes ?? null,
      }
      const gloss = [
        distGloss(a.sport, a.distanceKm),
        hms(a.movingTimeS),
        paceGloss(a.sport, a.distanceKm, a.movingTimeS),
        a.avgHr != null ? `hr ${a.avgHr}` : null,
        a.avgWatts != null ? `${a.avgWatts}w` : null,
        `load ${a.load}`,
      ]
        .filter(Boolean)
        .join(' · ')
      return `### ${a.date} · ${a.sport} · ${a.name || a.sport}\n\n${gloss}\n\n\`\`\`json\n${JSON.stringify(full, null, 2)}\n\`\`\``
    })
    .join('\n\n')

  return [header, '', acts, '', block('daily', days), '', block('weekly', weeks), ''].join('\n')
}
