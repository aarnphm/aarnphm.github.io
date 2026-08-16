import assert from 'node:assert/strict'
import test from 'node:test'
import { buildAnalytics } from '../plugins/stores/analytics'
import { emptyPayload } from '../plugins/stores/strava'
import { isFullSlug, type FullSlug } from './path'
import {
  buildTriathlonMarkdown,
  type TriathlonMarkdownOptions,
  type TriathlonMarkdownView,
} from './triathlon-markdown'

const fullSlug = (value: string): FullSlug => {
  if (!isFullSlug(value)) throw new Error(`invalid slug: ${value}`)
  return value
}

const activity = (id: number, date: string, name: string): Record<string, unknown> => ({
  kind: 'activity',
  id,
  date,
  sport: 'run',
  name,
  distanceKm: 10,
  movingTimeS: 3000,
  elapsedTimeS: 3060,
  elevationM: 80,
  avgHr: 150,
  maxHr: 175,
  avgWatts: 250,
  weightedWatts: 260,
  deviceWatts: true,
  cadence: 88,
  strokes: null,
  calories: 700,
  sufferScore: 80,
  avgTemp: 20,
  windKph: 10,
  windDir: 'W',
  windGustKph: 15,
  skipTraining: false,
  vGap: 0,
  intensity: 0.8,
  load: 75,
  pp30: null,
  pp60: null,
  pp300: null,
  pp1200: null,
  ps30: null,
  ps60: null,
  ps300: null,
  ps1200: null,
  ef: null,
  decoupling: null,
})

const dataFeed = [
  activity(1, '2026-07-30', 'current activity'),
  activity(2, '2025-07-30', 'old activity'),
  { kind: 'day', date: '2026-07-30', sessions: 1 },
  { kind: 'day', date: '2025-07-30', sessions: 1 },
  { kind: 'week', weekStart: '2026-07-27', sessions: 1 },
  { kind: 'week', weekStart: '2025-07-28', sessions: 1 },
]
  .map(row => JSON.stringify(row))
  .join('\n')

const options = (
  view: TriathlonMarkdownView,
  slug: string,
  title = `triathlon · ${view}`,
): TriathlonMarkdownOptions => {
  const payload = emptyPayload()
  payload.generatedAt = Date.parse('2026-07-31T12:00:00Z')
  const analytics = buildAnalytics(null)
  analytics.meta.today = '2026-07-31'
  return {
    view,
    slug: fullSlug(slug),
    title,
    description: `Generated ${view} data.`,
    baseUrl: 'aarnphm.xyz',
    dataFeed,
    analytics,
    payload,
    plans: [
      {
        id: 'plan-0',
        meta: 'Olympic build',
        distance: 'Olympic',
        date: '2026-08-01',
        target: 'finish',
        author: 'Aaron',
        html: '<h2>Week one</h2><p>Ride easy.</p><table><thead><tr><th>day</th><th>session</th></tr></thead><tbody><tr><td>Monday</td><td>Swim<br>easy</td></tr></tbody></table>',
      },
    ],
    tools: {
      conversions: [['pace', '/100m x 16.09 -> /mi']],
      gear: [['bike', ['Cervelo Soloist']]],
      maintenance: {
        chains: [
          {
            id: '3',
            distance: null,
            lubricant: 'UFO Wax Drip-On',
            since: '2026-08-10',
            waxed: true,
          },
        ],
        wheels: [
          {
            position: 'rear',
            part: 'tire',
            type: 'Pirelli P Zero Race SL-R',
            distance: null,
            start: '2026-08-12',
            end: null,
            reason: null,
          },
        ],
      },
    },
  }
}

test('builds exact analytics and calculator notes with stable permalinks', () => {
  const analytics = buildTriathlonMarkdown(options('analytics', 'triathlon/analytics'))
  const calculator = buildTriathlonMarkdown(options('calc', 'triathlon/calc'))

  assert.match(analytics, /permalink: https:\/\/aarnphm\.xyz\/triathlon\/analytics\.md/)
  assert.match(analytics, /## analytics\n\n```json/)
  assert.match(analytics, /"calibration"/)
  assert.match(calculator, /## calculator inputs/)
  assert.match(calculator, /"presets"/)
  assert.match(calculator, /"ftpHypothesis"/)
})

test('turns generated training HTML and tool constants into markdown', () => {
  const training = buildTriathlonMarkdown(options('training', 'triathlon/training'))
  const tools = buildTriathlonMarkdown(options('tools', 'triathlon/tools'))

  assert.match(training, /## Olympic build/)
  assert.match(training, /## Week one/)
  assert.match(training, /Ride easy\./)
  assert.match(training, /\| day \| session \|\n\| --- \| --- \|/)
  assert.match(training, /\| Monday \| Swim<br>easy \|/)
  assert.match(tools, /\| distance \| swim km \| bike km \| run km \|/)
  assert.match(tools, /## maintenance/)
  assert.match(tools, /chain 3: UFO Wax Drip-On; since 2026-08-10; waxed yes/)
  assert.match(tools, /rear tire: Pirelli P Zero Race SL-R; 2026-08-12 to current/)
  assert.match(tools, /### bike\n\n- Cervelo Soloist/)
})

test('scopes generated archive and day notes to their route', () => {
  const archiveOptions = options('on', 'triathlon/on/2026', 'triathlon · 2026')
  archiveOptions.scopePrefix = '2026'
  const archive = buildTriathlonMarkdown(archiveOptions)
  const dayOptions = options('day', 'triathlon/on/2026/07/30', 'triathlon · 2026-07-30')
  dayOptions.scopePrefix = '2026-07-30'
  const day = buildTriathlonMarkdown(dayOptions)

  assert.match(archive, /current activity/)
  assert.doesNotMatch(archive, /old activity/)
  assert.match(day, /"detail": null/)
  assert.match(day, /"activities": 1/)
  assert.doesNotMatch(day, /old activity/)
})

test('keeps the complete triathlon feed markdown route', () => {
  const feed = buildTriathlonMarkdown(options('feed', 'triathlon/feed'))

  assert.match(feed, /permalink: https:\/\/aarnphm\.xyz\/triathlon\/feed\.md/)
  assert.match(feed, /current activity/)
  assert.match(feed, /old activity/)
})

test('emits a map note with a raw-data pointer when no routes exist', () => {
  const maps = buildTriathlonMarkdown(options('maps', 'triathlon/maps'))

  assert.match(maps, /## mapped activities/)
  assert.match(maps, /"activityCount": 0/)
  assert.match(maps, /https:\/\/aarnphm\.xyz\/static\/strava-detail\.json/)
})
