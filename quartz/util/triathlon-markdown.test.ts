import assert from 'node:assert/strict'
import test from 'node:test'
import { buildAnalytics } from '../plugins/stores/analytics'
import { emptyPayload, type StravaActivityDetail } from '../plugins/stores/strava'
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

const activity = (
  id: number,
  date: string,
  name: string,
  sport = 'run',
): Record<string, unknown> => ({
  kind: 'activity',
  id,
  date,
  sport,
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
  sauna: null,
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

const activityDetail = (): StravaActivityDetail => ({
  id: 1,
  sport: 'run',
  name: 'current activity',
  date: '2026-07-30',
  start: '2026-07-30T12:00:00Z',
  distanceKm: 10,
  movingTimeS: 3_000,
  maxSpeedKph: 18,
  elevationM: 80,
  avgHr: 150,
  maxHr: 175,
  avgWatts: 250,
  npWatts: 260,
  maxWatts: 500,
  kilojoules: 750,
  deviceWatts: true,
  avgCadence: 88,
  sufferScore: 80,
  calories: 700,
  avgTemp: 20,
  windKph: 10,
  windDir: 'W',
  windDirDeg: 270,
  windGustKph: 15,
  location: 'Toronto',
  fueling: null,
  strength: null,
  sauna: null,
  garmin: null,
  calculatedIntensityFactor: null,
  calculatedExerciseLoad: null,
  calculatedTrainingEffect: null,
  gearShifts: [],
  cyclingDynamics: null,
  route: [],
  heartRateTrace: [],
  mapRoute: [],
  analysisRanges: [],
  runSplitsMetric: [],
  runSplitsStandard: [],
  runPaceZones: null,
  minAlt: 70,
  maxAlt: 100,
  descentM: 60,
  hrZones: [0, 300, 900, 1_200, 600],
  powerZones: [0, 250, 750, 1_200, 600],
  powerHist: null,
  powerWithoutZeros: null,
  powerCurve: [{ s: 30, w: 500, activityId: 1, activityDate: '2026-07-30' }],
  activityCriticalPower: null,
  bestEfforts: null,
  strokes: null,
  strokeCount: null,
  strokeRateSpm: null,
  swimPaceSPer100m: null,
  swimPaceSource: null,
  swimDurationS: null,
  swimIntervals: [],
  swimLocation: null,
  waterTemperatureC: null,
})

const dataFeed = [
  activity(1, '2026-07-30', 'current activity'),
  activity(2, '2025-07-30', 'old activity'),
  activity(3, '2026-07-28', 'strength activity', 'strength'),
  { kind: 'day', date: '2026-07-30', sessions: 1 },
  { kind: 'day', date: '2026-07-29', sessions: 0 },
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
        services: [
          { bike: 'soloist', date: '2026-08-20', distanceMiles: 1721.5, place: 'Racer Sportif' },
        ],
        components: [
          {
            component: 'OSPW',
            type: 'CeramicSpeed OSPW RS 5 Spoke',
            distanceMiles: null,
            ranges: [{ start: '2026-08-10', end: null }],
            reason: null,
          },
          {
            component: 'bottom bracket',
            type: 'FSA T47 BBright',
            distanceMiles: 1721.5,
            ranges: [{ start: '2026-05-16', end: '2026-08-20' }],
            reason: 'upgraded to CeramicSpeed',
          },
        ],
        chains: [
          {
            id: '3',
            distanceMiles: null,
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
            distanceMiles: null,
            ranges: [
              { start: '2026-07-16', end: '2026-08-10' },
              { start: '2026-08-18', end: null },
            ],
            reason: 'punctures, repaired',
            repaired: true,
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
  assert.match(analytics, /## relatedData\n\n\| field \| value \|/)
  assert.match(analytics, /## analytics/)
  assert.match(analytics, /### analytics\.calibration/)
  assert.match(analytics, /\| weakestSport \| "run" \|/)
  assert.match(calculator, /## calculatorInputs/)
  assert.match(calculator, /### calculatorInputs\.presets/)
  assert.match(calculator, /\| arrayIndex \| label \| swimKm \| bikeKm \| runKm \|/)
  assert.match(calculator, /### calculatorInputs\.calibration/)
  assert.doesNotMatch(analytics, /```json/)
  assert.doesNotMatch(calculator, /```json/)
})

test('turns generated training HTML and tool constants into markdown', () => {
  const training = buildTriathlonMarkdown(options('training', 'triathlon/training'))
  const tools = buildTriathlonMarkdown(options('tools', 'triathlon/tools'))

  assert.match(training, /## Olympic build/)
  assert.match(training, /\| id \| distance \| date \| target \| author \|/)
  assert.match(training, /\| "plan-0" \| "Olympic" \| "2026-08-01" \| "finish" \| "Aaron" \|/)
  assert.match(training, /## Week one/)
  assert.match(training, /Ride easy\./)
  assert.match(training, /\| day \| session \|\n\| --- \| --- \|/)
  assert.match(training, /\| Monday \| Swim<br>easy \|/)
  assert.match(tools, /\| arrayIndex \| distance \| swimKm \| bikeKm \| runKm \|/)
  assert.match(tools, /units: race distance km, maintenance distance mi/)
  assert.match(tools, /## maintenance/)
  assert.match(tools, /### maintenance\.services/)
  assert.match(tools, /\| 1 \| "soloist" \| "2026-08-20" \| "Racer Sportif" \| 1721\.5 \|/)
  assert.match(tools, /### maintenance\.components/)
  assert.match(
    tools,
    /\| 1 \| "OSPW" \| "CeramicSpeed OSPW RS 5 Spoke" \| "2026-08-10 to current" \| null \| null \|/,
  )
  assert.match(
    tools,
    /\| 2 \| "bottom bracket" \| "FSA T47 BBright" \| "2026-05-16 to 2026-08-20" \| 1721\.5 \| "upgraded to CeramicSpeed" \|/,
  )
  assert.match(tools, /### maintenance\.chains/)
  assert.match(tools, /\| 1 \| "3" \| "UFO Wax Drip-On" \| "2026-08-10" \| null \| true \|/)
  assert.match(tools, /### maintenance\.tires/)
  assert.match(
    tools,
    /\| 1 \| "rear" \| "tire" \| "Pirelli P Zero Race SL-R" \| "2026-07-16 to 2026-08-10, 2026-08-18 to current" \| null \| true \| "punctures, repaired" \|/,
  )
  assert.match(tools, /\| arrayIndex \| category \| item \|/)
  assert.match(tools, /\| 1 \| "bike" \| "Cervelo Soloist" \|/)
  assert.doesNotMatch(training, /```json/)
  assert.doesNotMatch(tools, /```json/)
})

test('neutralizes authored text in generated training headings', () => {
  const trainingOptions = options('training', 'triathlon/training')
  trainingOptions.plans[0].meta = 'Olympic\n# forged <script>'

  const training = buildTriathlonMarkdown(trainingOptions)

  assert.match(training, /^## Olympic \\# forged \\<script\\>$/m)
  assert.doesNotMatch(training, /<script>/)
  assert.doesNotMatch(training, /^# forged/m)
})

test('scopes generated archive and day notes to their route', () => {
  const archiveOptions = options('on', 'triathlon/on/2026', 'triathlon · 2026')
  archiveOptions.scopePrefix = '2026'
  const archive = buildTriathlonMarkdown(archiveOptions)
  const dayOptions = options('day', 'triathlon/on/2026/07/30', 'triathlon · 2026-07-30')
  dayOptions.scopePrefix = '2026-07-30'
  dayOptions.payload.details['1'] = activityDetail()
  const day = buildTriathlonMarkdown(dayOptions)

  assert.match(archive, /current activity/)
  assert.match(archive, /\| 1 \| "day" \| "2026-07-29" \| 0 \|/)
  assert.doesNotMatch(archive, /old activity/)
  assert.match(day, /#### activities\.1/)
  assert.match(day, /##### activities\.1\.detail/)
  assert.match(day, /###### activities\.1\.detail\.powerCurve/)
  assert.match(day, /\| 1 \| 30 \| 500 \| 1 \| "2026-07-30" \|/)
  assert.match(day, /\| activities \| 1 \|/)
  assert.doesNotMatch(day, /old activity/)
  assert.doesNotMatch(archive, /```json/)
  assert.doesNotMatch(day, /```json/)
})

test('keeps the complete triathlon feed markdown route', () => {
  const feed = buildTriathlonMarkdown(options('feed', 'triathlon/feed'))

  assert.match(feed, /permalink: https:\/\/aarnphm\.xyz\/triathlon\/feed\.md/)
  assert.match(feed, /current activity/)
  assert.match(feed, /old activity/)
  assert.match(feed, /strength activity/)
  assert.doesNotMatch(feed, /\| \d+ \| "day" \| "2026-07-29" \|/)
  assert.match(feed, /## daily/)
  assert.match(feed, /## weekly/)
  assert.doesNotMatch(feed, /```json/)
})

test('emits a map note with a raw-data pointer when no routes exist', () => {
  const maps = buildTriathlonMarkdown(options('maps', 'triathlon/maps'))

  assert.match(maps, /## mappedActivities/)
  assert.match(maps, /\| activityCount \| 0 \|/)
  assert.match(maps, /### mappedActivities\.activities/)
  assert.match(maps, /\| empty \| \[\] \|/)
  assert.match(maps, /https:\/\/aarnphm\.xyz\/static\/strava-detail\.json/)
  assert.doesNotMatch(maps, /```json/)
})
