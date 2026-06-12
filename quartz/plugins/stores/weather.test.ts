import assert from 'node:assert/strict'
import test from 'node:test'
import {
  compassFromDegrees,
  parseWeatherCache,
  summarizeWeatherDays,
  weatherActivityFromHours,
  type WeatherActivity,
  type WeatherActivityCandidate,
  type WeatherHour,
} from './weather'

function candidate(values: Partial<WeatherActivityCandidate> = {}): WeatherActivityCandidate {
  return {
    activityId: 101,
    date: '2026-06-11',
    start: '2026-06-11T13:30:00.000Z',
    end: '2026-06-11T15:00:00.000Z',
    latitude: 43.64,
    longitude: -79.4,
    durationS: 5400,
    ...values,
  }
}

function hour(values: Partial<WeatherHour>): WeatherHour {
  return {
    forecastStart: '2026-06-11T13:00:00.000Z',
    windSpeed: 0,
    windDirection: null,
    windGust: null,
    temperature: null,
    ...values,
  }
}

test('compassFromDegrees returns 16-point compass labels', () => {
  assert.equal(compassFromDegrees(0), 'N')
  assert.equal(compassFromDegrees(44), 'NE')
  assert.equal(compassFromDegrees(226), 'SW')
  assert.equal(compassFromDegrees(359), 'N')
  assert.equal(compassFromDegrees(null), null)
})

test('weatherActivityFromHours weights wind by activity overlap', () => {
  const activity = weatherActivityFromHours(candidate(), [
    hour({
      forecastStart: '2026-06-11T13:00:00.000Z',
      windSpeed: 10,
      windDirection: 270,
      windGust: 20,
      temperature: 22,
    }),
    hour({
      forecastStart: '2026-06-11T14:00:00.000Z',
      windSpeed: 20,
      windDirection: 270,
      windGust: 26,
      temperature: 24,
    }),
  ])

  assert.equal(activity?.windKph, 17)
  assert.equal(activity?.windDir, 'W')
  assert.equal(activity?.windDirDeg, 270)
  assert.equal(activity?.windGustKph, 26)
  assert.equal(activity?.temperatureC, 23)
})

test('summarizeWeatherDays folds activity weather into duration-weighted day wind', () => {
  const first = weatherActivityFromHours(candidate({ activityId: 101, durationS: 3600 }), [
    hour({ windSpeed: 10, windDirection: 350, windGust: 19 }),
  ])
  const second = weatherActivityFromHours(
    candidate({
      activityId: 102,
      start: '2026-06-11T16:00:00.000Z',
      end: '2026-06-11T18:00:00.000Z',
      durationS: 7200,
    }),
    [
      hour({
        forecastStart: '2026-06-11T16:00:00.000Z',
        windSpeed: 20,
        windDirection: 10,
        windGust: 32,
      }),
    ],
  )
  const activities: Record<string, WeatherActivity> = {}
  if (first) activities[String(first.activityId)] = first
  if (second) activities[String(second.activityId)] = second

  const days = summarizeWeatherDays(activities)
  assert.equal(days['2026-06-11'].windKph, 17)
  assert.equal(days['2026-06-11'].windDir, 'N')
  assert.equal(days['2026-06-11'].windGustKph, 32)
  assert.equal(days['2026-06-11'].activityCount, 2)
})

test('parseWeatherCache keeps valid activities and recomputes day summaries', () => {
  const cache = parseWeatherCache({
    version: 1,
    lastSync: 100,
    activities: {
      good: {
        activityId: 101,
        date: '2026-06-11',
        start: '2026-06-11T13:00:00.000Z',
        end: '2026-06-11T14:00:00.000Z',
        latitude: 43.64,
        longitude: -79.4,
        durationS: 3600,
        windKph: 18,
        windDir: 'SW',
        windDirDeg: 225,
        windGustKph: 28,
        temperatureC: 24,
      },
      bad: { activityId: 102 },
    },
  })

  assert.equal(cache?.lastSync, 100)
  assert.equal(Object.keys(cache?.activities ?? {}).length, 1)
  assert.equal(cache?.days['2026-06-11'].windKph, 18)
})
