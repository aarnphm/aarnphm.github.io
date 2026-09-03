import assert from 'node:assert/strict'
import { createVerify, generateKeyPairSync } from 'node:crypto'
import test from 'node:test'
import {
  fetchWeatherKitAttribution,
  fetchWeatherKitHours,
  parseWeatherKitAttribution,
  parseWeatherKitHours,
  WeatherKitRequestError,
  weatherKitHourlyUrl,
  weatherKitToken,
} from './weather-kit'

function decodePart(part: string): unknown {
  return JSON.parse(Buffer.from(part, 'base64url').toString('utf8'))
}

function privateKeyPem(): string {
  const { privateKey } = generateKeyPairSync('ec', { namedCurve: 'P-256' })
  return privateKey.export({ format: 'pem', type: 'pkcs8' })
}

test('weatherKitToken signs an ES256 developer token', () => {
  const { privateKey, publicKey } = generateKeyPairSync('ec', { namedCurve: 'P-256' })
  const pem = privateKey.export({ format: 'pem', type: 'pkcs8' })
  const token = weatherKitToken(
    {
      teamId: 'TEAM123456',
      serviceId: 'xyz.aarnphm.weather',
      keyId: 'KEY1234567',
      privateKey: pem,
      tokenTtlS: 600,
    },
    1000,
  )
  const parts = token.split('.')
  assert.equal(parts.length, 3)
  assert.deepEqual(decodePart(parts[0]), {
    alg: 'ES256',
    kid: 'KEY1234567',
    id: 'TEAM123456.xyz.aarnphm.weather',
  })
  assert.deepEqual(decodePart(parts[1]), {
    iss: 'TEAM123456',
    iat: 1000,
    exp: 1600,
    sub: 'xyz.aarnphm.weather',
  })
  assert.equal(
    createVerify('SHA256')
      .update(`${parts[0]}.${parts[1]}`)
      .verify({ key: publicKey, dsaEncoding: 'ieee-p1363' }, Buffer.from(parts[2], 'base64url')),
    true,
  )
})

test('weatherKitHourlyUrl requests only hourly forecast data', () => {
  const url = new URL(
    weatherKitHourlyUrl({
      latitude: 43.64,
      longitude: -79.4,
      hourlyStart: '2026-06-11T13:00:00.000Z',
      hourlyEnd: '2026-06-11T15:00:00.000Z',
      timezone: 'America/Toronto',
      language: 'en',
    }),
  )
  assert.equal(url.origin, 'https://weatherkit.apple.com')
  assert.equal(url.pathname, '/api/v1/weather/en/43.64/-79.4')
  assert.equal(url.searchParams.get('dataSets'), 'forecastHourly')
  assert.equal(url.searchParams.get('timezone'), 'America/Toronto')
})

test('parseWeatherKitHours extracts valid hourly wind rows', () => {
  assert.deepEqual(
    parseWeatherKitHours({
      forecastHourly: {
        hours: [
          {
            forecastStart: '2026-06-11T14:00:00Z',
            windSpeed: 18.4,
            windDirection: 225,
            windGust: 30.1,
            humidity: 0.684,
            temperature: 24.2,
            uvIndex: 6,
            cloudCover: 0.42,
            pressure: 1014.2,
            daylight: true,
            conditionCode: 'PartlyCloudy',
            precipitationChance: 0.35,
            precipitationType: 'rain',
          },
          { forecastStart: '2026-06-11T15:00:00Z' },
        ],
      },
    }),
    [
      {
        forecastStart: '2026-06-11T14:00:00Z',
        windSpeed: 18.4,
        windDirection: 225,
        windGust: 30.1,
        relativeHumidity: 0.684,
        temperature: 24.2,
        uvIndex: 6,
        cloudCover: 0.42,
        pressure: 1014.2,
        daylight: true,
        conditionCode: 'PartlyCloudy',
        precipitationChance: 0.35,
        precipitationType: 'rain',
      },
    ],
  )
})

test('parseWeatherKitAttribution resolves official relative logo assets', () => {
  assert.deepEqual(
    parseWeatherKitAttribution({
      serviceName: 'Apple Weather',
      'logoLight@2x': '/assets/logo-light@2x.png',
      'logoDark@2x': '/assets/logo-dark@2x.png',
    }),
    {
      serviceName: 'Apple Weather',
      logoLightUrl: 'https://weatherkit.apple.com/assets/logo-light@2x.png',
      logoDarkUrl: 'https://weatherkit.apple.com/assets/logo-dark@2x.png',
      legalPageUrl: 'https://weatherkit.apple.com/legal-attribution.html',
    },
  )
  assert.equal(
    parseWeatherKitAttribution({
      serviceName: 'Apple Weather',
      'logoLight@2x': 'https://tracker.example/logo.png',
      'logoDark@2x': '/dark.png',
    }),
    null,
  )
})

test('fetchWeatherKitAttribution authenticates the REST request', async () => {
  const previousFetch = globalThis.fetch
  let authorization = ''
  const fakeFetch: typeof fetch = async (_input, init) => {
    authorization = new Headers(init?.headers).get('authorization') ?? ''
    return Response.json({
      serviceName: 'Apple Weather',
      'logoLight@2x': '/light.png',
      'logoDark@2x': '/dark.png',
    })
  }
  globalThis.fetch = fakeFetch
  try {
    const attribution = await fetchWeatherKitAttribution(
      {
        teamId: 'TEAM123456',
        serviceId: 'xyz.aarnphm.weather',
        keyId: 'KEY1234567',
        privateKey: privateKeyPem(),
      },
      'en',
    )
    assert.equal(attribution.serviceName, 'Apple Weather')
    assert.match(authorization, /^Bearer [^.]+\.[^.]+\.[^.]+$/)
  } finally {
    globalThis.fetch = previousFetch
  }
})

test('parseWeatherKitHours retains humidity without wind and rejects values outside zero to one', () => {
  assert.deepEqual(
    parseWeatherKitHours({
      forecastHourly: {
        hours: [
          { forecastStart: '2026-06-11T13:00:00Z', humidity: 0 },
          { forecastStart: '2026-06-11T14:00:00Z', humidity: 1, windSpeed: 18 },
          { forecastStart: '2026-06-11T15:00:00Z', humidity: -0.01, windSpeed: 19 },
          { forecastStart: '2026-06-11T16:00:00Z', humidity: 1.01, temperature: 24 },
        ],
      },
    }).map(hour => ({
      forecastStart: hour.forecastStart,
      windSpeed: hour.windSpeed,
      relativeHumidity: hour.relativeHumidity,
    })),
    [
      { forecastStart: '2026-06-11T13:00:00Z', windSpeed: null, relativeHumidity: 0 },
      { forecastStart: '2026-06-11T14:00:00Z', windSpeed: 18, relativeHumidity: 1 },
      { forecastStart: '2026-06-11T15:00:00Z', windSpeed: 19, relativeHumidity: null },
      { forecastStart: '2026-06-11T16:00:00Z', windSpeed: null, relativeHumidity: null },
    ],
  )
})

test('parseWeatherKitHours rejects out-of-range route-hour conditions independently', () => {
  const [hour] = parseWeatherKitHours({
    forecastHourly: {
      hours: [
        {
          forecastStart: '2026-06-11T13:00:00Z',
          temperature: 20,
          uvIndex: 31,
          cloudCover: -0.1,
          windSpeed: -1,
          windDirection: 361,
          windGust: 501,
          pressure: 2_001,
          precipitationChance: 1.1,
        },
      ],
    },
  })

  assert.ok(hour)
  assert.equal(hour.temperature, 20)
  assert.equal(hour.uvIndex, null)
  assert.equal(hour.cloudCover, null)
  assert.equal(hour.windSpeed, null)
  assert.equal(hour.windDirection, null)
  assert.equal(hour.windGust, null)
  assert.equal(hour.pressure, null)
  assert.equal(hour.precipitationChance, null)
})

test('fetchWeatherKitHours exposes WeatherKit HTTP status', async () => {
  const previousFetch = globalThis.fetch
  const fakeFetch: typeof fetch = async () => new Response('denied', { status: 403 })
  globalThis.fetch = fakeFetch
  try {
    await assert.rejects(
      fetchWeatherKitHours(
        {
          teamId: 'TEAM123456',
          serviceId: 'xyz.aarnphm.weather',
          keyId: 'KEY1234567',
          privateKey: privateKeyPem(),
        },
        {
          latitude: 43.64,
          longitude: -79.4,
          hourlyStart: '2026-06-11T13:00:00.000Z',
          hourlyEnd: '2026-06-11T15:00:00.000Z',
          timezone: 'America/Toronto',
          language: 'en',
        },
      ),
      (err: unknown) => {
        assert.ok(err instanceof WeatherKitRequestError)
        assert.equal(err.status, 403)
        return true
      },
    )
  } finally {
    globalThis.fetch = previousFetch
  }
})
