import assert from 'node:assert/strict'
import { createVerify, generateKeyPairSync } from 'node:crypto'
import test from 'node:test'
import {
  fetchWeatherKitHours,
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
            temperature: 24.2,
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
        temperature: 24.2,
      },
    ],
  )
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
