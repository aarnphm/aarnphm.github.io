import assert from 'node:assert/strict'
import { mkdir, writeFile } from 'node:fs/promises'
import path from 'node:path'
import test from 'node:test'
import { nextSunEvent, readVisitorForecast } from '../util/visitor-weather'

const base = process.env.VISITOR_WEATHER_BASE_URL ?? 'http://localhost:8080'
const evidence = process.env.VISITOR_WEATHER_EVIDENCE

test('visitor weather returns a usable local forecast and the next solar transition over HTTP', async () => {
  const response = await fetch(new URL('/api/weather', base))
  assert.equal(response.status, 200)
  assert.match(response.headers.get('cache-control') ?? '', /^private,/)
  const body: unknown = await response.json()
  const forecast = readVisitorForecast(body)
  assert.ok(forecast, 'the HTTP response must satisfy the visitor forecast contract')
  assert.ok(forecast.place, 'this local Worker supplies Cloudflare visitor geography')
  assert.ok(forecast.weather, 'the live Open-Meteo response must contain current weather')
  assert.ok(forecast.solar, 'the live Open-Meteo response must contain solar times')
  assert.ok(
    forecast.lat !== null && Math.abs(forecast.lat * 10 - Math.round(forecast.lat * 10)) < 1e-8,
  )
  assert.ok(
    forecast.lon !== null && Math.abs(forecast.lon * 10 - Math.round(forecast.lon * 10)) < 1e-8,
  )
  assert.ok(forecast.weather.cloud >= 0 && forecast.weather.cloud <= 1)
  assert.ok(forecast.weather.wind >= 0)
  assert.ok(
    forecast.weather.snow !== null && forecast.weather.snow >= 0,
    'the live Open-Meteo response must carry the last twelve hours of snowfall',
  )
  new Intl.DateTimeFormat('en', { timeZone: forecast.solar.timezone }).format()
  const now = Date.now()
  const next = nextSunEvent(forecast.solar, now)
  assert.ok(next, 'Toronto must have an upcoming sunrise or sunset')
  assert.ok(next.at > now && next.at - now < 86400000)

  if (evidence) {
    await mkdir(evidence, { recursive: true })
    await writeFile(path.join(evidence, 'forecast.json'), JSON.stringify(body, null, 2) + '\n')
    await writeFile(
      path.join(evidence, 'http-check.json'),
      JSON.stringify(
        { base, checkedAt: new Date(now).toISOString(), status: response.status, next },
        null,
        2,
      ) + '\n',
    )
  }
})

test('visitor weather rejects writes through its HTTP entry point', async () => {
  const response = await fetch(new URL('/api/weather', base), { method: 'POST' })
  assert.equal(response.status, 405)
  assert.equal(response.headers.get('allow'), 'GET')
})
