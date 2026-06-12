import { createSign } from 'node:crypto'
import type { WeatherHour } from '../plugins/stores/weather'
import { isRecord, readNumber, readString } from './type-guards'

const BASE_URL = 'https://weatherkit.apple.com'

export interface WeatherKitConfig {
  teamId: string
  serviceId: string
  keyId: string
  privateKey: string
  tokenTtlS?: number
}

export interface WeatherKitHourlyRequest {
  latitude: number
  longitude: number
  hourlyStart: string
  hourlyEnd: string
  timezone: string
  language: string
}

export class WeatherKitRequestError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message)
  }
}

function base64urlJson(value: unknown): string {
  return Buffer.from(JSON.stringify(value)).toString('base64url')
}

export function weatherKitToken(
  config: WeatherKitConfig,
  nowS = Math.floor(Date.now() / 1000),
): string {
  const ttl = Math.max(60, Math.min(config.tokenTtlS ?? 1800, 3600))
  const header = { alg: 'ES256', kid: config.keyId, id: `${config.teamId}.${config.serviceId}` }
  const payload = { iss: config.teamId, iat: nowS, exp: nowS + ttl, sub: config.serviceId }
  const signingInput = `${base64urlJson(header)}.${base64urlJson(payload)}`
  const signature = createSign('SHA256')
    .update(signingInput)
    .sign({ key: config.privateKey, dsaEncoding: 'ieee-p1363' })
  return `${signingInput}.${signature.toString('base64url')}`
}

export function parseWeatherKitHours(raw: unknown): WeatherHour[] {
  if (!isRecord(raw) || !isRecord(raw.forecastHourly)) return []
  const hours = raw.forecastHourly.hours
  if (!Array.isArray(hours)) return []
  const out: WeatherHour[] = []
  for (const item of hours) {
    if (!isRecord(item)) continue
    const forecastStart = readString(item, 'forecastStart')
    const windSpeed = readNumber(item, 'windSpeed')
    if (!forecastStart || windSpeed == null) continue
    out.push({
      forecastStart,
      windSpeed,
      windDirection: readNumber(item, 'windDirection') ?? null,
      windGust: readNumber(item, 'windGust') ?? null,
      temperature: readNumber(item, 'temperature') ?? null,
    })
  }
  return out.sort((a, b) => a.forecastStart.localeCompare(b.forecastStart))
}

export function weatherKitHourlyUrl(request: WeatherKitHourlyRequest): string {
  const url = new URL(
    `/api/v1/weather/${request.language}/${request.latitude}/${request.longitude}`,
    BASE_URL,
  )
  url.searchParams.set('dataSets', 'forecastHourly')
  url.searchParams.set('hourlyStart', request.hourlyStart)
  url.searchParams.set('hourlyEnd', request.hourlyEnd)
  url.searchParams.set('timezone', request.timezone)
  return url.toString()
}

export async function fetchWeatherKitHours(
  config: WeatherKitConfig,
  request: WeatherKitHourlyRequest,
): Promise<WeatherHour[]> {
  const response = await fetch(weatherKitHourlyUrl(request), {
    headers: { Authorization: `Bearer ${weatherKitToken(config)}` },
  })
  if (!response.ok) {
    const text = await response.text()
    throw new WeatherKitRequestError(
      response.status,
      `WeatherKit ${response.status}: ${text.slice(0, 200)}`,
    )
  }
  return parseWeatherKitHours(await response.json())
}
