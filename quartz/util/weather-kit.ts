import { createSign } from 'node:crypto'
import type { WeatherAttribution, WeatherHour } from '../plugins/stores/weather'
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
    const boundedNumber = (key: string, minimum: number, maximum: number): number | null => {
      const value = readNumber(item, key)
      return value != null && value >= minimum && value <= maximum ? value : null
    }
    const windSpeed = boundedNumber('windSpeed', 0, 500)
    const windDirection = boundedNumber('windDirection', 0, 360)
    const windGust = boundedNumber('windGust', 0, 500)
    const temperature = boundedNumber('temperature', -100, 100)
    const rawUvIndex = readNumber(item, 'uvIndex')
    const uvIndex = rawUvIndex != null && rawUvIndex >= 0 && rawUvIndex <= 30 ? rawUvIndex : null
    const rawCloudCover = readNumber(item, 'cloudCover')
    const cloudCover =
      rawCloudCover != null && rawCloudCover >= 0 && rawCloudCover <= 1 ? rawCloudCover : null
    const pressure = boundedNumber('pressure', 0, 2_000)
    const daylight = typeof item.daylight === 'boolean' ? item.daylight : null
    const rawRelativeHumidity = readNumber(item, 'humidity')
    const relativeHumidity =
      rawRelativeHumidity != null && rawRelativeHumidity >= 0 && rawRelativeHumidity <= 1
        ? rawRelativeHumidity
        : null
    if (
      !forecastStart ||
      (windSpeed == null &&
        temperature == null &&
        relativeHumidity == null &&
        uvIndex == null &&
        cloudCover == null &&
        pressure == null &&
        daylight == null)
    )
      continue
    out.push({
      forecastStart,
      windSpeed: windSpeed ?? null,
      windDirection,
      windGust,
      relativeHumidity,
      temperature: temperature ?? null,
      uvIndex,
      cloudCover,
      pressure,
      daylight,
      conditionCode: readString(item, 'conditionCode') ?? null,
      precipitationChance: boundedNumber('precipitationChance', 0, 1),
      precipitationType: readString(item, 'precipitationType') ?? null,
    })
  }
  return out.sort((a, b) => a.forecastStart.localeCompare(b.forecastStart))
}

const weatherKitAssetUrl = (value: unknown): string | null => {
  if (typeof value !== 'string' || value.length === 0) return null
  if (!URL.canParse(value, BASE_URL)) return null
  const url = new URL(value, BASE_URL)
  return url.origin === BASE_URL ? url.toString() : null
}

export function weatherKitAttributionUrl(language: string): string {
  return new URL(`/attribution/${encodeURIComponent(language)}`, BASE_URL).toString()
}

export function parseWeatherKitAttribution(raw: unknown): WeatherAttribution | null {
  if (!isRecord(raw)) return null
  const serviceName = readString(raw, 'serviceName')
  const logoLightUrl = weatherKitAssetUrl(raw['logoLight@2x'])
  const logoDarkUrl = weatherKitAssetUrl(raw['logoDark@2x'])
  if (!serviceName || !logoLightUrl || !logoDarkUrl) return null
  return {
    serviceName,
    logoLightUrl,
    logoDarkUrl,
    legalPageUrl: `${BASE_URL}/legal-attribution.html`,
  }
}

export async function fetchWeatherKitAttribution(
  config: WeatherKitConfig,
  language: string,
): Promise<WeatherAttribution> {
  const response = await fetch(weatherKitAttributionUrl(language), {
    headers: { Authorization: `Bearer ${weatherKitToken(config)}` },
  })
  if (!response.ok) {
    const body = await response.text()
    throw new WeatherKitRequestError(
      response.status,
      `WeatherKit attribution ${response.status}: ${body.slice(0, 200)}`,
    )
  }
  const attribution = parseWeatherKitAttribution(await response.json())
  if (!attribution) throw new Error('WeatherKit returned malformed attribution')
  return attribution
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
