export type UvSeverity = 'negligible' | 'low' | 'moderate' | 'high' | 'serious' | 'extreme'

export interface ProviderNativeProvenance {
  source: 'provider-native'
  transport: 'strava-description'
  schemaVersion: 1
  activityId: number
  retrievedAt: number
}

export interface PelotanNativeReport extends ProviderNativeProvenance {
  provider: 'pelotan'
  score: number | null
  rawBand: string | null
  severity: UvSeverity | null
  averageUvIndex: number | null
  averageTemperatureC: number | null
  averageCloudCoverPct: number | null
}

export interface MyWindsockNativeReport extends ProviderNativeProvenance {
  provider: 'mywindsock'
  weatherImpactPct: number | null
  cdaM2: number | null
  feelsLikeElevationM: number | null
  headwindPct: number | null
  headwindMinKph: number | null
  headwindMaxKph: number | null
  longestHeadwindS: number | null
  airSpeedKph: number | null
  averageTemperatureC: number | null
  precipitationProbabilityPct: number | null
  precipitationRateMmPerHour: number | null
}

export interface NativeActivityReports {
  myWindsock: MyWindsockNativeReport | null
  pelotan: PelotanNativeReport | null
}

const number = (value: string | undefined): number | null => {
  if (value == null) return null
  const parsed = Number(value.replace(',', '.'))
  return Number.isFinite(parsed) ? parsed : null
}

const rounded = (value: number, digits = 3): number => {
  const factor = 10 ** digits
  return Math.round(value * factor) / factor
}

const bounded = (value: number | null, minimum: number, maximum: number): number | null =>
  value != null && value >= minimum && value <= maximum ? value : null

const delimiter = '[-–—─]'
const pelotanHeader = new RegExp(
  `^(?:\\s*${delimiter}{2,}\\s*pelotan(?:\\.cc/uv)?\\s+uv\\s*load(?:™)?\\s+analysis\\s*${delimiter}{2,}\\s*|\\s*pelotan\\s+uv\\s*load(?:™)?\\s+analysis\\s*,\\s*via\\s+pelotan\\.cc/uv\\s*)$`,
  'i',
)
const myWindsockHeader = new RegExp(
  `^\\s*${delimiter}{2,}\\s*my\\s*windsock\\s+(?:report|analysis)\\s*${delimiter}{2,}\\s*$`,
  'i',
)
const reportEnd = new RegExp(`^\\s*${delimiter}{2,}\\s*end\\s*${delimiter}{2,}\\s*$`, 'i')

const pelotanBlocks = (description: string): string[] => {
  const lines = description.split(/\r?\n/)
  const blocks: string[] = []
  for (let index = 0; index < lines.length; index += 1) {
    if (!pelotanHeader.test(lines[index])) continue
    const fields: string[] = []
    for (let fieldIndex = index + 1; fieldIndex < lines.length; fieldIndex += 1) {
      const line = lines[fieldIndex]
      if (
        line.trim().length === 0 ||
        pelotanHeader.test(line) ||
        myWindsockHeader.test(line) ||
        reportEnd.test(line)
      )
        break
      fields.push(line)
    }
    blocks.push(fields.join('\n'))
  }
  return blocks
}

const myWindsockBlocks = (description: string): string[] => {
  const lines = description.split(/\r?\n/)
  const blocks: string[] = []
  for (let index = 0; index < lines.length; index += 1) {
    if (!myWindsockHeader.test(lines[index])) continue
    const fields: string[] = []
    let closed = false
    for (let fieldIndex = index + 1; fieldIndex < lines.length; fieldIndex += 1) {
      const line = lines[fieldIndex]
      if (reportEnd.test(line)) {
        closed = true
        break
      }
      if (pelotanHeader.test(line) || myWindsockHeader.test(line)) break
      fields.push(line)
    }
    if (closed) blocks.push(fields.join('\n'))
  }
  return blocks
}

const severity = (value: string | null): UvSeverity | null => {
  const normalized = value?.trim().toLowerCase()
  if (normalized === 'negligible') return 'negligible'
  if (normalized === 'light' || normalized === 'low') return 'low'
  if (normalized === 'moderate') return 'moderate'
  if (normalized === 'high') return 'high'
  if (normalized === 'serious') return 'serious'
  if (normalized === 'extreme') return 'extreme'
  return null
}

const temperatureC = (value: number | null, unit: string | undefined): number | null => {
  if (value == null || unit == null) return null
  const converted = unit.toLowerCase() === 'f' ? ((value - 32) * 5) / 9 : value
  return converted >= -90 && converted <= 70 ? rounded(converted, 1) : null
}

const speedKph = (value: number | null, unit: string | undefined): number | null => {
  if (value == null || value < 0 || unit == null) return null
  const normalized = unit.toLowerCase().replaceAll(' ', '')
  if (normalized === 'mph') return rounded(value * 1.609344)
  if (normalized === 'm/s' || normalized === 'mps') return rounded(value * 3.6)
  if (normalized === 'km/h' || normalized === 'kmh' || normalized === 'kph') return rounded(value)
  return null
}

const elevationM = (value: number | null, unit: string | undefined): number | null => {
  if (value == null || unit == null) return null
  const normalized = unit.toLowerCase()
  if (normalized === 'ft' || normalized === 'feet') return rounded(value * 0.3048)
  if (normalized === 'm' || normalized === 'metres' || normalized === 'meters')
    return rounded(value)
  return null
}

const precipitationMmPerHour = (value: number | null, unit: string | undefined): number | null => {
  if (value == null || value < 0 || unit == null) return null
  const normalized = unit.toLowerCase().replaceAll(' ', '')
  if (normalized === 'in/hr' || normalized === 'inch/hr' || normalized === 'inches/hr')
    return rounded(value * 25.4)
  if (normalized === 'mm/hr' || normalized === 'mm/h') return rounded(value)
  return null
}

const clockSeconds = (value: string | undefined): number | null => {
  if (value == null) return null
  const trimmed = value.trim()
  const colon = /^(\d{1,3}):(\d{2})(?::(\d{2}))?$/.exec(trimmed)
  if (colon) {
    const hours = colon[3] == null ? 0 : Number(colon[1])
    const minutes = Number(colon[3] == null ? colon[1] : colon[2])
    const seconds = Number(colon[3] ?? colon[2])
    if (minutes >= 60 || seconds >= 60) return null
    return hours * 3600 + minutes * 60 + seconds
  }
  const units = /^(?:(\d{1,3})\s*h)?\s*(?:(\d{1,2})\s*m)?\s*(?:(\d{1,2})\s*s)?$/i.exec(trimmed)
  if (!units || !units.slice(1).some(Boolean)) return null
  const hours = Number(units[1] ?? 0)
  const minutes = Number(units[2] ?? 0)
  const seconds = Number(units[3] ?? 0)
  if (minutes >= 60 || seconds >= 60) return null
  return hours * 3600 + minutes * 60 + seconds
}

const provenance = (activityId: number, retrievedAt: number): ProviderNativeProvenance => ({
  source: 'provider-native',
  transport: 'strava-description',
  schemaVersion: 1,
  activityId,
  retrievedAt: Number.isFinite(retrievedAt) && retrievedAt >= 0 ? retrievedAt : 0,
})

const parsePelotanBlock = (
  block: string,
  activityId: number,
  retrievedAt: number,
): PelotanNativeReport | null => {
  const scoreMatch = /uv\s*load(?:™)?\s*:?[ \t]*(\d{1,3})(?:[ \t]*[—–-][ \t]*([^\r\n]+))?/i.exec(
    block,
  )
  const score = bounded(number(scoreMatch?.[1]), 0, 100)
  const rawBand = scoreMatch?.[2]?.trim() || null
  const averageUvIndex = bounded(
    number(
      /avg(?:\.|erage)?[ \t]+uv(?:[ \t]+index)?[ \t]*:?[ \t]*([+-]?\d+(?:[.,]\d+)?)/i.exec(
        block,
      )?.[1],
    ),
    0,
    30,
  )
  const temperatureMatch =
    /(?:🌡️?|temperature)[ \t]*:?[ \t]*([+-]?\d+(?:[.,]\d+)?)[ \t]*°?[ \t]*([CF])/i.exec(block)
  const cloudMatch = /(?:☁️?|cloud(?:[ \t]+cover)?)[ \t]*:?[ \t]*(\d+(?:[.,]\d+)?)[ \t]*%/i.exec(
    block,
  )
  const averageTemperatureC = temperatureC(number(temperatureMatch?.[1]), temperatureMatch?.[2])
  const averageCloudCoverPct = bounded(number(cloudMatch?.[1]), 0, 100)
  if (
    score == null &&
    averageUvIndex == null &&
    averageTemperatureC == null &&
    averageCloudCoverPct == null
  )
    return null
  return {
    ...provenance(activityId, retrievedAt),
    provider: 'pelotan',
    score,
    rawBand,
    severity: severity(rawBand),
    averageUvIndex,
    averageTemperatureC,
    averageCloudCoverPct,
  }
}

export function parsePelotanReport(
  description: string | null,
  activityId: number,
  retrievedAt: number,
): PelotanNativeReport | null {
  if (!description) return null
  for (const block of pelotanBlocks(description).toReversed()) {
    const report = parsePelotanBlock(block, activityId, retrievedAt)
    if (report) return report
  }
  return null
}

const parseMyWindsockBlock = (
  block: string,
  activityId: number,
  retrievedAt: number,
): MyWindsockNativeReport | null => {
  const weatherImpactPct = bounded(
    number(
      /weather[ \t]+impact(?:™)?[ \t]*:?[ \t]*([+-]?\d+(?:[.,]\d+)?)[ \t]*%/i.exec(block)?.[1],
    ),
    -1000,
    1000,
  )
  const cdaMatch =
    /(?:^|\n)[^\r\n]*?\bcda\b(?:[ \t]*\([^)]*\))?[ \t]*:?[ \t]*(\d+(?:[.,]\d+)?)/i.exec(block) ??
    /(?:^|\n)[ \t]*aerodynamics[ \t]*:?[ \t]*(\d+(?:[.,]\d+)?)[ \t]+cda\b/i.exec(block)
  const cdaM2 = bounded(number(cdaMatch?.[1]), 0, 5)
  const feelsLikeMatch =
    /feels[ \t-]*like[ \t]+elev(?:ation)?(?:™)?[ \t]*:?[ \t]*([+-]?\d+(?:[.,]\d+)?)[ \t]*(ft|feet|m|metres|meters)\b/i.exec(
      block,
    )
  const headwindPct = bounded(
    number(
      /headwind(?:[ \t]+(?:time|percentage|share))?[ \t]*:?[ \t]*(\d+(?:[.,]\d+)?)[ \t]*%/i.exec(
        block,
      )?.[1],
    ),
    0,
    100,
  )
  const headwindSpeedMatch =
    /headwind[^\r\n]*?%[^\r\n]*?(\d+(?:[.,]\d+)?)[ \t]*[–—-][ \t]*(\d+(?:[.,]\d+)?)[ \t]*(mph|km\/?h|kmh|kph|m\/?s|mps)\b/i.exec(
      block,
    )
  const longestHeadwindS = clockSeconds(
    /longest[ \t]+headwind[ \t]*:?[ \t]*([^\r\n]+)/i.exec(block)?.[1],
  )
  const airSpeedMatch =
    /(?:^|\n)[^\r\n]*?\bair[ \t]+speed\b[ \t]*:?[ \t]*(\d+(?:[.,]\d+)?)[ \t]*(mph|km\/?h|kmh|kph|m\/?s|mps)\b/i.exec(
      block,
    )
  const temperatureMatch =
    /(?:^|\n)[^\r\n]*?\btemp(?:erature)?\b[ \t]*:?[ \t]*([+-]?\d+(?:[.,]\d+)?)(?:[ \t]*[–—-][ \t]*([+-]?\d+(?:[.,]\d+)?))?[ \t]*°?[ \t]*([CF])/i.exec(
      block,
    )
  const precipitationMatch =
    /(?:precipitation|precip)[ \t]*:?[ \t]*(\d+(?:[.,]\d+)?)[ \t]*%(?:[^\r\n]*?([0-9]+(?:[.,]\d+)?)[ \t]*(in\/?hr|inch(?:es)?\/?hr|mm\/?h(?:r)?))?/i.exec(
      block,
    )
  const temperatureStartC = temperatureC(number(temperatureMatch?.[1]), temperatureMatch?.[3])
  const temperatureEndC = temperatureC(number(temperatureMatch?.[2]), temperatureMatch?.[3])
  const averageTemperatureC =
    temperatureStartC == null
      ? null
      : temperatureEndC == null
        ? temperatureStartC
        : rounded((temperatureStartC + temperatureEndC) / 2, 1)
  const report: MyWindsockNativeReport = {
    ...provenance(activityId, retrievedAt),
    provider: 'mywindsock',
    weatherImpactPct,
    cdaM2,
    feelsLikeElevationM: elevationM(number(feelsLikeMatch?.[1]), feelsLikeMatch?.[2]),
    headwindPct,
    headwindMinKph: speedKph(number(headwindSpeedMatch?.[1]), headwindSpeedMatch?.[3]),
    headwindMaxKph: speedKph(number(headwindSpeedMatch?.[2]), headwindSpeedMatch?.[3]),
    longestHeadwindS,
    airSpeedKph: speedKph(number(airSpeedMatch?.[1]), airSpeedMatch?.[2]),
    averageTemperatureC,
    precipitationProbabilityPct: bounded(number(precipitationMatch?.[1]), 0, 100),
    precipitationRateMmPerHour: precipitationMmPerHour(
      number(precipitationMatch?.[2]),
      precipitationMatch?.[3],
    ),
  }
  if (
    Object.entries(report).every(
      ([key, value]) =>
        key === 'source' ||
        key === 'transport' ||
        key === 'schemaVersion' ||
        key === 'activityId' ||
        key === 'retrievedAt' ||
        key === 'provider' ||
        value == null,
    )
  )
    return null
  return report
}

export function parseMyWindsockReport(
  description: string | null,
  activityId: number,
  retrievedAt: number,
): MyWindsockNativeReport | null {
  if (!description) return null
  for (const block of myWindsockBlocks(description).toReversed()) {
    const report = parseMyWindsockBlock(block, activityId, retrievedAt)
    if (report) return report
  }
  return null
}

export function parseActivityProviderReports(
  description: string | null,
  activityId: number,
  retrievedAt: number,
): NativeActivityReports {
  return {
    myWindsock: parseMyWindsockReport(description, activityId, retrievedAt),
    pelotan: parsePelotanReport(description, activityId, retrievedAt),
  }
}
