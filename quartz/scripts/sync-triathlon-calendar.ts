import type { Element, Root } from 'hast'
import { Defuddle } from 'defuddle/node'
import { fromHtml } from 'hast-util-from-html'
import { toString } from 'hast-util-to-string'
import { toText } from 'hast-util-to-text'
import { execFileSync } from 'node:child_process'
import { readFile, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { pathToFileURL } from 'node:url'
import { parseArgs } from 'node:util'
import { visit } from 'unist-util-visit'
import type {
  TriathlonCalendarCourse,
  TriathlonCalendarDetails,
  TriathlonCalendarLeg,
  TriathlonCalendarSchedule,
  TriathlonCalendarScheduleDay,
  TriathlonCalendarScheduleItem,
} from '../util/triathlon-calendar'
import {
  calendarSeries,
  parseCalendarDetails,
  TRIATHLON_CALENDAR_LEGS,
} from '../util/triathlon-calendar'
import { TRIATHLON_CALENDAR_CATALOG } from '../util/triathlon-calendar-data'
import {
  mapDistanceMeters,
  simplifyMapRoute,
  type MapRoutePoint,
} from '../util/triathlon-map-route'
import { isRecord, type UnknownRecord } from '../util/type-guards'

const OUTPUT = path.join(process.cwd(), 'quartz', 'util', 'triathlon-calendar-details.json')
const USER_AGENT = 'aarnphm-garden-sync/1.0 (+https://aarnphm.xyz)'
const MAX_RESPONSE_CHARS = 6_000_000
const ROUTE_POINTS = 160
const PROFILE_SAMPLES = 64
const MONTHS = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

async function fetchText(url: string): Promise<string> {
  const response = await fetch(url, {
    headers: { 'User-Agent': USER_AGENT, Accept: 'text/html,application/json;q=0.9,*/*;q=0.5' },
    signal: AbortSignal.timeout(30_000),
  })
  if (!response.ok) throw new Error(`${url} answered HTTP ${response.status}`)
  const body = await response.text()
  if (body.length > MAX_RESPONSE_CHARS) throw new Error(`${url} exceeds the response size limit`)
  return body
}

async function readable(html: string, url: string): Promise<string> {
  const result = await Defuddle(html, url, { markdown: true, removeSmallImages: false })
  return result.content
}

/** Markdown inline syntax and escapes removed, whitespace collapsed. */
const plain = (value: string): string =>
  value
    .replace(/\\([\\`*_{}[\]()#+\-.!|])/g, '$1')
    .replace(/\*+/g, '')
    .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/\s+/g, ' ')
    .trim()

const decodeEntities = (value: string): string =>
  value
    .replace(/&#(\d+);/g, (_, code: string) => String.fromCodePoint(Number(code)))
    .replace(/&quot;/g, '"')
    .replace(/&#x27;|&apos;/g, "'")
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&amp;/g, '&')

const LEG_NAMES: Record<string, TriathlonCalendarLeg> = { swim: 'swim', bike: 'bike', run: 'run' }

/** IRONMAN prints "Swim\n\nRiver —"; T100 prints "### Bike\n\nRolling". */
function parseTerrain(markdown: string): Partial<Record<TriathlonCalendarLeg, string>> {
  const terrain: Partial<Record<TriathlonCalendarLeg, string>> = {}
  const pattern =
    /^(?:#+\s*)?(Swim|Bike|Run)(?:[ \t]*\n+[ \t]*|[ \t]+)([A-Z][A-Za-z ]{1,24}?)(?:\s+—)?[ \t]*$/gm
  for (const [, leg, label] of markdown.matchAll(pattern))
    terrain[LEG_NAMES[leg.toLowerCase()]] ??= label.trim().toLowerCase()
  return terrain
}

/** Celsius value or range printed under a label, e.g. "78 °F / 26 °C" or "15-23°C". */
function celsiusUnder(markdown: string, label: string): [number, number] | null {
  const number = String.raw`(-?\d+(?:\.\d+)?)`
  const match = new RegExp(
    String.raw`${label}[^\n]*\n+\s*(?:-?\d+(?:\.\d+)?\s*°\s*F\s*/\s*)?${number}(?:\s*(?:°\s*C)?\s*[-–]\s*${number})?\s*°\s*C`,
    'i',
  ).exec(markdown)
  if (!match) return null
  const low = Number(match[1])
  return [low, match[2] === undefined ? low : Number(match[2])]
}

function parseConditions(markdown: string): Pick<TriathlonCalendarDetails, 'airC' | 'waterC'> {
  const high = celsiusUnder(markdown, 'High Air Temp')
  const low = celsiusUnder(markdown, 'Low Air Temp')
  return {
    airC: high && low ? [low[0], high[0]] : celsiusUnder(markdown, 'Air Temperature'),
    waterC: celsiusUnder(markdown, 'Water Temp(?:erature)?'),
  }
}

const TIME = String.raw`\d{1,2}(?::\d{2})?\s*(?:[ap]\.?\s?m\.?)?`
const RANGE = new RegExp(
  String.raw`^(${TIME})(?:\s*(?:-|–|to)\s*(${TIME}))?(?:\s*[-–:]\s*|\s+)(.+)$`,
  'i',
)

function clock(raw: string, fallbackMeridiem: 'a' | 'p' | null): number | null {
  const match = /^(\d{1,2})(?::(\d{2}))?\s*([ap])?/i.exec(raw.trim())
  if (!match) return null
  let hour = Number(match[1])
  const minute = Number(match[2] ?? 0)
  const meridiem = (match[3]?.toLowerCase() as 'a' | 'p' | undefined) ?? fallbackMeridiem
  if (hour > 23 || minute > 59 || (meridiem && hour > 12)) return null
  if (meridiem === 'p' && hour < 12) hour += 12
  if (meridiem === 'a' && hour === 12) hour = 0
  return hour * 60 + minute
}

const clockText = (minutes: number): string =>
  `${String(Math.floor(minutes / 60) % 24).padStart(2, '0')}:${String(minutes % 60).padStart(2, '0')}`

function timeRange(startRaw: string, endRaw: string | undefined): [string, string | null] | null {
  const endMeridiem = endRaw
    ? (/([ap])\.?\s?m/i.exec(endRaw)?.[1]?.toLowerCase() as 'a' | 'p' | undefined)
    : undefined
  const end = endRaw ? clock(endRaw, null) : null
  let start = clock(startRaw, endMeridiem ?? null)
  // "11:00 to 1:00pm" borrows the wrong meridiem; the start cannot follow the end.
  if (start !== null && end !== null && endMeridiem && !/[ap]/i.test(startRaw) && start > end)
    start = clock(startRaw, endMeridiem === 'p' ? 'a' : 'p')
  if (start === null) return null
  return [clockText(start), end === null ? null : clockText(end)]
}

/** First clause of an activity description; a short trailing parenthetical becomes the place. */
function tidyActivity(
  raw: string,
  placeFromParenthetical: boolean,
): { activity: string; location: string | null } {
  const text = plain(raw)
  let location = /\bLocation:\s*([^.;]+?)(?:\s+-\s+|[.;]|$)/i.exec(text)?.[1]?.trim() ?? null
  let activity = text.split(/\s*;\s*|\s+[-–]\s+|\.\s+/)[0].replace(/,?\s*\(\d+ of \d+\)/gi, '')
  const place = placeFromParenthetical ? /\(([^()]{2,40})\)[\s,]*$/.exec(activity) : null
  if (place && place[1].trim().split(/\s+/).length <= 4) {
    location ??= place[1].trim()
    activity = activity.slice(0, place.index)
  }
  activity = activity
    .replace(/\([^()]*$/, '')
    .replace(/\s*\([^()]{25,}\)/g, '')
    .replace(/[\s,:(-]+$/, '')
    .trim()
  return { activity, location }
}

function dayDate(heading: string, fallbackYear: number): string | null {
  const text = plain(heading)
  const month = MONTHS.findIndex(name => new RegExp(String.raw`\b${name}[a-z]*\b`, 'i').test(text))
  const year = Number(/\b(20\d{2})\b/.exec(text)?.[1] ?? fallbackYear)
  const day = Number(
    /\b(\d{1,2})(?:st|nd|rd|th)?\b(?!\d)/.exec(text.replace(/\b20\d{2}\b/, ''))?.[1],
  )
  if (month < 0 || !(day >= 1 && day <= 31)) return null
  const date = new Date(Date.UTC(year, month, day))
  return date.getUTCMonth() === month ? date.toISOString().slice(0, 10) : null
}

const raceStart = (item: TriathlonCalendarScheduleItem): boolean =>
  /\bstart\b/i.test(item.activity) &&
  !/\b(pro|ironkids|kids|transition|shuttle|warm)\b/i.test(item.activity)

interface ParsedDay {
  date: string
  raceHeading: boolean
  items: TriathlonCalendarScheduleItem[]
}

/**
 * Reads day headings followed by either markdown table rows (IRONMAN: time | activity | location)
 * or list items (Supertri: "- 6:45am - Transition Opens (Bandshell Park)").
 */
function parseScheduleMarkdown(markdown: string, fallbackYear: number): ParsedDay[] {
  const days: ParsedDay[] = []
  let current: ParsedDay | null = null
  for (const line of markdown.split('\n')) {
    const heading = /^#{2,4}\s+(.+)$/.exec(line)
    if (heading) {
      const date = dayDate(heading[1], fallbackYear)
      current = date ? { date, raceHeading: /race day/i.test(heading[1]), items: [] } : null
      if (current) days.push(current)
      continue
    }
    if (!current) continue
    let cells: {
      time: string
      activity: string
      location: string | null
      parenthetical: boolean
    } | null = null
    if (line.startsWith('|')) {
      const row = line
        .split('|')
        .slice(1, -1)
        .map(cell => cell.trim())
      if (row.length < 2 || /^-+$/.test(row[0]) || /^\**time\**$/i.test(row[0])) continue
      // Notes after an emphasis marker ("*(AWA Priority Ends…)*") are not the activity.
      cells = {
        time: plain(row[0]),
        activity: row[1].split('*').find(Boolean) ?? '',
        location: plain(row[2] ?? '') || null,
        parenthetical: false,
      }
    } else if (/^\s*[-*]\s+/.test(line)) {
      const item = plain(line.replace(/^\s*[-*]\s+/, ''))
      const match = RANGE.exec(item)
      if (!match) continue
      cells = {
        time: match[2] ? `${match[1]} - ${match[2]}` : match[1],
        activity: match[3],
        location: null,
        parenthetical: true,
      }
    }
    if (!cells) continue
    const time = RANGE.exec(`${cells.time} x`)
    const range = time ? timeRange(time[1], time[2]) : null
    if (!range) continue
    const { activity, location } = tidyActivity(cells.activity, cells.parenthetical)
    if (!activity) continue
    current.items.push({
      start: range[0],
      end: range[1],
      activity,
      location: cells.location ?? location,
      race: false,
    })
  }
  return days.filter(day => day.items.length > 0)
}

function buildSchedule(days: ParsedDay[], source: string): TriathlonCalendarSchedule | null {
  const race =
    days.find(day => day.raceHeading) ??
    days.find(day => day.items.some(item => /race start/i.test(item.activity)))
  if (!race) return null
  const start = race.items.find(raceStart)
  if (start) start.race = true
  const raceTime = Date.parse(`${race.date}T00:00:00Z`)
  const scheduleDays: TriathlonCalendarScheduleDay[] = days.map(day => ({
    offset: Math.round((Date.parse(`${day.date}T00:00:00Z`) - raceTime) / 86_400_000),
    items: day.items,
  }))
  return { source, raceDate: race.date, days: scheduleDays }
}

function measureMeters(value: string | undefined): number | null {
  const match = /([\d,.]+)\s*(km|m|mi|ft)\b/i.exec(value ?? '')
  if (!match) return null
  const amount = Number(match[1].replaceAll(',', ''))
  const scale = { km: 1000, m: 1, mi: 1609.344, ft: 0.3048 }[
    match[2].toLowerCase() as 'km' | 'm' | 'mi' | 'ft'
  ]
  return Number.isFinite(amount) ? Math.round(amount * scale) : null
}

function encodeSigned(value: number): string {
  let rest = value < 0 ? ~(value << 1) : value << 1
  let out = ''
  while (rest >= 0x20) {
    out += String.fromCharCode((0x20 | (rest & 0x1f)) + 63)
    rest >>= 5
  }
  return out + String.fromCharCode(rest + 63)
}

function encodePolyline(points: readonly MapRoutePoint[]): string {
  let lat = 0
  let lng = 0
  let out = ''
  for (const point of points) {
    const nextLat = Math.round(point.lat * 1e5)
    const nextLng = Math.round(point.lng * 1e5)
    out += encodeSigned(nextLat - lat) + encodeSigned(nextLng - lng)
    lat = nextLat
    lng = nextLng
  }
  return out
}

function simplifyToLimit(points: readonly MapRoutePoint[], limit: number): MapRoutePoint[] {
  let tolerance = 2
  let simplified = simplifyMapRoute(points, tolerance)
  while (simplified.length > limit && tolerance < 2_000) {
    tolerance *= 1.4
    simplified = simplifyMapRoute(points, tolerance)
  }
  return simplified
}

function sampleProfile(points: readonly MapRoutePoint[], elevations: readonly number[]): number[] {
  const total = points[points.length - 1]?.d ?? 0
  if (total <= 0) return []
  const profile: number[] = []
  let index = 1
  for (let sample = 0; sample < PROFILE_SAMPLES; sample++) {
    const target = (total * sample) / (PROFILE_SAMPLES - 1)
    while (index < points.length - 1 && points[index].d < target) index++
    const a = points[index - 1]
    const b = points[index]
    const span = b.d - a.d
    const fraction = span > 0 ? Math.min(1, Math.max(0, (target - a.d) / span)) : 0
    profile.push(
      Math.round(elevations[index - 1] + (elevations[index] - elevations[index - 1]) * fraction),
    )
  }
  return profile
}

/** Strava's route embed ships the full route as JSON: [lng, lat, elevation] plus waypoints. */
async function stravaCourse(route: StravaRouteRef): Promise<TriathlonCalendarCourse | null> {
  const routeId = route.id
  const html = await fetchText(`https://strava-embeds.com/route/${routeId}`)
  const data = /<script id="__ROUTE_DATA__" type="application\/json">([\s\S]*?)<\/script>/.exec(
    html,
  )
  if (!data) return null
  const routeData: unknown = JSON.parse(data[1])
  if (!isRecord(routeData) || !Array.isArray(routeData.coordinates)) return null
  const title = decodeEntities(/<h1 class="title">([^<]+)/.exec(html)?.[1] ?? '').trim()
  // Organisers draw swims as Run routes, so the page's own label and the title outrank the type.
  const type = /<title>([^<]+)<\/title>/.exec(html)?.[1]
  const leg =
    route.leg ??
    (/\bswim/i.test(title)
      ? 'swim'
      : /\b(bike|ride|cycl)/i.test(title)
        ? 'bike'
        : /\brun\b/i.test(title)
          ? 'run'
          : type === 'Swim'
            ? 'swim'
            : type === 'Ride'
              ? 'bike'
              : type === 'Run'
                ? 'run'
                : null)
  const stats = new Map(
    Array.from(
      html.matchAll(/stat-label">([^<]*)<\/div>\s*<div class="stat-value">([^<]*)</g),
      match => [match[1], match[2]],
    ),
  )
  const points: MapRoutePoint[] = []
  const elevations: number[] = []
  for (const coordinate of routeData.coordinates) {
    if (!Array.isArray(coordinate)) continue
    const [lng, lat, elevation] = coordinate as unknown[]
    if (typeof lat !== 'number' || typeof lng !== 'number') continue
    const previous = points[points.length - 1]
    const point = { lat, lng, d: 0 }
    point.d = previous ? previous.d + mapDistanceMeters(previous, point) / 1000 : 0
    points.push(point)
    elevations.push(
      typeof elevation === 'number' ? elevation : (elevations[elevations.length - 1] ?? 0),
    )
  }
  const distanceM =
    measureMeters(stats.get('Distance')) ?? Math.round((points[points.length - 1]?.d ?? 0) * 1000)
  if (!leg || !title || points.length < 2 || distanceM <= 0) return null
  const aidM = Array.isArray(routeData.waypoints)
    ? routeData.waypoints
        .filter(isRecord)
        // Turning points and course alerts share the waypoint list with aid stations.
        .filter(
          waypoint =>
            waypoint.category === 'WaterSource' ||
            waypoint.category === 'Restaurant' ||
            /\b(aid|water|feed)\b/i.test(String(waypoint.title)),
        )
        .map(waypoint =>
          typeof waypoint.distance === 'number' ? Math.round(waypoint.distance) : null,
        )
        .filter(
          (distance): distance is number =>
            distance !== null && distance > 0 && distance < distanceM,
        )
        .sort((left, right) => left - right)
    : []
  return {
    leg,
    title,
    source: `https://www.strava.com/routes/${routeId}`,
    edition: null,
    mapUrl: null,
    gpxUrl: null,
    distanceM,
    raceDistanceM: null,
    laps: null,
    aidStationsPerLap: null,
    elevationGainM: leg === 'swim' ? null : measureMeters(stats.get('Elev Gain')),
    polyline: encodePolyline(simplifyToLimit(points, ROUTE_POINTS)),
    profile: leg === 'swim' ? [] : sampleProfile(points, elevations),
    aidM,
  }
}

interface StravaRouteRef {
  id: string
  leg?: TriathlonCalendarLeg
}

const courseOrder: Record<TriathlonCalendarLeg, number> = { swim: 0, bike: 1, run: 2 }

async function stravaCourses(
  routes: readonly StravaRouteRef[],
): Promise<TriathlonCalendarCourse[]> {
  const courses: TriathlonCalendarCourse[] = []
  const seen = new Set<string>()
  for (const route of routes) {
    if (seen.has(route.id)) continue
    seen.add(route.id)
    const course = await stravaCourse(route)
    if (course) courses.push(course)
    else console.warn(`  strava route ${route.id}: no route data`)
  }
  return courses.sort((left, right) => courseOrder[left.leg] - courseOrder[right.leg])
}

const emptyDetails = (sources: string[]): TriathlonCalendarDetails => ({
  sources,
  fetched: null,
  venue: null,
  venueMap: null,
  athleteGuide: null,
  terrain: {},
  airC: null,
  waterC: null,
  course: [],
  schedule: null,
  schedulePending: false,
  kept: {},
})

async function ironmanDetails(url: string, year: number): Promise<TriathlonCalendarDetails> {
  const base = /^https:\/\/www\.ironman\.com\/races\/[^/]+/.exec(url)?.[0]
  if (!base) throw new Error(`${url} is not an IRONMAN race page`)
  const home = await fetchText(base)
  const courseHtml = await fetchText(`${base}/course`)
  const scheduleHtml = await fetchText(`${base}/schedule`)
  const homeText = await readable(home, base)
  const scheduleText = await readable(scheduleHtml, `${base}/schedule`)
  const routes = Array.from(
    courseHtml.matchAll(/data-embed-type="route" data-embed-id="(\d+)"/g),
    match => ({ id: match[1] }),
  )
  const details = emptyDetails([base, `${base}/course`, `${base}/schedule`])
  details.terrain = parseTerrain(homeText)
  Object.assign(details, parseConditions(homeText))
  details.course = await stravaCourses(routes)
  details.schedule = buildSchedule(
    parseScheduleMarkdown(scheduleText, year - 1),
    `${base}/schedule`,
  )
  details.sources.push(...details.course.map(course => course.source))
  return details
}

/** Supertri is a Gatsby site: the Strava route IDs live in page-data, the schedule in the HTML. */
async function supertriDetails(url: string, year: number): Promise<TriathlonCalendarDetails> {
  const origin = new URL(url).origin
  const city = new URL(url).pathname.split('/').filter(Boolean)[0]
  const pageData = await fetchText(`${origin}/page-data/${city}/olympic/page-data.json`)
  const routes = Array.from(
    pageData.matchAll(/"tabLabel":\s*"([^"]+)",\s*"stravaRouteId":\s*"(\d+)"/g),
    match => ({ id: match[2], leg: LEG_NAMES[match[1].trim().toLowerCase()] }),
  )
  const scheduleUrl = `${origin}/${city}/schedule`
  // Each day heading sits in a sibling column that Defuddle drops as page chrome; move it into the
  // day's rich text so the markdown keeps one heading per day.
  const scheduleHtml = (await fetchText(scheduleUrl)).replace(
    /<h2 class="title">([^<]*)<\/h2>([\s\S]*?)<div class="rich-text-wrap">/g,
    '$2<div class="rich-text-wrap"><h2>$1</h2>',
  )
  const scheduleText = await readable(scheduleHtml, scheduleUrl)
  const edition = Number(/(20\d{2}) Weekend Schedule/.exec(scheduleHtml)?.[1] ?? year - 1)
  const details = emptyDetails([url, `${origin}/${city}/olympic`, scheduleUrl])
  details.course = await stravaCourses(routes)
  details.schedule = buildSchedule(parseScheduleMarkdown(scheduleText, edition), scheduleUrl)
  details.sources.push(...details.course.map(course => course.source))
  return details
}

const elements = (tree: Root | Element, matches: (node: Element) => boolean): Element[] => {
  const found: Element[] = []
  visit(tree, 'element', node => {
    if (matches(node)) found.push(node)
  })
  return found
}

const elementText = (node: Element): string => plain(toText(node))

const imageLink = (tree: Element | undefined): string | null => {
  if (!tree) return null
  const anchor = elements(
    tree,
    node => node.tagName === 'a' && /\.(png|jpe?g|webp)$/i.test(String(node.properties.href)),
  )[0]
  return typeof anchor?.properties.href === 'string' ? anchor.properties.href : null
}

interface RideWithGpsRef {
  id: string
  leg: TriathlonCalendarLeg
  edition: number | null
  raceDistanceM: number | null
  laps: number | null
  aidStationsPerLap: number | null
  mapUrl: string | null
}

/** Public route JSON uses metres for distance and elevation, x for longitude, and y for latitude. */
async function rideWithGpsCourse(route: RideWithGpsRef): Promise<TriathlonCalendarCourse | null> {
  const source = `https://ridewithgps.com/routes/${route.id}`
  const data: unknown = JSON.parse(await fetchText(`${source}.json`))
  if (!isRecord(data) || !Array.isArray(data.track_points)) return null
  const points: MapRoutePoint[] = []
  const elevations: number[] = []
  for (const point of data.track_points) {
    if (
      !isRecord(point) ||
      typeof point.x !== 'number' ||
      !Number.isFinite(point.x) ||
      Math.abs(point.x) > 180 ||
      typeof point.y !== 'number' ||
      !Number.isFinite(point.y) ||
      Math.abs(point.y) > 90 ||
      typeof point.d !== 'number' ||
      !Number.isFinite(point.d) ||
      point.d < 0
    )
      continue
    if (point.d / 1000 < (points.at(-1)?.d ?? 0)) return null
    points.push({ lng: point.x, lat: point.y, d: point.d / 1000 })
    if (typeof point.e === 'number' && Number.isFinite(point.e)) elevations.push(point.e)
  }
  if (points.length < 2 || typeof data.name !== 'string' || !data.name.trim()) return null
  const distanceM = typeof data.distance === 'number' ? Math.round(data.distance) : 0
  if (!Number.isFinite(distanceM) || distanceM <= 0) return null
  const elevationGainM =
    route.leg !== 'swim' &&
    typeof data.elevation_gain === 'number' &&
    Number.isFinite(data.elevation_gain) &&
    data.elevation_gain >= 0
      ? Math.round(data.elevation_gain)
      : null
  return {
    leg: route.leg,
    title: data.name.trim(),
    source,
    edition: route.edition,
    mapUrl: route.mapUrl,
    gpxUrl: null,
    distanceM,
    raceDistanceM: route.raceDistanceM,
    laps: route.laps,
    aidStationsPerLap: route.aidStationsPerLap,
    elevationGainM,
    polyline: encodePolyline(simplifyToLimit(points, ROUTE_POINTS)),
    profile:
      route.leg !== 'swim' && elevations.length === points.length
        ? sampleProfile(points, elevations)
        : [],
    aidM: [],
  }
}

/** Elementor keeps the course accordions and schedule tables in tabs that reader extraction drops. */
async function t100Details(url: string): Promise<TriathlonCalendarDetails> {
  const infoUrl = new URL('event-info/', url).href
  const tree = fromHtml(await fetchText(infoUrl))
  const all = elements(tree, () => true)
  const byId = new Map(all.map(node => [node.properties.id, node]))
  const tabs = all
    .filter(node => node.properties.role === 'tab')
    .flatMap(tab => {
      const controls = tab.properties.ariaControls
      const id = Array.isArray(controls) ? controls[0] : controls
      const panel = byId.get(id)
      return panel ? [{ tab, panel }] : []
    })
  const text = toText(tree)
  const editionMatch = /\b(20\d{2})\s+(?:Schedule|Athlete guide)\b/i.exec(text)
  const edition = editionMatch ? Number(editionMatch[1]) : null
  const details = emptyDetails([url, infoUrl])
  details.terrain = parseTerrain(text)
  Object.assign(details, parseConditions(text))
  details.venue = /Start Location\s*\n+([^\n]+)/i.exec(text)?.[1]?.trim() ?? null
  details.venueMap = imageLink(tabs.find(({ tab }) => /venue\s*map/i.test(toString(tab)))?.panel)
  const guide = all.find(
    node =>
      node.tagName === 'a' && /athlete[-_ ]?guide.*\.pdf$/i.test(String(node.properties.href)),
  )
  details.athleteGuide = typeof guide?.properties.href === 'string' ? guide.properties.href : null
  const coursePanel = tabs.find(({ tab }) => /100\s*km\s*triathlon/i.test(toString(tab)))?.panel
  const routes: RideWithGpsRef[] = []
  for (const section of coursePanel
    ? elements(coursePanel, node => node.tagName === 'details')
    : []) {
    const heading = section.children.find(
      node => node.type === 'element' && node.tagName === 'summary',
    )
    const label = heading ? plain(toString(heading)) : ''
    const match = /\b(swim|bike|run)\s*\(([\d.]+\s*km)\)/i.exec(label)
    if (!match) continue
    const iframe = elements(section, node => node.tagName === 'iframe')[0]
    const src = iframe?.properties.src
    if (typeof src !== 'string' || !URL.canParse(src)) continue
    const embed = new URL(src)
    const id = embed.searchParams.get('id')
    if (
      embed.hostname !== 'ridewithgps.com' ||
      embed.searchParams.get('type') !== 'route' ||
      !id ||
      !/^\d+$/.test(id)
    )
      continue
    const description = elementText(section)
    routes.push({
      id,
      leg: LEG_NAMES[match[1].toLowerCase()],
      edition,
      raceDistanceM: measureMeters(match[2]),
      laps: Number(/\b(\d+)\s+laps?\b/i.exec(description)?.[1]) || null,
      aidStationsPerLap: /\b(?:one|1)\s+aid station per lap\b/i.test(description) ? 1 : null,
      mapUrl: imageLink(section),
    })
  }
  for (const route of routes) {
    try {
      const course = await rideWithGpsCourse(route)
      if (course) details.course.push(course)
      else console.warn(`  Ride with GPS route ${route.id}: no route data`)
    } catch (error) {
      console.warn(
        `  Ride with GPS route ${route.id}: ${error instanceof Error ? error.message : String(error)}`,
      )
    }
  }
  const days: ParsedDay[] = []
  for (const { tab, panel } of tabs) {
    if (!edition) continue
    const dateLabel = elements(
      tab,
      node =>
        Array.isArray(node.properties.className) &&
        node.properties.className.includes('tab-location'),
    )[0]
    const date = dayDate(elementText(dateLabel ?? tab), edition)
    if (!date) continue
    const items: TriathlonCalendarScheduleItem[] = []
    for (const row of elements(panel, node => node.tagName === 'tr')) {
      const cells = row.children.filter(
        (node): node is Element => node.type === 'element' && node.tagName === 'td',
      )
      if (cells.length < 2) continue
      const parts = cells[1].children
        .filter((node): node is Element => node.type === 'element')
        .map(elementText)
        .filter(Boolean)
      const activity = parts[0] ?? elementText(cells[1])
      const description = elementText(cells[1])
      if (
        !/100\s*km/i.test(description) &&
        /\b(olympic|sprint|youth|junior|pro)\b/i.test(description)
      )
        continue
      // This source writes some 24-hour values with a redundant PM suffix.
      const time = elementText(cells[0]).replace(/\b(1[3-9]|2[0-3])(:\d{2})\s*PM\b/gi, '$1$2')
      const match = RANGE.exec(`${time} x`)
      const range = match ? timeRange(match[1], match[2]) : null
      if (!range || !activity) continue
      items.push({
        start: range[0],
        end: range[1],
        activity,
        location: parts.slice(1).join(' ') || null,
        race: false,
      })
    }
    if (items.length > 0)
      days.push({
        date,
        items,
        raceHeading: items.some(item => /100\s*km.*race start/i.test(item.activity)),
      })
  }
  details.schedule = buildSchedule(days, infoUrl)
  details.sources.push(...details.course.map(course => course.source))
  return details
}

async function hyroxDetails(url: string): Promise<TriathlonCalendarDetails> {
  const html = await fetchText(url)
  const text = decodeEntities(
    html.replace(/<(script|style)[\s\S]*?<\/\1>/g, '').replace(/<[^>]+>/g, '\n'),
  )
  const lines = text
    .split('\n')
    .map(line => line.trim())
    .filter(Boolean)
  const at = lines.findIndex(line => /^Event Location:?$/i.test(line))
  const details = emptyDetails([url])
  // The address line precedes the venue name.
  details.venue = at >= 0 ? (lines[at + 2] ?? lines[at + 1] ?? null) : null
  details.schedulePending = /Race Schedule:\s*To Be Announced/i.test(text.replace(/\s+/g, ' '))
  return details
}

async function eventDetails(url: string, year: number): Promise<TriathlonCalendarDetails> {
  switch (calendarSeries(url)) {
    case 'ironman':
      return ironmanDetails(url, year)
    case 'supertri':
      return supertriDetails(url, year)
    case 't100':
      return t100Details(url)
    case 'hyrox':
      return hyroxDetails(url)
    default:
      return emptyDetails([url])
  }
}

interface CachedDetails {
  fetched: string
  rawEvents: UnknownRecord
  events: Map<string, TriathlonCalendarDetails>
}

async function cachedDetails(): Promise<CachedDetails | null> {
  let cached: unknown
  try {
    cached = JSON.parse(await readFile(OUTPUT, 'utf8'))
  } catch {
    return null
  }
  if (!isRecord(cached) || typeof cached.fetched !== 'string' || !isRecord(cached.events))
    return null
  const events = new Map<string, TriathlonCalendarDetails>()
  for (const [url, value] of Object.entries(cached.events)) {
    const details = parseCalendarDetails(value)
    if (details) events.set(url, details)
  }
  return { fetched: cached.fetched, rawEvents: cached.events, events }
}

/**
 * A page that loads but no longer parses returns empty fields, which would overwrite the routes and
 * schedule of the last good sync. Each field the new extraction lacks falls back to the cached
 * value, and `kept` records it with the date it was actually extracted.
 */
function keepCached(
  fresh: TriathlonCalendarDetails,
  cached: TriathlonCalendarDetails,
  cachedFetched: string,
): string[] {
  const keep = (label: string): void => {
    fresh.kept[label] = cached.kept[label] ?? cachedFetched
  }
  for (const course of cached.course) {
    if (fresh.course.some(entry => entry.leg === course.leg)) continue
    fresh.course.push(course)
    if (!fresh.sources.includes(course.source)) fresh.sources.push(course.source)
    keep(`${course.leg} course`)
  }
  fresh.course.sort((left, right) => courseOrder[left.leg] - courseOrder[right.leg])
  if (!fresh.schedule && cached.schedule && !fresh.schedulePending) {
    fresh.schedule = cached.schedule
    keep('schedule')
  }
  for (const leg of TRIATHLON_CALENDAR_LEGS) {
    const terrain = cached.terrain[leg]
    if (fresh.terrain[leg] || !terrain) continue
    fresh.terrain[leg] = terrain
    keep(`${leg} terrain`)
  }
  if (!fresh.airC && cached.airC) {
    fresh.airC = cached.airC
    keep('air')
  }
  if (!fresh.waterC && cached.waterC) {
    fresh.waterC = cached.waterC
    keep('water')
  }
  if (!fresh.venue && cached.venue) {
    fresh.venue = cached.venue
    keep('venue')
  }
  if (!fresh.venueMap && cached.venueMap) {
    fresh.venueMap = cached.venueMap
    keep('venue map')
  }
  if (!fresh.athleteGuide && cached.athleteGuide) {
    fresh.athleteGuide = cached.athleteGuide
    keep('athlete guide')
  }
  return Object.keys(fresh.kept)
}

function summary(details: TriathlonCalendarDetails): string {
  const course = details.course
    .map(course => `${course.leg} ${(course.distanceM / 1000).toFixed(1)} km`)
    .join(', ')
  const schedule = details.schedule
    ? `schedule ${details.schedule.raceDate} (${details.schedule.days.length} days)`
    : details.schedulePending
      ? 'schedule pending'
      : 'no schedule'
  const terrain = Object.entries(details.terrain)
    .map(([leg, label]) => `${leg} ${label}`)
    .join(', ')
  return [course || 'no course', schedule, terrain, details.venue].filter(Boolean).join(' · ')
}

async function main(): Promise<void> {
  const { values } = parseArgs({
    options: { source: { type: 'string' }, output: { type: 'string' } },
  })
  if (
    values.source &&
    !Object.values(TRIATHLON_CALENDAR_CATALOG).some(
      season => values.source && values.source in season.sources,
    )
  )
    throw new Error(`Unknown calendar source: ${values.source}`)
  const output = values.output ? path.resolve(values.output) : OUTPUT
  const cache = await cachedDetails()
  if (values.source && !cache) throw new Error('A scoped sync requires a readable calendar cache')
  const events: UnknownRecord = values.source ? { ...cache?.rawEvents } : {}
  const fetched = new Intl.DateTimeFormat('en-CA', { timeZone: 'America/Toronto' }).format(
    new Date(),
  )
  let failures = 0
  for (const [yearKey, season] of Object.entries(TRIATHLON_CALENDAR_CATALOG)) {
    for (const [url, metadata] of Object.entries(season.sources)) {
      if (values.source && url !== values.source) continue
      const cached = cache?.events.get(url)
      let details: TriathlonCalendarDetails
      try {
        details = await eventDetails(url, Number(yearKey))
        details.fetched = fetched
      } catch (error) {
        console.warn(`${metadata.name}: ${error instanceof Error ? error.message : String(error)}`)
        failures++
        if (!cached || !cache) continue
        details = { ...emptyDetails(cached.sources), schedulePending: cached.schedulePending }
        details.fetched = cached.fetched ?? cache.fetched
      }
      const kept =
        cached && cache ? keepCached(details, cached, cached.fetched ?? cache.fetched) : []
      if (kept.length > 0) {
        failures++
        console.warn(`${metadata.name}: kept from an earlier sync: ${kept.join(', ')}`)
      }
      events[url] = details
      console.log(`${metadata.name}: ${summary(details)}`)
    }
  }
  await writeFile(
    output,
    `${JSON.stringify({ fetched: values.source ? (cache?.fetched ?? fetched) : fetched, events }, null, 2)}\n`,
  )
  execFileSync('pnpm', ['exec', 'oxfmt', '--write', output], {
    cwd: process.cwd(),
    stdio: 'inherit',
  })
  console.log(
    `wrote ${path.relative(process.cwd(), output)}${failures ? ` (${failures} events incomplete, see warnings)` : ''}`,
  )
  if (failures) process.exitCode = 1
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  main().catch(error => {
    console.error(error)
    process.exitCode = 1
  })
}
