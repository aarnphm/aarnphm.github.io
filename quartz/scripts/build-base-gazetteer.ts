import matter from 'gray-matter'
import { readFileSync } from 'node:fs'
import { open, readdir, writeFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import type { GazetteerFeature, GazetteerFile, GazetteerKind } from '../util/base/geo-types'
import { normalizeAnchorText } from '../util/base/geo-parse'
import { isRecord } from '../util/type-guards'

const gazetteerPath = join(dirname(fileURLToPath(import.meta.url)), '../util/base/gazetteer.json')
const contentDir = join(process.cwd(), 'content')
const nominatim = 'https://nominatim.openstreetmap.org'
const userAgent =
  process.env.NOMINATIM_USER_AGENT ?? 'aarnphm-garden-sync/1.0 (contact@aarnphm.xyz)'
const delayMs = Number(process.env.NOMINATIM_DELAY_MS ?? '1100')

type Pin = { lat: number; lon: number }
type Lookup = {
  osmType: string
  osmId: number
  name?: string
  lat: number
  lon: number
  bbox: [number, number, number, number]
}

function isKind(value: unknown): value is GazetteerKind {
  return value === 'park' || value === 'neighbourhood' || value === 'city'
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

function isFeature(value: unknown): value is GazetteerFeature {
  if (!isRecord(value)) return false
  if (
    typeof value.id !== 'string' ||
    typeof value.name !== 'string' ||
    typeof value.city !== 'string'
  ) {
    return false
  }
  if (!isKind(value.kind) || !Array.isArray(value.aliases)) return false
  if (!value.aliases.every(alias => typeof alias === 'string')) return false
  if (
    !Array.isArray(value.centroid) ||
    value.centroid.length !== 2 ||
    !value.centroid.every(isFiniteNumber)
  ) {
    return false
  }
  if (!Array.isArray(value.bbox) || value.bbox.length !== 4 || !value.bbox.every(isFiniteNumber))
    return false
  const [minLat, maxLat, minLon, maxLon] = value.bbox
  return minLat <= maxLat && minLon <= maxLon
}

export function readGazetteer(path = gazetteerPath): GazetteerFile {
  const parsed: unknown = JSON.parse(readFileSync(path, 'utf8'))
  if (!isRecord(parsed) || typeof parsed.licence !== 'string' || !Array.isArray(parsed.features)) {
    throw new Error('gazetteer is not a GazetteerFile')
  }
  const features: GazetteerFeature[] = []
  const ids = new Set<string>()
  for (const entry of parsed.features) {
    if (!isFeature(entry)) throw new Error('gazetteer feature does not match GazetteerFeature')
    if (ids.has(entry.id)) throw new Error(`duplicate gazetteer id ${entry.id}`)
    ids.add(entry.id)
    features.push(entry)
  }
  return { licence: parsed.licence, features }
}

function osmCode(id: string): string | null {
  const match = /^osm:(node|way|relation):(\d+)$/.exec(id)
  if (!match) return null
  const prefix = match[1] === 'node' ? 'N' : match[1] === 'way' ? 'W' : 'R'
  return `${prefix}${match[2]}`
}

function contains(feature: GazetteerFeature, pin: Pin): boolean {
  const [minLat, maxLat, minLon, maxLon] = feature.bbox
  return pin.lat >= minLat && pin.lat <= maxLat && pin.lon >= minLon && pin.lon <= maxLon
}

function knownName(features: readonly GazetteerFeature[], name: string): boolean {
  const needle = normalizeAnchorText(name)
  return features.some(feature =>
    [feature.name, feature.city, ...feature.aliases].some(
      value => normalizeAnchorText(value) === needle,
    ),
  )
}

let nextRequest = 0

async function polite(url: URL): Promise<unknown> {
  const wait = nextRequest - Date.now()
  if (wait > 0) await new Promise(resolve => setTimeout(resolve, wait))
  try {
    const response = await fetch(url, {
      headers: { 'User-Agent': userAgent, Accept: 'application/json' },
      signal: AbortSignal.timeout(15_000),
    })
    if (!response.ok) throw new Error(`${url.pathname} ${response.status}`)
    return await response.json()
  } finally {
    nextRequest = Date.now() + delayMs
  }
}

function bboxOf(value: unknown): [number, number, number, number] | null {
  if (!Array.isArray(value) || value.length < 4) return null
  const nums = value.slice(0, 4).map(item => Number(item))
  if (!nums.every(Number.isFinite)) return null
  const [minLat, maxLat, minLon, maxLon] = nums
  if (minLat > maxLat || minLon > maxLon) return null
  return [minLat, maxLat, minLon, maxLon]
}

function lookupFeature(value: unknown): Lookup | null {
  if (!isRecord(value)) return null
  const lat = Number(value.lat)
  const lon = Number(value.lon)
  const bbox = bboxOf(value.boundingbox)
  const osmId = Number(value.osm_id)
  if (!Number.isFinite(lat) || !Number.isFinite(lon) || !Number.isFinite(osmId) || !bbox)
    return null
  if (typeof value.osm_type !== 'string') return null
  return {
    osmType: value.osm_type,
    osmId,
    name: typeof value.name === 'string' ? value.name : undefined,
    lat,
    lon,
    bbox,
  }
}

async function refreshFeature(feature: GazetteerFeature): Promise<GazetteerFeature> {
  const code = osmCode(feature.id)
  if (!code) return feature
  const url = new URL('/lookup', nominatim)
  url.searchParams.set('osm_ids', code)
  url.searchParams.set('format', 'jsonv2')
  const body = await polite(url)
  const hit = Array.isArray(body) ? lookupFeature(body[0]) : null
  if (!hit) return feature
  return { ...feature, centroid: [hit.lat, hit.lon], bbox: hit.bbox }
}

async function walk(dir: string): Promise<string[]> {
  const out: string[] = []
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    if (entry.name.startsWith('.')) continue
    const full = join(dir, entry.name)
    if (entry.isDirectory()) out.push(...(await walk(full)))
    else if (entry.name.endsWith('.md')) out.push(full)
  }
  return out
}

async function readHead(path: string): Promise<string> {
  const handle = await open(path, 'r')
  try {
    const buf = Buffer.alloc(4000)
    const { bytesRead } = await handle.read(buf, 0, 4000, 0)
    return buf.subarray(0, bytesRead).toString('utf8')
  } finally {
    await handle.close()
  }
}

function pinFrom(data: unknown): Pin | null {
  if (!isRecord(data)) return null
  const value = data.coordinates
  if (!Array.isArray(value) || value.length < 2) return null
  const lat = Number(value[0])
  const lon = Number(value[1])
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null
  if (lat < -90 || lat > 90 || lon < -180 || lon > 180) return null
  return { lat, lon }
}

async function collectPins(): Promise<Pin[]> {
  const pins: Pin[] = []
  for (const path of await walk(contentDir)) {
    const head = await readHead(path)
    if (!head.includes('coordinates:')) continue
    const parsed = matter(head)
    const pin = pinFrom(parsed.data)
    if (pin) pins.push(pin)
  }
  return pins
}

function cityName(address: unknown): string | null {
  if (!isRecord(address)) return null
  for (const key of ['city', 'town', 'village', 'municipality']) {
    const value = address[key]
    if (typeof value === 'string' && value.trim().length > 0) return value.trim()
  }
  return null
}

async function cityFor(pin: Pin): Promise<GazetteerFeature | null> {
  const reverse = new URL('/reverse', nominatim)
  reverse.searchParams.set('format', 'jsonv2')
  reverse.searchParams.set('lat', String(pin.lat))
  reverse.searchParams.set('lon', String(pin.lon))
  const reversed = await polite(reverse)
  if (!isRecord(reversed)) return null
  const name = cityName(reversed.address)
  if (!name) return null
  const search = new URL('/search', nominatim)
  search.searchParams.set('q', name)
  search.searchParams.set('format', 'jsonv2')
  search.searchParams.set('limit', '1')
  search.searchParams.set('featuretype', 'settlement')
  const found = await polite(search)
  const hit = Array.isArray(found) ? lookupFeature(found[0]) : null
  if (!hit) return null
  const id = `osm:${hit.osmType}:${hit.osmId}`
  const alias = normalizeAnchorText(hit.name ?? name)
  return {
    id,
    name: hit.name ?? name,
    aliases: alias.length > 0 ? [alias] : [],
    kind: 'city',
    city: hit.name ?? name,
    centroid: [hit.lat, hit.lon],
    bbox: hit.bbox,
  }
}

function writeGazetteer(file: GazetteerFile): Promise<void> {
  const features = [...file.features].sort((left, right) =>
    left.id < right.id ? -1 : left.id > right.id ? 1 : 0,
  )
  const body = {
    licence: file.licence,
    features: features.map(feature => ({
      id: feature.id,
      name: feature.name,
      aliases: feature.aliases,
      kind: feature.kind,
      city: feature.city,
      centroid: feature.centroid,
      bbox: feature.bbox,
    })),
  }
  return writeFile(gazetteerPath, `${JSON.stringify(body, null, 2)}\n`)
}

export async function refreshGazetteer(): Promise<number> {
  const current = readGazetteer()
  const features: GazetteerFeature[] = []
  for (const feature of current.features) {
    try {
      features.push(await refreshFeature(feature))
    } catch (error) {
      console.warn(
        `[gazetteer] kept ${feature.id}: ${error instanceof Error ? error.message : error}`,
      )
      features.push(feature)
    }
  }
  const pins = await collectPins()
  for (const pin of pins) {
    if (features.some(feature => feature.kind === 'city' && contains(feature, pin))) continue
    try {
      const city = await cityFor(pin)
      if (
        !city ||
        features.some(feature => feature.id === city.id) ||
        knownName(features, city.name)
      )
        continue
      features.push(city)
      console.log(`[gazetteer] added ${city.name}`)
    } catch (error) {
      console.warn(
        `[gazetteer] skipped pin ${pin.lat},${pin.lon}: ${error instanceof Error ? error.message : error}`,
      )
    }
  }
  await writeGazetteer({ licence: current.licence, features })
  return features.length
}

function invokedDirectly(): boolean {
  const entry = process.argv[1]
  return entry !== undefined && import.meta.url === pathToFileURL(entry).href
}

if (invokedDirectly()) {
  const checkOnly = process.argv.includes('--check')
  const run = checkOnly ? Promise.resolve(readGazetteer().features.length) : refreshGazetteer()
  run
    .then(count => console.log(count))
    .catch((error: unknown) => {
      console.error(error instanceof Error ? error.message : error)
      process.exitCode = 1
    })
}
