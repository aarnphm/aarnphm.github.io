import matter from "gray-matter"
import { execSync } from "node:child_process"
import { promises as fs } from "node:fs"
import path from "node:path"

interface Coordinates {
  lat: string
  lon: string
}

interface PlaceEntry {
  title: string
  url: string
  tags: string[]
  date?: string
  coordinates?: Coordinates
  address?: string
}

const ARENA_FILE = path.join(process.cwd(), "content", "are.na.md")
const PLACES_DIR = path.join(process.cwd(), "content", "places")
const CATEGORY_LINK = "[[places to go]]"
const DEFAULT_TAGS = ["places"]
const DEFAULT_RATING = 5
const NOMINATIM_ENDPOINT = "https://nominatim.openstreetmap.org/reverse"
const NOMINATIM_USER_AGENT =
  process.env.NOMINATIM_USER_AGENT ?? "aarnphm-garden-sync/1.0 (contact@aarnphm.xyz)"
const NOMINATIM_DELAY_MS = Number(process.env.NOMINATIM_DELAY_MS ?? "1100")

const addressCache = new Map<string, string>()

function sanitizeFileName(title: string): string {
  return title
    .replace(/[\\/]/g, "-")
    .replace(/[:*?"<>|]/g, " ")
    .replace(/\s+/g, " ")
    .trim()
}

function stripDecorations(input: string): string {
  return input.replace(/\s*\[(?:\*\*|--)]\s*$/, "").trim()
}

function normalizeDate(value?: string): string | undefined {
  if (!value) {
    return undefined
  }
  const trimmed = value.trim().replace(/^"|"$/g, "")
  if (/^\d{4}-\d{2}-\d{2}$/.test(trimmed)) {
    return trimmed
  }
  const slash = trimmed.match(/^(\d{1,2})\/(\d{1,2})\/(\d{4})$/)
  if (slash) {
    const [, month, day, year] = slash
    const pad = (num: string) => num.padStart(2, "0")
    return `${year}-${pad(month)}-${pad(day)}`
  }
  return trimmed
}

function normalizeCoordinatePart(value: string): string {
  return value.trim().replace(/^"|"$/g, "")
}

function normalizeUrl(value?: string): string | undefined {
  if (!value) {
    return undefined
  }
  const trimmed = value.trim()
  return trimmed.length > 0 ? trimmed : undefined
}

function parseCoordinate(value?: string): Coordinates | undefined {
  if (!value) {
    return undefined
  }
  const inner = value.replace(/^[[]|[\]]$/g, "")
  const parts = inner
    .split(",")
    .map((part) => normalizeCoordinatePart(part))
    .filter(Boolean)
  if (parts.length < 2) {
    return undefined
  }
  const [lat, lon] = parts
  return { lat, lon }
}

function parseTagList(value?: string): string[] {
  if (!value) {
    return []
  }
  const trimmed = value.trim()
  if (!trimmed.startsWith("[") || !trimmed.endsWith("]")) {
    return [trimmed.replace(/^"|"$/g, "")].filter(Boolean)
  }
  const inner = trimmed.slice(1, -1).trim()
  if (inner.length === 0) {
    return []
  }
  return inner
    .split(",")
    .map((item) => item.trim().replace(/^"|"$/g, ""))
    .filter(Boolean)
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

function buildAddressFromParts(parts?: Record<string, string>): string | undefined {
  if (!parts) {
    return undefined
  }

  const segments: string[] = []
  const streetComponents = [parts.house_number, parts.road].filter(Boolean).join(" ").trim()
  if (streetComponents.length > 0) {
    segments.push(streetComponents)
  } else if (parts.road) {
    segments.push(parts.road)
  }

  const localityOrder = ["neighbourhood", "suburb", "city_district", "town", "city", "village"]
  for (const key of localityOrder) {
    const value = parts[key]
    if (value && !segments.includes(value)) {
      segments.push(value)
    }
  }

  if (parts.state) {
    segments.push(parts.state)
  }
  if (parts.postcode) {
    segments.push(parts.postcode)
  }
  if (parts.country) {
    segments.push(parts.country)
  }

  const cleaned = segments.map((segment) => segment.trim()).filter((segment) => segment.length > 0)
  return cleaned.length > 0 ? cleaned.join(", ") : undefined
}

type NominatimResponse = { display_name?: string; address?: Record<string, string> }

async function reverseGeocode(coords: Coordinates): Promise<string | undefined> {
  const key = `${coords.lat},${coords.lon}`
  if (addressCache.has(key)) {
    return addressCache.get(key)
  }

  const url = new URL(NOMINATIM_ENDPOINT)
  url.searchParams.set("format", "jsonv2")
  url.searchParams.set("lat", coords.lat)
  url.searchParams.set("lon", coords.lon)

  try {
    const response = await fetch(url, {
      headers: { "User-Agent": NOMINATIM_USER_AGENT, Accept: "application/json" },
    })

    if (!response.ok) {
      console.warn(
        `[syncPlaces] Failed to fetch address for ${key}: ${response.status} ${response.statusText}`,
      )
      return undefined
    }

    const data = (await response.json()) as NominatimResponse
    const resolved = data.display_name ?? buildAddressFromParts(data.address)
    if (resolved) {
      addressCache.set(key, resolved)
    }

    // be polite with the public API
    if (NOMINATIM_DELAY_MS > 0) {
      await delay(NOMINATIM_DELAY_MS)
    }

    return resolved
  } catch (error) {
    console.warn(`[syncPlaces] Error during reverse geocoding for ${key}:`, error)
    return undefined
  }
}

type ParsedMeta = { date?: string; tags?: string[]; coord?: Coordinates; title?: string }

function parseMeta(lines: string[]): ParsedMeta {
  const meta: ParsedMeta = {}
  for (const raw of lines) {
    const match = raw.match(/^\s{4}-\s*([^:]+):\s*(.*)$/)
    if (!match) {
      continue
    }
    const key = match[1].trim().toLowerCase()
    const value = match[2].trim()
    if (key === "date") {
      meta.date = normalizeDate(value)
    } else if (key === "tags") {
      meta.tags = parseTagList(value)
    } else if (key === "coord") {
      meta.coord = parseCoordinate(value)
    } else if (key === "title") {
      meta.title = value.trim()
    }
  }
  return meta
}

function findPlacesSection(lines: string[]): { start: number; end: number } | null {
  let start = -1
  for (let index = 0; index < lines.length; index += 1) {
    if (lines[index].trim().toLowerCase() === "## places") {
      start = index + 1
      break
    }
  }
  if (start === -1) {
    return null
  }
  let end = lines.length
  for (let index = start; index < lines.length; index += 1) {
    const trimmed = lines[index].trim()
    if (trimmed.startsWith("## ") && trimmed.toLowerCase() !== "## places") {
      end = index
      break
    }
  }
  return { start, end }
}

function parsePlaces(markdown: string): PlaceEntry[] {
  const lines = markdown.split(/\r?\n/)
  const section = findPlacesSection(lines)
  if (!section) {
    return []
  }
  const entries: PlaceEntry[] = []
  for (let index = section.start; index < section.end; index += 1) {
    const line = lines[index]
    const match = line.match(/^-\s+(https?:\S+)\s+--\s+(.+)$/)
    if (!match) {
      continue
    }
    const [, url, rawTitle] = match
    const title = stripDecorations(rawTitle)
    let metaPointer = index + 1
    while (metaPointer < section.end && lines[metaPointer].trim() === "") {
      metaPointer += 1
    }
    if (!lines[metaPointer]?.startsWith("  - [meta]:")) {
      throw new Error(`Missing [meta] block after place "${title}" on line ${index + 1}`)
    }
    const metaLines: string[] = []
    for (let cursor = metaPointer + 1; cursor < section.end; cursor += 1) {
      const metaLine = lines[cursor]
      if (/^\s{4}-/.test(metaLine)) {
        metaLines.push(metaLine)
        continue
      }
      break
    }
    const meta = parseMeta(metaLines)
    index = metaPointer + metaLines.length
    entries.push({
      title: meta.title && meta.title.length > 0 ? meta.title : title,
      url,
      tags: meta.tags ?? [],
      date: meta.date,
      coordinates: meta.coord,
    })
  }
  return entries
}

function formatTypeTag(tag: string): string | undefined {
  const trimmed = tag.trim()
  if (!trimmed) {
    return undefined
  }
  if (trimmed.startsWith("[[") && trimmed.endsWith("]]")) {
    return trimmed
  }
  return `${trimmed}`
}

function buildTypeList(tags: string[], existingType: unknown): string[] {
  const types = new Set<string>()
  if (Array.isArray(existingType)) {
    for (const raw of existingType) {
      const value = String(raw).trim()
      if (!value) {
        continue
      }
      types.add(value)
    }
  }
  for (const tag of tags) {
    const formatted = formatTypeTag(tag)
    if (!formatted) {
      continue
    }
    types.add(formatted)
  }
  return Array.from(types)
}

function formatModified(date: Date): string {
  const pad = (value: number) => String(value).padStart(2, "0")
  const tzOffsetMinutes = -date.getTimezoneOffset()
  const sign = tzOffsetMinutes >= 0 ? "+" : "-"
  const abs = Math.abs(tzOffsetMinutes)
  const hours = pad(Math.floor(abs / 60))
  const minutes = pad(abs % 60)
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())} GMT${sign}${hours}:${minutes}`
}

function buildFrontmatter(
  entry: PlaceEntry,
  existing: Record<string, unknown>,
): Record<string, unknown> {
  const categories = new Set<string>()
  if (Array.isArray(existing.categories)) {
    for (const value of existing.categories) {
      const stringified = String(value).trim()
      if (stringified) {
        categories.add(stringified)
      }
    }
  }
  categories.add(CATEGORY_LINK)

  const coordinates = entry.coordinates
    ? [entry.coordinates.lat, entry.coordinates.lon]
    : Array.isArray(existing.coordinates)
      ? existing.coordinates.map((coord) => String(coord))
      : undefined

  const data: Record<string, unknown> = {}
  if (entry.address && entry.address.trim().length > 0) {
    data.address = entry.address.trim()
  } else if (typeof existing.address === "string" && existing.address.trim().length > 0) {
    data.address = existing.address.trim()
  }
  data.category = Array.from(categories)
  if (coordinates) {
    data.coordinates = coordinates
  }
  if (entry.date) {
    data.date = entry.date
  } else if (typeof existing.date === "string") {
    data.date = existing.date
  }
  if (typeof existing.description === "string" && existing.description.trim().length > 0) {
    data.description = existing.description.trim()
  }
  data.id = entry.title
  data.modified = formatModified(new Date())
  data.rating = typeof existing.rating === "number" ? existing.rating : DEFAULT_RATING
  if (Array.isArray(existing.tags) && existing.tags.length > 0) {
    data.tags = existing.tags.map((tag) => String(tag))
  } else {
    data.tags = DEFAULT_TAGS
  }
  data.title = entry.title
  data.type = buildTypeList(entry.tags, existing.type)
  data.url = entry.url
  return data
}

type ExistingNote = { data: Record<string, unknown>; body: string }

function parseExistingContent(raw?: string): ExistingNote {
  if (!raw) {
    return { data: {}, body: "" }
  }
  const parsed = matter(raw)
  return { data: parsed.data as Record<string, unknown>, body: parsed.content.trimEnd() }
}

async function buildExistingPlaceIndex(): Promise<Map<string, string>> {
  const index = new Map<string, string>()
  let entries: string[]
  try {
    entries = await fs.readdir(PLACES_DIR)
  } catch (error) {
    const nodeError = error as NodeJS.ErrnoException
    if (nodeError.code === "ENOENT") {
      return index
    }
    throw error
  }

  await Promise.all(
    entries
      .filter((file) => file.endsWith(".md"))
      .map(async (file) => {
        const filePath = path.join(PLACES_DIR, file)
        try {
          const raw = await fs.readFile(filePath, "utf8")
          const parsed = matter(raw)
          const normalizedUrl = normalizeUrl(
            typeof parsed.data.url === "string" ? parsed.data.url : undefined,
          )
          if (!normalizedUrl) {
            return
          }
          if (index.has(normalizedUrl) && index.get(normalizedUrl) !== filePath) {
            console.warn(
              `[syncPlaces] Duplicate URL ${normalizedUrl} already mapped to ${index.get(normalizedUrl)}`,
            )
            return
          }
          index.set(normalizedUrl, filePath)
        } catch (error) {
          console.warn(`[syncPlaces] Unable to index place file ${filePath}:`, error)
        }
      }),
  )

  return index
}

function renderPlaceFile(entry: PlaceEntry, existing: ExistingNote): string {
  const frontmatter = buildFrontmatter(entry, existing.data)
  const serializedBody = existing.body.length > 0 ? `${existing.body}\n` : ""
  return matter.stringify(serializedBody, frontmatter)
}

async function ensureAddress(
  entry: PlaceEntry,
  existingData: Record<string, unknown>,
): Promise<void> {
  if (entry.address && entry.address.trim().length > 0) {
    entry.address = entry.address.trim()
    return
  }

  const existingAddress =
    typeof existingData.address === "string" ? existingData.address.trim() : undefined
  if (existingAddress && existingAddress.length > 0) {
    entry.address = existingAddress
    return
  }

  if (!entry.coordinates) {
    return
  }

  const resolved = await reverseGeocode(entry.coordinates)
  if (resolved) {
    entry.address = resolved.trim()
  }
}

async function writePlace(
  entry: PlaceEntry,
  existingIndex: Map<string, string>,
): Promise<{ path: string; changed: boolean; existed: boolean; hasAddress: boolean }> {
  const normalizedUrl = normalizeUrl(entry.url)
  const fallbackPath = path.join(PLACES_DIR, `${sanitizeFileName(entry.title)}.md`)
  const filePath = (normalizedUrl ? existingIndex.get(normalizedUrl) : undefined) ?? fallbackPath
  let existing: string | undefined
  try {
    existing = await fs.readFile(filePath, "utf8")
  } catch (error) {
    const nodeError = error as NodeJS.ErrnoException
    if (nodeError.code !== "ENOENT") {
      throw error
    }
  }
  const existingNote = parseExistingContent(existing)
  await ensureAddress(entry, existingNote.data)
  const nextContent = renderPlaceFile(entry, existingNote)
  if (existing && existing === nextContent) {
    return {
      path: filePath,
      changed: false,
      existed: true,
      hasAddress: Boolean(entry.address && entry.address.length > 0),
    }
  }
  await fs.writeFile(filePath, nextContent, "utf8")
  if (normalizedUrl) {
    existingIndex.set(normalizedUrl, filePath)
  }
  return {
    path: filePath,
    changed: true,
    existed: Boolean(existing),
    hasAddress: Boolean(entry.address && entry.address.length > 0),
  }
}

async function main() {
  const arena = await fs.readFile(ARENA_FILE, "utf8")
  const places = parsePlaces(arena)
  if (places.length === 0) {
    console.log("No places section found in content/are.na.md")
    return
  }
  const existingIndex = await buildExistingPlaceIndex()
  await fs.mkdir(PLACES_DIR, { recursive: true })
  let created = 0
  let updated = 0
  let skipped = 0
  let missingAddress = 0
  const changedFiles: string[] = []
  for (const entry of places) {
    const result = await writePlace(entry, existingIndex)
    if (!result.changed) {
      skipped += 1
    } else if (result.existed) {
      updated += 1
      changedFiles.push(result.path)
    } else {
      created += 1
      changedFiles.push(result.path)
    }
    if (result.hasAddress) {
      continue
    }
    missingAddress += 1
  }
  console.log(
    `Processed ${places.length} places (created: ${created}, updated: ${updated}, skipped: ${skipped}).`,
  )
  if (missingAddress > 0) {
    console.warn(`Unable to resolve addresses for ${missingAddress} place(s).`)
  }
  if (changedFiles.length > 0) {
    console.log(`Formatting ${changedFiles.length} changed file(s) with prettier...`)
    try {
      execSync(`pnpm prettier --write ${changedFiles.map((f) => `"${f}"`).join(" ")}`, {
        cwd: process.cwd(),
        stdio: "inherit",
      })
    } catch (error) {
      console.warn("Failed to run prettier:", error)
    }
  }
}

main().catch((error) => {
  console.error(error)
  process.exitCode = 1
})
