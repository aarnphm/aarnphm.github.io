import { isStringRecord } from './type-guards'

const U200D = String.fromCharCode(8205)
const UFE0Fg = /\uFE0F/g

export function getIconCode(char: string) {
  return toCodePoint(char.indexOf(U200D) < 0 ? char.replace(UFE0Fg, '') : char)
}

function toCodePoint(unicodeSurrogates: string) {
  const r = []
  let c = 0,
    p = 0,
    i = 0

  while (i < unicodeSurrogates.length) {
    c = unicodeSurrogates.charCodeAt(i++)
    if (p) {
      r.push((65536 + ((p - 55296) << 10) + (c - 56320)).toString(16))
      p = 0
    } else if (55296 <= c && c <= 56319) {
      p = c
    } else {
      r.push(c.toString(16))
    }
  }
  return r.join('-')
}

function codePointToEmoji(codepoint: string): string {
  const codepoints = codepoint.split('-').map(cp => parseInt(cp, 16))
  return String.fromCodePoint(...codepoints)
}

export type EmojiEntry = { name: string; codepoint: string; emoji: string }

let codePointToName: Record<string, string> | undefined = undefined
let emojiEntries: EmojiEntry[] | null = null
const base64Chunks = new Map<string, Record<string, string>>()
let assetManifest: Promise<Record<string, string>> | undefined = undefined

function staticScriptsUrl(assetPath: string): string {
  const source = new URL(import.meta.url)
  const scripts = '/static/scripts/'
  const scriptsIndex = source.pathname.indexOf(scripts)
  if (scriptsIndex >= 0) {
    source.pathname = `${source.pathname.slice(0, scriptsIndex + scripts.length)}${assetPath}`
    source.search = ''
    source.hash = ''
    return source.href
  }

  return new URL(`static/scripts/${assetPath}`, document.baseURI).href
}

async function loadAssetManifest(): Promise<Record<string, string>> {
  assetManifest ??= fetch(staticScriptsUrl('asset-manifest.json'))
    .then(async response => (response.ok ? ((await response.json()) as unknown) : {}))
    .then(value => (isStringRecord(value) ? value : {}))
    .catch(() => ({}))
  return assetManifest
}

async function emojiAssetUrl(assetPath: string): Promise<string> {
  const manifest = await loadAssetManifest()
  const logicalPath = `static/scripts/emoji/${assetPath}`
  const emittedPath = manifest[logicalPath] ?? logicalPath
  const relativePath = emittedPath.startsWith('static/scripts/')
    ? emittedPath.slice('static/scripts/'.length)
    : emittedPath
  if (relativePath !== emittedPath) return staticScriptsUrl(relativePath)

  return new URL(emittedPath, document.baseURI).href
}

async function fetchEmojiRecord(assetPath: string): Promise<Record<string, string>> {
  const response = await fetch(await emojiAssetUrl(assetPath))
  if (!response.ok) throw new Error(`failed to load emoji asset ${assetPath}`)
  const value: unknown = await response.json()
  if (!isStringRecord(value)) throw new Error(`invalid emoji asset ${assetPath}`)
  return value
}

async function ensureCodePointToName(): Promise<Record<string, string>> {
  if (!codePointToName) {
    codePointToName = await fetchEmojiRecord('codepoint-to-name.json')
  }
  return codePointToName
}

function emojiBase64Prefix(codepoint: string): string {
  return codepoint.split('-')[0].toUpperCase().slice(0, 3)
}

async function ensureBase64Chunk(prefix: string): Promise<Record<string, string>> {
  const cached = base64Chunks.get(prefix)
  if (cached) return cached
  const chunk = await fetchEmojiRecord(`base64/${prefix}.json`)
  base64Chunks.set(prefix, chunk)
  return chunk
}

export async function loadEmoji(code: string) {
  const codepoint = code.toUpperCase()
  const names = await ensureCodePointToName()

  const name = names[codepoint]
  if (!name) throw new Error(`codepoint ${code} not found in map`)

  const prefix = emojiBase64Prefix(codepoint)
  const chunk = await ensureBase64Chunk(prefix)
  const b64 = chunk[codepoint]
  if (!b64) throw new Error(`name ${name} not found in map`)

  return b64
}

export async function getEmojiEntries(): Promise<EmojiEntry[]> {
  if (emojiEntries) return emojiEntries
  const names = await ensureCodePointToName()

  emojiEntries = Object.entries(names).map(([codepoint, name]) => ({
    name,
    codepoint,
    emoji: codePointToEmoji(codepoint),
  }))

  return emojiEntries
}
