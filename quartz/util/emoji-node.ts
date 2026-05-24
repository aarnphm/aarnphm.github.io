import { readFile } from 'node:fs/promises'
import path from 'path'
import { isStringRecord } from './type-guards'

const emojiAssetRoot = path.join(process.cwd(), 'quartz/util/emojimap')
const base64Chunks = new Map<string, Record<string, string>>()
let codePointToName: Record<string, string> | undefined = undefined

async function readEmojiRecord(assetPath: string): Promise<Record<string, string>> {
  const raw = await readFile(path.join(emojiAssetRoot, assetPath), 'utf8')
  const value: unknown = JSON.parse(raw)
  if (!isStringRecord(value)) throw new Error(`invalid emoji asset ${assetPath}`)
  return value
}

async function ensureCodePointToName(): Promise<Record<string, string>> {
  if (!codePointToName) {
    codePointToName = await readEmojiRecord('codepoint-to-name.json')
  }
  return codePointToName
}

function emojiBase64Prefix(codepoint: string): string {
  return codepoint.split('-')[0].toUpperCase().slice(0, 3)
}

async function ensureBase64Chunk(prefix: string): Promise<Record<string, string>> {
  const cached = base64Chunks.get(prefix)
  if (cached) return cached
  const chunk = await readEmojiRecord(`base64/${prefix}.json`)
  base64Chunks.set(prefix, chunk)
  return chunk
}

export async function loadEmoji(code: string): Promise<string> {
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
