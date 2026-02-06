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

type EmojiMap = { codePointToName: Record<string, string>; nameToBase64: Record<string, string> }

export type EmojiEntry = { name: string; codepoint: string; emoji: string }

let emojimap: EmojiMap | undefined = undefined
let emojiEntries: EmojiEntry[] | null = null

async function ensureEmojiMap(): Promise<EmojiMap | undefined> {
  if (!emojimap) {
    const data = await import('./emojimap.json')
    emojimap = data
  }
  return emojimap
}

export async function loadEmoji(code: string) {
  await ensureEmojiMap()

  const name = emojimap!.codePointToName[`${code.toUpperCase()}`]
  if (!name) throw new Error(`codepoint ${code} not found in map`)

  const b64 = emojimap!.nameToBase64[name]
  if (!b64) throw new Error(`name ${name} not found in map`)

  return b64
}

export async function getEmojiEntries(): Promise<EmojiEntry[]> {
  if (emojiEntries) return emojiEntries
  await ensureEmojiMap()

  emojiEntries = Object.entries(emojimap!.codePointToName).map(([codepoint, name]) => ({
    name,
    codepoint,
    emoji: codePointToEmoji(codepoint),
  }))

  return emojiEntries
}
