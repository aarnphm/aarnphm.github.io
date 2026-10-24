/** Clipboard write with the legacy fallback for panes that deny the async clipboard. */
export async function copyText(text: string): Promise<void> {
  try {
    await navigator.clipboard.writeText(text)
    return
  } catch {
    // fall through
  }
  const area = document.createElement('textarea')
  area.value = text
  area.setAttribute('readonly', '')
  area.style.position = 'fixed'
  area.style.opacity = '0'
  document.body.appendChild(area)
  area.select()
  const ok = document.execCommand('copy')
  area.remove()
  if (!ok) throw new Error('copy rejected')
}

/** Typographic marks the markdown pipeline applies, undone so cue text matches the rendered text. */
const plainChar: Record<string, string> = {
  '\u2018': "'",
  '\u2019': "'",
  '\u201c': '"',
  '\u201d': '"',
  '\u2026': '...',
  '\u2014': '--',
  '\u2013': '--',
}

/** `text` with typographic marks undone; `at[i]` is the source index of normalized char i. */
export function plain(text: string): { text: string; at: number[] } {
  let out = ''
  const at: number[] = []
  for (let i = 0; i < text.length; i++) {
    const piece = plainChar[text[i]] ?? text[i]
    out += piece
    for (let k = 0; k < piece.length; k++) at.push(i)
  }
  return { text: out, at }
}

export function parseStamp(href: string): number | null {
  try {
    const url = new URL(href)
    if (!/(^|\.)youtu(\.be|be\.com)$/.test(url.hostname)) return null
    const t = url.searchParams.get('t')
    if (!t) return null
    const seconds = Number.parseInt(t, 10)
    return Number.isFinite(seconds) ? seconds : null
  } catch {
    return null
  }
}

export function formatStamp(total: number, hours: boolean): string {
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const s = Math.floor(total % 60)
  const mm = hours || h > 0 ? String(m).padStart(2, '0') : String(m)
  return h > 0 || hours
    ? `${h}:${mm}:${String(s).padStart(2, '0')}`
    : `${mm}:${String(s).padStart(2, '0')}`
}
