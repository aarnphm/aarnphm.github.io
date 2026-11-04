import { KIND_ORDER, type LatentDoc, type LatentKind } from './types'

/**
 * `filter:` in the latent search box: keep only some sources (note, pdf, image, base) in the
 * links pane. Types join with `,` or `+`, a token may repeat, `-type` drops one, and the rest of
 * the box is searched as usual, so `filter:pdf,image attention` embeds "attention" and ranks pdf
 * and image chunks only; `filter:pdf goto:lebesgue` lists matching pdfs; `filter:-image` alone
 * keeps the focus's neighbours minus images.
 */

const TOKEN = /(^|\s)filter:(\S*)/gi

const ALIASES: Record<string, LatentKind> = {
  note: 'note',
  notes: 'note',
  md: 'note',
  pdf: 'pdf',
  pdfs: 'pdf',
  paper: 'pdf',
  papers: 'pdf',
  image: 'image',
  images: 'image',
  img: 'image',
  imgs: 'image',
  base: 'base',
  bases: 'base',
}

export interface LatentFilter {
  /** the kept kinds in KIND_ORDER, or null when no `filter:` token names a known type */
  kinds: LatentKind[] | null
  /** names after `filter:` that are no type */
  unknown: string[]
  /** a `filter:` token is in the box, typed or not */
  present: boolean
  /** the box with every `filter:` token taken out */
  rest: string
  /** the `filter:` tokens as typed, for writing back in front of a completion */
  tokens: string
}

export function parseFilter(raw: string): LatentFilter {
  const keep = new Set<LatentKind>()
  const drop = new Set<LatentKind>()
  const unknown: string[] = []
  const tokens: string[] = []
  let present = false
  const rest = raw.replace(TOKEN, (_, lead: string, value: string) => {
    present = true
    tokens.push(`filter:${value}`)
    for (const part of value.split(/[,+|]/)) {
      const negated = /^[-!]/.test(part)
      const name = part.replace(/^[-!]/, '').toLowerCase()
      if (!name) continue
      const kind = ALIASES[name]
      if (!kind) unknown.push(name)
      else (negated ? drop : keep).add(kind)
    }
    return lead
  })
  const named = keep.size > 0 || drop.size > 0
  const kinds = named
    ? KIND_ORDER.filter(kind => (keep.size === 0 || keep.has(kind)) && !drop.has(kind))
    : null
  return {
    kinds,
    unknown,
    present,
    rest: rest.replace(/\s+/g, ' ').trim(),
    tokens: tokens.join(' '),
  }
}

/** True when the filter leaves every kind of the store in, so nothing needs masking. */
export function keepsAll(kinds: LatentKind[] | null, present: LatentKind[]): boolean {
  return !kinds || present.every(kind => kinds.includes(kind))
}

export function filterLabel(kinds: LatentKind[]): string {
  return kinds.length ? kinds.join(' + ') : 'nothing'
}

/** One byte per row, 1 where the row's document is of a kept kind (the worker skips zeros). */
export function rowMask(docByRow: LatentDoc[], kinds: LatentKind[], rows: number): Uint8Array {
  const mask = new Uint8Array(rows)
  const kept = new Set(kinds)
  for (let r = 0; r < rows; r++) if (docByRow[r] && kept.has(docByRow[r].s)) mask[r] = 1
  return mask
}

/**
 * Completions for the `filter:` token under the caret at the end of the box: the kinds not yet
 * named that start with the part being typed. `from`/`to` cover that part only, so a pick keeps
 * what came before it (`filter:pdf,im` → `filter:pdf,image`).
 */
export function filterCompletions(
  value: string,
  counts: Partial<Record<LatentKind, number>>,
): { kind: LatentKind; from: number; to: number; insert: string; count: number }[] {
  const m = /(?:^|\s)filter:(\S*)$/i.exec(value)
  if (!m) return []
  const typed = m[1]
  const cut = Math.max(typed.lastIndexOf(','), typed.lastIndexOf('+'), typed.lastIndexOf('|'))
  const named = new Set(
    typed
      .slice(0, cut + 1)
      .split(/[,+|]/)
      .map(part => ALIASES[part.replace(/^[-!]/, '').toLowerCase()])
      .filter(Boolean),
  )
  let partial = typed.slice(cut + 1)
  const sign = /^[-!]/.test(partial) ? partial[0] : ''
  partial = partial.slice(sign.length).toLowerCase()
  if (ALIASES[partial] && partial === ALIASES[partial]) return []
  const from = value.length - partial.length - sign.length
  return KIND_ORDER.filter(
    kind => !named.has(kind) && kind.startsWith(partial) && (counts[kind] ?? 0) > 0,
  ).map(kind => ({ kind, from, to: value.length, insert: `${sign}${kind}`, count: counts[kind]! }))
}
