import type { Element, ElementContent } from 'hast'
import { h } from 'hastscript'
import type { LatentDoc, LatentKind } from './types'
import { buildPathTree, matchPaths, type PathTree } from '../path-query'
import { mark } from './markup'
import { titleRuns, titleText } from './title'

/**
 * `goto:` in the latent search box: jump to a document by its title, slug or path, without an
 * embedding query. A needle with a slash goes through the Ctrl-O path matcher (path-query.ts);
 * anything else ranks titles and slug basenames, all words prefix-matched.
 */

const PREFIX = /^goto:\s*/i

/** The needle after `goto:`, or null when the box holds something else. */
export function parseGoto(raw: string): string | null {
  const value = raw.trimStart()
  const m = PREFIX.exec(value)
  return m ? value.slice(m[0].length).trim() : null
}

/** Lowercase, `-`/`_` to spaces, so `Arrow's-impossibility-theorem` reads as its title. */
function fold(text: string): string {
  return text
    .toLowerCase()
    .normalize('NFKD')
    .replace(/[̀-ͯ]/g, '')
    .replace(/[-_]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

const basename = (id: string) => id.slice(id.lastIndexOf('/') + 1).replace(/\.[a-z0-9]{2,5}$/i, '')

const KIND_RANK: Record<LatentKind, number> = { note: 0, pdf: 1, base: 2, image: 3 }

export interface GotoIndex {
  docs: LatentDoc[]
  title: string[]
  name: string[]
  id: string[]
  words: string[][]
  tree: PathTree | null
  byId: Map<string, LatentDoc>
}

export function gotoIndex(docs: LatentDoc[]): GotoIndex {
  const title = docs.map(d => fold(titleText(d.t)))
  const name = docs.map(d => fold(basename(d.id)))
  return {
    docs,
    title,
    name,
    id: docs.map(d => d.id.toLowerCase()),
    words: docs.map((_, i) => `${title[i]} ${name[i]}`.split(/[^a-z0-9]+/).filter(Boolean)),
    tree: null,
    byId: new Map(docs.map(d => [d.id, d])),
  }
}

/**
 * A tight subsequence: the needle's letters in order inside a window half again its length
 * (two letters of slack for a short one), so `lbsg` finds "lebesgue" and a long title that
 * merely holds the letters does not.
 */
function subsequence(haystack: string, needle: string): boolean {
  const most = Math.max(needle.length + 2, Math.ceil(needle.length * 1.5))
  for (let start = haystack.indexOf(needle[0]); start !== -1; ) {
    let at = 0
    let i = start
    for (; i < haystack.length && i - start < most; i++)
      if (haystack[i] === needle[at] && ++at === needle.length) return true
    start = haystack.indexOf(needle[0], start + 1)
  }
  return false
}

/**
 * 0 exact title, slug or path · 1 title prefix · 2 slug prefix · 3 every word starts a word of
 * the title or slug · 4 substring of title or path · 5 subsequence (four letters or more).
 */
function tier(index: GotoIndex, i: number, needle: string, words: string[]): number {
  const title = index.title[i]
  const name = index.name[i]
  if (title === needle || name === needle || index.id[i] === needle) return 0
  if (title.startsWith(needle)) return 1
  if (name.startsWith(needle)) return 2
  if (words.every(w => index.words[i].some(have => have.startsWith(w)))) return 3
  if (title.includes(needle) || index.id[i].includes(needle)) return 4
  const tight = needle.replace(/\s/g, '')
  if (tight.length >= 4 && (subsequence(title, tight) || subsequence(name, tight))) return 5
  return -1
}

function byKindThenLength(a: LatentDoc, b: LatentDoc): number {
  return KIND_RANK[a.s] - KIND_RANK[b.s] || a.t.length - b.t.length || a.d - b.d
}

export function matchDocs(index: GotoIndex, raw: string, limit = 60): LatentDoc[] {
  const typed = raw.trim()
  if (!typed) return []
  if (typed.includes('/')) {
    index.tree ??= buildPathTree(index.docs.map(d => ({ slug: d.id, title: d.t })))
    const { exact, items } = matchPaths(index.tree, typed.replace(/^\/+/, ''), limit)
    const out: LatentDoc[] = []
    for (const entry of exact ? [exact, ...items] : items) {
      const doc = entry.slug === undefined ? undefined : index.byId.get(entry.slug)
      if (doc && !out.includes(doc)) out.push(doc)
      if (out.length >= limit) break
    }
    if (out.length) return out
  }
  const needle = fold(typed)
  if (!needle) return []
  const words = needle.split(/[^a-z0-9]+/).filter(Boolean)
  const hits: { doc: LatentDoc; tier: number }[] = []
  for (let i = 0; i < index.docs.length; i++) {
    const t = tier(index, i, needle, words)
    if (t !== -1) hits.push({ doc: index.docs[i], tier: t })
  }
  hits.sort((a, b) => a.tier - b.tier || byKindThenLength(a.doc, b.doc))
  return hits.slice(0, limit).map(hit => hit.doc)
}

/** What Tab writes into the box for a document. */
export function completeGoto(doc: LatentDoc): string {
  return `goto:${doc.id}`
}

/** The links pane while `goto:` is typed: one ranked list, the first row active. */
export function gotoPane(
  docs: LatentDoc[],
  needle: string,
  total: number,
  active: number,
  outOfScope: (doc: LatentDoc) => boolean,
): ElementContent[] {
  const head = h('header.lat-pane-head', [
    h('span', 'goto'),
    h('span.lat-count', needle ? `${docs.length} of ${total} documents` : `${total} documents`),
  ])
  if (!needle)
    return [head, h('p.lat-goto-empty', 'a title, slug or path; ↑↓ to pick, ⏎ to open, ⇥ to fill')]
  if (docs.length === 0) return [head, h('p.lat-goto-empty', `no document matches “${needle}”`)]
  return [
    head,
    h(
      'ol.lat-list.lat-goto',
      docs.map((doc, i): Element => {
        const year = doc.date?.slice(0, 4) ?? null
        const size =
          doc.s === 'pdf' && doc.pages
            ? `${doc.pages} pages`
            : `${doc.n} ${doc.n === 1 ? 'chunk' : 'chunks'}`
        return h(
          'li.lat-link.lat-goto-item',
          {
            dataGoto: String(doc.d),
            dataActive: i === active ? '' : undefined,
            dataOut: outOfScope(doc) ? '' : undefined,
          },
          [
            h(
              'a.lat-link-title',
              { href: `?focus=${doc.r0}`, dataLatFocus: String(doc.r0) },
              titleRuns(doc.t),
            ),
            mark(doc.s),
            h('span.lat-link-locus', [
              doc.id,
              year ? h('em', String(year)) : null,
              h('em', size),
              outOfScope(doc) ? h('em', 'out of scope') : null,
            ]),
          ],
        )
      }),
    ),
  ]
}
