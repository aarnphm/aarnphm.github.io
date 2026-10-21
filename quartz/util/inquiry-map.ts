import GithubSlugger from 'github-slugger'

/**
 * Inquiry map: a page declares `inquiries` in its frontmatter, either a Mermaid-style edge list
 * or `{ computed: <slug prefix>, edges: <edge list> }`. Edges join file slugs (`slug#anchor` for a
 * section); the builder resolves each id to a page, its title and its cards.
 *
 * ```
 * graph LR
 *   %% comment
 *   fr/episode-13 --> fr/mots#le-corps
 *   a1["a1"]:::mvt --> fr/mots#le-cours & fr/mots#la-meteo
 *   fr/mots --- fr/verbs
 *   fr/phrases:::theme
 * ```
 */

/** `theme` boxes (sized by cards), `mvt` capsules (hubs), `pg` solid capsules (pages). */
export type InquiryClass = 'theme' | 'mvt' | 'pg'
export type InquiryLinkKind = 'edge' | 'seealso' | 'card' | 'wikilink'

export interface InquiryNode {
  id: string
  label: string
  cls: InquiryClass
  /** what the node is, to place same-kind nodes together: episode, lesson, section, … */
  kind: string
  /** cards behind the node, 0 when it has none */
  w: number
  /** slug with optional `#anchor`, no leading slash; empty when the id names no page */
  href: string
}

export interface InquiryMapData {
  nodes: InquiryNode[]
  /** `[source index, target index, kind]` */
  links: [number, number, InquiryLinkKind][]
}

export interface InquirySpec {
  edges?: string
  /** slug prefix whose seealso, wikilinks and card citations become links, e.g. `fr/` */
  computed?: string
}

export interface InquiryPage {
  slug: string
  title: string
  /** markdown source with frontmatter */
  source: string
}

export interface InquiryDeck {
  /** slug of the page the deck belongs to */
  slug: string
  /** `N:` text of each card, `undefined` when it has none */
  notes: (string | undefined)[]
  /** what the card asks for, back first, to find the section a card was cut from */
  probes: string[]
}

/** Reads the frontmatter value; anything but a string or `{ edges, computed }` declares nothing. */
export function inquirySpecOf(value: unknown): InquirySpec | undefined {
  if (typeof value === 'string') return value.trim() ? { edges: value } : undefined
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return undefined
  const { edges, computed } = value as Record<string, unknown>
  const spec: InquirySpec = {
    edges: typeof edges === 'string' ? edges : undefined,
    computed: typeof computed === 'string' && computed.trim() ? computed.trim() : undefined,
  }
  return spec.edges || spec.computed ? spec : undefined
}

/** Where the emitter writes a declaring page's graph and where the browser fetches it. */
export function inquiriesSlug(slug: string): string {
  return `${slug.replace(/(^|\/)index$/, '')}/inquiries`.replace(/^\//, '')
}

// ---- mermaid-style edges

export interface ParsedEdges {
  decls: Map<string, { label?: string; cls?: InquiryClass }>
  edges: [string, string][]
}

const ARROW = /(?:<-->|-->|==>|-\.->|---)(?:\|[^|]*\|)?/
const CLASSES = new Set<string>(['theme', 'mvt', 'pg'])

/** Splits at top-level occurrences of `sep`; brackets and quotes protect their contents. */
function splitTop(text: string, sep: RegExp): string[] {
  const out: string[] = []
  let depth = 0
  let quoted = false
  let start = 0
  for (let i = 0; i < text.length; i++) {
    const ch = text[i]
    if (ch === '"') quoted = !quoted
    else if (!quoted && ch === '[') depth++
    else if (!quoted && ch === ']') depth--
    else if (!quoted && depth === 0) {
      const m = sep.exec(text.slice(i))
      if (m && m.index === 0 && m[0].length > 0) {
        out.push(text.slice(start, i))
        i += m[0].length - 1
        start = i + 1
      }
    }
  }
  out.push(text.slice(start))
  return out
}

const TOKEN = /^(\S+?)(\[.*\])?(?::::(\w+))?$/

export function parseEdges(text: string | undefined): ParsedEdges {
  const decls: ParsedEdges['decls'] = new Map()
  const edges: [string, string][] = []
  const token = (raw: string): string | undefined => {
    const m = TOKEN.exec(raw.trim())
    if (!m) return undefined
    const [, id, bracket, cls] = m
    const label = bracket
      ?.slice(1, -1)
      .trim()
      .replace(/^"(.*)"$/, '$1')
    const known = decls.get(id) ?? {}
    if (label) known.label = label
    if (cls && CLASSES.has(cls)) known.cls = cls as InquiryClass
    decls.set(id, known)
    return id
  }
  for (const rawLine of (text ?? '').split(/\r?\n/)) {
    const line = rawLine.replace(/%%.*$/, '').trim()
    if (!line || /^(graph|flowchart)\b/.test(line)) continue
    const groups = splitTop(line, new RegExp(`^\\s*${ARROW.source}\\s*`)).map(group =>
      splitTop(group, /^\s*&\s*/).flatMap(raw => (token(raw) ? [token(raw)!] : [])),
    )
    for (let g = 0; g + 1 < groups.length; g++) {
      for (const a of groups[g]) for (const b of groups[g + 1]) edges.push([a, b])
    }
  }
  return { decls, edges }
}

// ---- sections and cards

const flat = (text: string): string =>
  text
    .replace(/\{\{|\}\}|`|\*\*/g, '')
    .replace(/\s+/g, ' ')
    .trim()
    .toLowerCase()

const anchorOf = (text: string): string => new GithubSlugger().slug(text.trim())

const splitFrontmatter = (source: string): { front: string; body: string } => {
  const match = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?/.exec(source)
  return match
    ? { front: match[1], body: source.slice(match[0].length) }
    : { front: '', body: source }
}

const WIKILINK = /\[\[([^\]|]+)(?:\|([^\]]*))?\]\]/g

interface Target {
  slug: string
  anchor?: string
  alias?: string
}

/** Wikilinks of `text`, resolved against the page they sit on. */
const targetsIn = (text: string, from: string): Target[] => {
  const out: Target[] = []
  for (const match of text.matchAll(WIKILINK)) {
    const [pathPart, ...rest] = match[1].split('#')
    const anchor = rest.length > 0 ? anchorOf(rest.join('#')) : undefined
    const stem = pathPart.trim().replace(/^\/+/, '').replace(/\.md$/, '')
    out.push({ slug: stem === '' ? from : stem, anchor, alias: match[2]?.trim() || undefined })
  }
  return out
}

interface Sections {
  /** `##` anchors in page order, with their heading text */
  order: { anchor: string; heading: string }[]
  /** every heading anchor to the `##` anchor that holds it */
  parent: Map<string, string>
  /** heading text of every heading, by anchor */
  headings: Map<string, string>
  /** cards per `##` anchor */
  cards: Map<string, number>
}

/**
 * The `##` sections of a page and how many cards of its deck each holds: a card that cites
 * `slug#anchor` counts for that section; the rest go where their answer sits in the text, and
 * stay in the section before when the text does not place them.
 */
function sectionsOf(page: InquiryPage, deck: InquiryDeck | undefined): Sections {
  const order: Sections['order'] = []
  const parent = new Map<string, string>()
  const text = new Map<string, string[]>()
  const headings = new Map<string, string>()
  let current = ''
  for (const line of splitFrontmatter(page.source).body.split(/\r?\n/)) {
    const h2 = /^## (.+?)\s*$/.exec(line)
    const deeper = /^#{3,6} (.+?)\s*$/.exec(line)
    if (h2) {
      current = anchorOf(h2[1])
      headings.set(current, h2[1])
      order.push({ anchor: current, heading: h2[1] })
      parent.set(current, current)
      text.set(current, [])
    } else {
      if (deeper) headings.set(anchorOf(deeper[1]), deeper[1])
      if (deeper && current) parent.set(anchorOf(deeper[1]), current)
      if (current) text.get(current)?.push(line)
    }
  }
  const cards = new Map<string, number>()
  if (deck && order.length > 0) {
    const flatText = order.map(s => flat(text.get(s.anchor)!.join(' ')))
    let cursor = 0
    deck.probes.forEach((probe, n) => {
      const cited = deck.notes[n]
        ? targetsIn(deck.notes[n]!, deck.slug).find(t => t.slug === page.slug && t.anchor)
        : undefined
      const named = cited?.anchor ? parent.get(cited.anchor) : undefined
      if (named) {
        cursor = order.findIndex(s => s.anchor === named)
      } else {
        const key = flat(probe)
        let at = -1
        if (key.length > 3) {
          flatText.forEach((t, k) => {
            if (t.includes(key) && (at < 0 || Math.abs(k - cursor) < Math.abs(at - cursor))) at = k
          })
        }
        if (at >= 0) cursor = at
      }
      const anchor = order[cursor]?.anchor
      if (anchor) cards.set(anchor, (cards.get(anchor) ?? 0) + 1)
    })
  }
  return { order, parent, headings, cards }
}

// ---- computed layer (french notes)

const FR_MOTS = 'fr/mots'
const FR_REPERES: ReadonlySet<string> = new Set([
  'fr/phrases',
  'fr/les-nombres',
  'fr/orthography-phonetics',
  'fr/verbs',
  'fr/subjonctif',
  'fr/relatifs',
  'fr/temps-passes',
  'fr/registre-soutenu',
  'fr/quebec-france',
  'fr/discours-rapporte',
])
const MONTHS = [
  'janv.',
  'févr.',
  'mars',
  'avr.',
  'mai',
  'juin',
  'juil.',
  'août',
  'sept.',
  'oct.',
  'nov.',
  'déc.',
]
const PRIORITY: Record<InquiryLinkKind, number> = { edge: 5, seealso: 4, card: 3, wikilink: 2 }

/** `yymmdd` slug stem to `20 oct`; anything else comes back unchanged. */
const dateLabel = (stem: string): string => {
  const m = /^(\d{2})(\d{2})(\d{2})$/.exec(stem)
  const month = m ? MONTHS[Number(m[2]) - 1] : undefined
  return m && month ? `${Number(m[3])} ${month}` : stem
}

class Graph {
  nodes: InquiryNode[] = []
  index = new Map<string, number>()
  private linkSet = new Map<string, [number, number, InquiryLinkKind]>()

  add(node: InquiryNode): number {
    const known = this.index.get(node.id)
    if (known !== undefined) return known
    this.index.set(node.id, this.nodes.length)
    this.nodes.push(node)
    return this.nodes.length - 1
  }

  link(a: number | undefined, b: number | undefined, kind: InquiryLinkKind) {
    if (a === undefined || b === undefined || a === b) return
    const key = a < b ? `${a}|${b}` : `${b}|${a}`
    const known = this.linkSet.get(key)
    if (known && PRIORITY[known[2]] >= PRIORITY[kind]) return
    this.linkSet.set(key, [a, b, kind])
  }

  links(): [number, number, InquiryLinkKind][] {
    return [...this.linkSet.values()]
  }
}

/** Seealso, wikilinks and card citations among the pages under `prefix`. */
function computeLayer(
  g: Graph,
  prefix: string,
  pages: Map<string, InquiryPage>,
  decks: Map<string, InquiryDeck>,
) {
  const french = prefix.startsWith('fr/')
  const members = [...pages.values()].filter(
    p =>
      p.slug.startsWith(prefix) && !p.slug.endsWith('/flashcards') && !/(^|\/)index$/.test(p.slug),
  )
  const sourceOf = new Map<string, InquiryPage>()
  for (const page of members) {
    sourceOf.set(page.slug, page)
    const stem = page.slug.split('/').pop() ?? page.slug
    const cards = decks.get(page.slug)?.notes.length ?? 0
    const node = (label: string, cls: InquiryClass, kind: string, w = cards): InquiryNode => ({
      id: page.slug,
      label,
      cls,
      kind,
      w,
      href: page.slug,
    })
    const episode = french ? /^episode-(\d+)$/.exec(stem) : null
    if (french && page.slug === FR_MOTS) continue
    if (episode) {
      const label = page.title.replace(/^[^:]*:\s*/, '') || page.title
      g.add(node(`ép. ${Number(episode[1])} · ${label}`, 'pg', 'episode'))
    } else if (french && page.slug.startsWith('fr/lessons/')) {
      g.add(node(dateLabel(stem), 'pg', 'lesson'))
    } else if (french && page.slug.startsWith('fr/practices/')) {
      g.add(node(`pratique ${dateLabel(stem)}`, 'pg', 'practice'))
    } else if (french && FR_REPERES.has(page.slug)) {
      g.add(node(page.title, 'theme', 'repere'))
    } else {
      g.add(node(page.title, 'pg', 'note'))
    }
  }

  // sections of the page that holds the vocabulary bank
  const motsPage = french ? pages.get(FR_MOTS) : undefined
  const motsSections = motsPage ? sectionsOf(motsPage, decks.get(FR_MOTS)) : undefined
  if (motsPage && motsSections) {
    sourceOf.set(FR_MOTS, motsPage)
    for (const s of motsSections.order) {
      if (s.anchor === anchorOf('parcours A1 → C1')) continue
      g.add({
        id: `${FR_MOTS}#${s.anchor}`,
        label: s.heading,
        cls: 'theme',
        kind: 'section',
        w: motsSections.cards.get(s.anchor) ?? 0,
        href: `${FR_MOTS}#${s.anchor}`,
      })
    }
  }

  const nodeFor = (t: Target): number | undefined => {
    if (t.slug === FR_MOTS && motsSections) {
      const section = t.anchor ? motsSections.parent.get(t.anchor) : undefined
      return section ? g.index.get(`${FR_MOTS}#${section}`) : undefined
    }
    return g.index.get(t.slug)
  }

  for (const deck of decks.values()) {
    if (!deck.slug.startsWith(prefix) || deck.slug === FR_MOTS) continue
    const own = g.index.get(deck.slug)
    for (const note of deck.notes) {
      if (!note) continue
      for (const t of targetsIn(note, deck.slug)) g.link(own, nodeFor(t), 'card')
    }
  }

  const aliases = new Map<string, Map<string, number>>()
  const pdfNode = (slug: string, alias?: string): number => {
    const named = aliases.get(slug) ?? new Map<string, number>()
    if (alias) named.set(alias, (named.get(alias) ?? 0) + 1)
    aliases.set(slug, named)
    const file = slug.split('/').pop() ?? slug
    return g.add({
      id: slug,
      label: file.replace(/\.pdf$/, ''),
      cls: 'pg',
      kind: 'book',
      w: 0,
      href: slug,
    })
  }

  for (const page of sourceOf.values()) {
    const { front, body } = splitFrontmatter(page.source)
    const pageNode = g.index.get(page.slug)
    const scan = (text: string, kind: InquiryLinkKind, source: number | undefined) => {
      for (const t of targetsIn(text, page.slug)) {
        if (!t.slug.startsWith(prefix)) continue
        g.link(source, t.slug.endsWith('.pdf') ? pdfNode(t.slug, t.alias) : nodeFor(t), kind)
      }
    }
    scan(front, 'seealso', pageNode)
    if (page.slug !== FR_MOTS || !motsSections) {
      scan(body, 'wikilink', pageNode)
      continue
    }
    // a link in the vocabulary bank belongs to the section it sits in
    let section: number | undefined
    let skip = false
    const buffer: string[] = []
    const flush = () => {
      if (!skip) scan(buffer.join('\n'), 'wikilink', section)
      buffer.length = 0
    }
    for (const line of body.split(/\r?\n/)) {
      const h2 = /^## (.+?)\s*$/.exec(line)
      if (h2) {
        flush()
        skip = anchorOf(h2[1]) === anchorOf('parcours A1 → C1')
        section = g.index.get(`${FR_MOTS}#${anchorOf(h2[1])}`)
      } else buffer.push(line)
    }
    flush()
  }

  // books are named by how the notes call them
  for (const [slug, named] of aliases) {
    const best = [...named.entries()].sort((a, b) => b[1] - a[1])[0]
    if (best) g.nodes[g.index.get(slug)!].label = best[0]
  }
}

/** Builds the graph a page declares: its edges, plus the computed layer when it asks for one. */
export function buildInquiryMap(
  spec: InquirySpec,
  pageList: InquiryPage[],
  deckList: InquiryDeck[],
): InquiryMapData {
  const pages = new Map(pageList.map(p => [p.slug, p]))
  const decks = new Map(deckList.map(d => [d.slug, d]))
  const g = new Graph()
  const computed = new Set<string>()
  if (spec.computed) {
    computeLayer(g, spec.computed, pages, decks)
    g.nodes.forEach(n => computed.add(n.id))
  }
  const linkedByLayer = new Set<number>()
  const computedLinks = g.links()
  for (const [a, b] of computedLinks) {
    linkedByLayer.add(a)
    linkedByLayer.add(b)
  }

  const { decls, edges } = parseEdges(spec.edges)
  const sections = new Map<string, Sections>()
  const sectionsFor = (slug: string): Sections | undefined => {
    const page = pages.get(slug)
    if (!page) return undefined
    const known = sections.get(slug) ?? sectionsOf(page, decks.get(slug))
    sections.set(slug, known)
    return known
  }
  const resolve = (id: string): number => {
    const decl = decls.get(id)
    const known = g.index.get(id)
    if (known !== undefined) {
      const node = g.nodes[known]
      if (decl?.label) node.label = decl.label
      if (decl?.cls) node.cls = decl.cls
      return known
    }
    const [slug, ...rest] = id.split('#')
    const anchor = rest.length > 0 ? anchorOf(rest.join('#')) : undefined
    const page = pages.get(slug)
    if (!page) {
      return g.add({
        id,
        label: decl?.label ?? id,
        cls: decl?.cls ?? 'mvt',
        kind: 'label',
        w: 0,
        href: '',
      })
    }
    if (!anchor) {
      return g.add({
        id,
        label: decl?.label ?? page.title,
        cls: decl?.cls ?? 'pg',
        kind: 'page',
        w: decks.get(slug)?.notes.length ?? 0,
        href: slug,
      })
    }
    const held = sectionsFor(slug)
    const owner = held?.parent.get(anchor) ?? anchor
    const heading = held?.headings.get(anchor)
    return g.add({
      id,
      label: decl?.label ?? heading ?? anchor,
      cls: decl?.cls ?? 'theme',
      kind: 'section',
      w: held?.cards.get(owner) ?? 0,
      href: id,
    })
  }
  // computed nodes that the page also names by hand take part in `edges`
  for (const id of decls.keys()) resolve(id)

  const handLinks: [number, number][] = edges.map(([a, b]) => [resolve(a), resolve(b)])
  for (const [a, b] of handLinks) g.link(a, b, 'edge')

  // computed nodes that touch nothing and hold no cards have nowhere to sit
  const linked = new Set<number>(handLinks.flat())
  for (const [a, b] of g.links()) {
    linked.add(a)
    linked.add(b)
  }
  const keepAt = g.nodes.map((n, i) => linked.has(i) || n.w > 0 || !computed.has(n.id))
  const remap = new Map<number, number>()
  g.nodes.forEach((_, i) => keepAt[i] && remap.set(i, remap.size))

  // a hub's weight is the cards of what it joins
  const nodes = g.nodes.filter((_, i) => keepAt[i]).map(n => ({ ...n }))
  const links = g
    .links()
    .map(([a, b, k]) => [remap.get(a)!, remap.get(b)!, k] as [number, number, InquiryLinkKind])
  const base = nodes.map(n => n.w)
  for (const [a, b] of links) {
    for (const [m, c] of [
      [a, b],
      [b, a],
    ]) {
      if (nodes[m].cls === 'mvt' && nodes[c].cls !== 'mvt') nodes[m].w += base[c]
    }
  }
  return { nodes, links }
}
