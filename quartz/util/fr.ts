import { QuartzPluginData } from '../plugins/vfile'
import { fileBySlug } from './course'
import { flashcardsSlug } from './flashcards-path'
import { ReloadableWeakMap } from './module-caches'
import { FullSlug } from './path'

/** The French folder as one course: sixteen episodes in four seasons, dated lessons, shelves. */
export const FR_DIR = 'fr'
export const FR_SEASONS = ['automne', 'hiver', 'printemps', 'été'] as const
export const FR_EPISODES = 16

const MONTHS = [
  'janv',
  'févr',
  'mars',
  'avr',
  'mai',
  'juin',
  'juil',
  'août',
  'sept',
  'oct',
  'nov',
  'déc',
]

const episodeRe = /^fr\/episode-(\d+)$/
const datedRe = /^fr\/(lessons|practices)\/(\d{2})(\d{2})(\d{2})$/
const refRe = /fr\/episode-(\d+)/g

export interface FrDated {
  date: string
  slug: FullSlug
  kind: 'lessons' | 'practices'
  episodes: number[]
  cards: number
}

export interface FrEpisode {
  n: number
  title: string
  season: (typeof FR_SEASONS)[number]
  slug: FullSlug
  note?: QuartzPluginData
  deck?: QuartzPluginData
  cardIds: string[]
  lessons: FrDated[]
  /** The last episode a lesson has reached. */
  frontier: boolean
}

export interface FrShelfItem {
  title: string
  slug: FullSlug
  cards: number
}

export interface FrView {
  episodes: FrEpisode[]
  seasons: { name: (typeof FR_SEASONS)[number]; episodes: FrEpisode[] }[]
  lessons: FrDated[]
  practices: FrDated[]
  /** Top-level notes with a deck, besides the episodes and the word bank. */
  reperes: FrShelfItem[]
  mots?: FrShelfItem
  /** Every deck under `fr/`, in drill order. */
  decks: { note: QuartzPluginData; deck: QuartzPluginData; label: string; read: boolean }[]
  notes: number
  cards: number
  frontier: number
}

export const seasonOf = (n: number) => FR_SEASONS[Math.min(3, Math.ceil(n / 4) - 1)]

/** `[[fr/episode-13|…]]` links in a note's `seealso`. */
function episodesOf(file: QuartzPluginData): number[] {
  const see = file.frontmatter?.seealso
  const text = Array.isArray(see) ? see.join(' ') : typeof see === 'string' ? see : ''
  return [...text.matchAll(refRe)].map(match => Number(match[1])).sort((a, b) => a - b)
}

/** `6, 13 oct` for dates in one month, `29 sept, 6 oct` across months. */
export function dateWords(dates: string[]): string {
  const parts = dates.map(date => ({ day: Number(date.slice(8)), month: Number(date.slice(5, 7)) }))
  return parts
    .map((part, i) =>
      parts[i + 1]?.month === part.month ? `${part.day}` : `${part.day} ${MONTHS[part.month - 1]}`,
    )
    .join(', ')
}

const deckMaps = new ReloadableWeakMap<QuartzPluginData[], Map<string, QuartzPluginData>>()
const views = new ReloadableWeakMap<QuartzPluginData[], WeakMap<QuartzPluginData[], FrView>>()

function deckMap(decks: QuartzPluginData[]) {
  let map = deckMaps.get(decks)
  if (!map) {
    map = new Map(
      decks.filter(deck => deck.flashcards).map(deck => [deck.flashcards!.sourceSlug, deck]),
    )
    deckMaps.set(decks, map)
  }
  return map
}

const cardIdsOf = (deck: QuartzPluginData | undefined) =>
  deck?.flashcards?.cards.map(card => card.id) ?? []

/**
 * Joins `fr/` with its decks over the course helpers. Episodes are the units, a season is four of
 * them, and a lesson belongs to the episodes its `seealso` names.
 */
export function frView(allFiles: QuartzPluginData[], decks: QuartzPluginData[] = []): FrView {
  let byDecks = views.get(allFiles)
  if (!byDecks) {
    byDecks = new WeakMap()
    views.set(allFiles, byDecks)
  }
  const cached = byDecks.get(decks)
  if (cached) return cached

  const deckOf = deckMap(decks)
  const dated: FrDated[] = []
  for (const file of allFiles) {
    const match = datedRe.exec(file.slug ?? '')
    if (!match) continue
    dated.push({
      date: `20${match[2]}-${match[3]}-${match[4]}`,
      slug: file.slug as FullSlug,
      kind: match[1] as FrDated['kind'],
      episodes: episodesOf(file),
      cards: cardIdsOf(deckOf.get(file.slug!)).length,
    })
  }
  dated.sort((a, b) => a.date.localeCompare(b.date))
  const lessons = dated.filter(item => item.kind === 'lessons')
  const practices = dated.filter(item => item.kind === 'practices')
  const frontier = Math.max(0, ...lessons.flatMap(lesson => lesson.episodes))

  const episodes: FrEpisode[] = []
  for (let n = 1; n <= FR_EPISODES; n++) {
    const slug = `fr/episode-${n}` as FullSlug
    const note = fileBySlug(slug, allFiles)
    const deck = deckOf.get(slug)
    const title = String(note?.frontmatter?.title ?? `épisode ${n}`).replace(/^épisode \d+ : /, '')
    episodes.push({
      n,
      title,
      season: seasonOf(n),
      slug,
      ...(note ? { note } : {}),
      ...(deck ? { deck } : {}),
      cardIds: cardIdsOf(deck),
      lessons: lessons.filter(lesson => lesson.episodes.includes(n)),
      frontier: n === frontier,
    })
  }

  const shelf: FrShelfItem[] = []
  for (const file of allFiles) {
    const slug = file.slug ?? ''
    if (!slug.startsWith('fr/') || slug.indexOf('/', 3) !== -1 || episodeRe.test(slug)) continue
    const deck = deckOf.get(slug)
    if (!deck) continue
    shelf.push({
      title: String(file.frontmatter?.title ?? slug.slice(3)),
      slug: slug as FullSlug,
      cards: cardIdsOf(deck).length,
    })
  }
  shelf.sort((a, b) => a.title.localeCompare(b.title, 'fr'))
  const mots = shelf.find(item => item.slug === 'fr/mots')

  const ordered: FrView['decks'] = []
  const push = (slug: string, label: string, read: boolean) => {
    const note = fileBySlug(slug, allFiles)
    const deck = deckOf.get(slug)
    if (note && deck) ordered.push({ note, deck, label, read })
  }
  for (const episode of episodes) push(episode.slug, `ép. ${episode.n}`, episode.n <= frontier)
  for (const lesson of lessons) push(lesson.slug, lesson.date.slice(2).replaceAll('-', ''), true)
  for (const item of shelf) push(item.slug, item.title, true)

  const view: FrView = {
    episodes,
    seasons: FR_SEASONS.map(name => ({ name, episodes: episodes.filter(e => e.season === name) })),
    lessons,
    practices,
    reperes: shelf.filter(item => item.slug !== 'fr/mots'),
    ...(mots ? { mots } : {}),
    decks: ordered,
    notes: allFiles.filter(file => (file.slug ?? '').startsWith('fr/') && !file.flashcards).length,
    cards: ordered.reduce((sum, item) => sum + cardIdsOf(item.deck).length, 0),
    frontier,
  }
  byDecks.set(decks, view)
  return view
}

/** True for the pages the unit bar sits on: an episode note. */
export const frEpisodeOf = (slug: string | undefined): number | null => {
  const match = episodeRe.exec(slug ?? '')
  const n = match ? Number(match[1]) : 0
  return n >= 1 && n <= FR_EPISODES ? n : null
}

/** The merged drill's deck list for `fr/`: the shape `unitManifest` gives course decks. */
export function frManifest(view: FrView) {
  return view.decks.map((item, i) => ({
    deck: flashcardsSlug(item.note.slug!),
    course: FR_DIR,
    n: i + 1,
    label: item.label,
    read: item.read,
    ids: cardIdsOf(item.deck),
    groups: Object.fromEntries(
      (item.deck.flashcards!.cards ?? [])
        .filter(card => card.groupId)
        .map(card => [card.id, card.groupId!]),
    ),
  }))
}

/** `13/16`, the episode's place among the sixteen. */
export const episodeLabel = (n: number) => `${n}/${FR_EPISODES}`

/** French grouping: 3 440, not the course pages' 3,440. */
export const frCount = (n: number) => n.toLocaleString('fr-CA')

export const cardsWord = (n: number) => `${frCount(n)} ${n === 1 ? 'carte' : 'cartes'}`

const LEVELS = ['A1', 'A2', 'B1', 'B2', 'C1'] as const

/**
 * The word bank's `parcours A1 → C1` section: each level and the sections it lists. Read from the
 * note's plain text, where a wikilink is its label and a list item ends at a newline. A text that
 * runs on past a level (no newline) gives no levels rather than wrong counts.
 */
export function motsLevels(mots: QuartzPluginData | undefined) {
  const text = mots?.text ?? ''
  const from = text.indexOf('parcours A1')
  if (from < 0) return []
  const body = text.slice(from)
  const marks = [...body.matchAll(new RegExp(`(${LEVELS.join('|')})\\s*:\\s*`, 'g'))]
  const levels = marks.map((mark, i) => {
    const start = mark.index + mark[0].length
    const stop = Math.min(marks[i + 1]?.index ?? body.length, body.indexOf('\n', start) >>> 0)
    const themes = body
      .slice(start, stop)
      .split(/\s*,\s*/)
      .map(name => name.trim())
      .filter(Boolean)
    return { level: mark[1], themes }
  })
  return levels.some(item => item.themes.length > 12) ? [] : levels
}
