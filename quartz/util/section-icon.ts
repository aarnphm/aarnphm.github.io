import { TRIATHLON_HOSTNAME } from './triathlon-host'

// Sections whose pages carry their own mascot icon (drawn by quartz/scripts/brand-art.ts into
// quartz/static/icon-<section>.png): the favicon in Head, the icon on generated social cards and
// the outfit the site pet wears there.
export const SECTIONS = [
  'triathlon',
  'fr',
  'posts',
  'legal',
  'art',
  'library',
  'slides',
  'stream',
  'courses',
  'engineering',
  'cinema',
] as const

export type Section = (typeof SECTIONS)[number]

// What a page is, as the server knows it from its file data and the client from <body>'s
// data-slug, data-is-base and data-slides.
export type SectionPage = { slug: string; base?: boolean; slides?: boolean }

const FOLDERS: readonly Section[] = ['triathlon', 'fr', 'posts', 'stream', 'courses']

// Single pages that wear a section's icon.
const PAGES: Record<string, Section> = {
  'terms-of-service': 'legal',
  'privacy-policy': 'legal',
  quotes: 'art',
  vibes: 'art',
}

// Paths that wear a section's icon with everything under them: folders below the top level, a
// folder whose name is not its section's, and a base with its views.
const PREFIXES: Record<string, Section> = {
  'thoughts/university': 'engineering',
  movies: 'cinema',
  cinematheque: 'cinema',
}

// A slide page (`<note>/slides/<n>`) of a note with `slides: true`.
export const isSlidePage = (slug: string, slides: unknown) =>
  slides === true && /\/slides\/\d+$/.test(slug)

export function sectionOf({ slug, base, slides }: SectionPage): Section | null {
  if (slides) return 'slides'
  // A named page or path beats the base rule: the cinematheque is a base, and wears the cinema.
  const prefix = Object.keys(PREFIXES).find(f => slug === f || slug.startsWith(`${f}/`))
  const named = PAGES[slug] ?? (prefix ? PREFIXES[prefix] : undefined)
  if (named) return named
  if (base) return 'library'
  const head = slug.split('/')[0]
  return FOLDERS.find(s => s === head) ?? null
}

export const sectionIconFile = (page: SectionPage) => {
  const section = sectionOf(page)
  return section ? `static/icon-${section}.png` : 'static/icon.png'
}

// The section of the page a link to the site points at, for the icon after the link. A deck link
// (`<note>/slides`, which opens its first slide) counts as a slide. The triathlon microsite is the
// triathlon section; other subdomains are sites of their own.
export function sectionOfLink(
  url: URL,
  apexHostname: string,
  baseSlugs: ReadonlySet<string>,
): Section | null {
  const host = url.hostname.replace(/^www\./, '')
  if (host === TRIATHLON_HOSTNAME) return 'triathlon'
  if (host !== apexHostname.replace(/^www\./, '')) return null
  const slug = url.pathname.replace(/^\/+|\/+$/g, '').replace(/\.html$/, '')
  return sectionOf({ slug, base: baseSlugs.has(slug), slides: /\/slides(?:\/\d+)?$/.test(slug) })
}
