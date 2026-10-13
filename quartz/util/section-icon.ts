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

// Folders below the top level that wear a section's icon, with everything under them.
const SUBFOLDERS: Record<string, Section> = { 'thoughts/university': 'engineering' }

// A slide page (`<note>/slides/<n>`) of a note with `slides: true`.
export const isSlidePage = (slug: string, slides: unknown) =>
  slides === true && /\/slides\/\d+$/.test(slug)

export function sectionOf({ slug, base, slides }: SectionPage): Section | null {
  if (slides) return 'slides'
  if (base) return 'library'
  const head = slug.split('/')[0]
  const sub = Object.keys(SUBFOLDERS).find(f => slug === f || slug.startsWith(`${f}/`))
  return PAGES[slug] ?? (sub ? SUBFOLDERS[sub] : FOLDERS.find(s => s === head)) ?? null
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
