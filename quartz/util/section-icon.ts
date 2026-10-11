// Sections whose pages carry their own mascot icon (drawn by quartz/scripts/brand-art.ts into
// quartz/static/icon-<section>.png): the favicon in Head and the icon on generated social cards.
export const SECTIONS = ['triathlon', 'fr', 'posts'] as const

export type Section = (typeof SECTIONS)[number]

export function sectionOf(slug: string): Section | null {
  const head = slug.split('/')[0]
  return SECTIONS.find(s => s === head) ?? null
}

export const sectionIconFile = (slug: string) => {
  const section = sectionOf(slug)
  return section ? `static/icon-${section}.png` : 'static/icon.png'
}
