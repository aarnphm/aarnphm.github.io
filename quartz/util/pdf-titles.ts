import type { QuartzPluginData } from '../plugins/vfile'
import { sluggify } from './path'
import { normalizePdfSlug } from './pdf-marks'

/** Folder indexes name PDFs with a `pdfTitles` map keyed by relative filenames. */
export function pdfTitlesFor(files: QuartzPluginData[]): Map<string, string> {
  const titles = new Map<string, string>()
  for (const file of files) {
    if (!file.slug || !/(^|\/)index$/.test(file.slug)) continue
    const values = file.frontmatter?.pdfTitles
    if (typeof values !== 'object' || values === null || Array.isArray(values)) continue
    const folder = file.slug.slice(0, -'index'.length)
    for (const [filename, value] of Object.entries(values)) {
      if (typeof value !== 'string') continue
      const title = value.normalize('NFC').replace(/\s+/g, ' ').trim()
      const slug = normalizePdfSlug(sluggify(`${folder}${filename}`))
      if (slug && title) titles.set(slug, title)
    }
  }
  return titles
}
