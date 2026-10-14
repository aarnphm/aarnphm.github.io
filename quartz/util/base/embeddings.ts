import type { QuartzPluginData } from '../../plugins/vfile'
import type { FullSlug } from '../path'
import { descriptionToPlainText } from '../description'

/** One row of a base's embedding text; the offline builder embeds `title: {title} | text: {text}`. */
export type BaseEmbeddingRecord = { slug: string; title: string; text: string }

/** Site path of a base's vector manifest; the builder writes it under content/embeddings/bases. */
export const baseEmbeddingsDir = 'embeddings/bases'
export const baseEmbeddingsTextSlug = (baseSlug: FullSlug): FullSlug =>
  `${baseSlug}.embeddings-text` as FullSlug
export const baseEmbeddingsManifestPath = (baseSlug: FullSlug): string =>
  `/${baseEmbeddingsDir}/${baseSlug}/manifest.json`

// Frontmatter keys that carry no meaning for a search query (identity, timestamps, media, geo).
const SKIPPED_KEYS = new Set([
  'aliases',
  'coordinates',
  'created',
  'cssclasses',
  'date',
  'description',
  'draft',
  'id',
  'modified',
  'pageLayout',
  'posters',
  'published',
  'semantic',
  'title',
  'url',
])
const BODY_LIMIT = 1500

const unwrapLink = (value: string) => value.replace(/^\[\[(.*?)(?:\|.*?)?\]\]$/, '$1')

function propertyText(value: unknown): string | null {
  if (typeof value === 'string') return value.trim() ? unwrapLink(value.trim()) : null
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  if (Array.isArray(value)) {
    const parts = value.map(propertyText).filter((part): part is string => part !== null)
    return parts.length > 0 ? parts.join(', ') : null
  }
  return null
}

/** Builds an embedding row for an unprotected base member that has text and has not opted out. */
export function baseEmbeddingRecord(file: QuartzPluginData): BaseEmbeddingRecord | null {
  const slug = file.slug
  if (!slug || file.frontmatter?.protected || file.frontmatter?.semantic === false) return null
  const frontmatter: Record<string, unknown> = file.frontmatter ?? {}
  const title =
    typeof frontmatter.title === 'string' && frontmatter.title.length > 0
      ? frontmatter.title
      : (slug.split('/').pop() ?? slug)
  const lines: string[] = []
  const description = descriptionToPlainText(
    file.rawDescription ??
      (typeof frontmatter.description === 'string' ? frontmatter.description : file.description),
    slug,
  )
  if (description) lines.push(description)
  for (const [key, value] of Object.entries(frontmatter)) {
    if (SKIPPED_KEYS.has(key)) continue
    const text = propertyText(value)
    if (text) lines.push(`${key}: ${text}`)
  }
  const body = (file.text ?? '').trim()
  if (body) lines.push(body.length > BODY_LIMIT ? `${body.slice(0, BODY_LIMIT)}…` : body)
  if (lines.length === 0) return null
  return { slug, title, text: lines.join('\n') }
}
