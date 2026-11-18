import type { FullSlug } from '../path'
import type { ReferenceCatalogEntry, ReferenceSelector } from './catalog-types'
import { flashcardsSlug, sourceSlugForDeck } from '../flashcards-path'
import {
  getFileExtension,
  isFilePath,
  isFullSlug,
  resolveRelative,
  slugifyFilePath,
  sluggify,
} from '../path'
import { normalizeReferencePath, referenceFileKind, referencePublicationSlug } from './catalog'
import { interpretReference } from './target'

export interface ReferenceRoute {
  readonly slug: FullSlug
  readonly anchor?: string
}

export function referenceHref(from: FullSlug, route: ReferenceRoute, samePage = false): string {
  const anchor = (route.anchor ?? '').replace(/^#\^/, '#')
  if (samePage) return anchor
  if (route.slug.toLowerCase().endsWith('.pdf')) return `/${route.slug}${anchor}`
  return `${resolveRelative(from, route.slug)}${anchor}`
}

/** Projects a destination without adding a catalog member or claiming existence. */
export function missingReferenceRoute(
  target: string,
  origin: string,
  fragments: readonly string[] = [],
): ReferenceRoute | undefined {
  if (/^(?:[a-z][a-z\d+.-]*:|\/\/)/i.test(target)) return undefined
  const source = normalizeReferencePath(origin)
  if (!source) return undefined
  const raw = target.trim().replaceAll('\\', '/')
  const directory = source.slice(0, Math.max(0, source.lastIndexOf('/')))
  const relative = raw.startsWith('./') || raw.startsWith('../')
  const projected = normalizeReferencePath(!raw ? source : relative ? `${directory}/${raw}` : raw)
  if (projected === undefined) return undefined
  const path = raw.endsWith('/')
    ? `${projected ? `${projected}/` : ''}index.md`
    : getFileExtension(projected)
      ? projected
      : `${projected}.md`
  if (!isFilePath(path)) return undefined
  const entry: ReferenceCatalogEntry = {
    path,
    slug: referencePublicationSlug(path) ?? slugifyFilePath(path),
    kind: referenceFileKind(path),
    aliases: [],
  }
  return referenceRoute(entry, interpretReference(entry, { fragments }).selector)
}

export function referenceRoute(
  entry: ReferenceCatalogEntry,
  selector?: ReferenceSelector,
): ReferenceRoute {
  if (entry.kind === 'flashcard') return { slug: flashcardsSlug(sourceSlugForDeck(entry.slug)) }
  if (!selector || selector.kind === 'whole') return { slug: entry.slug }
  if (selector.kind === 'base') {
    if (selector.slug) return { slug: selector.slug }
    const index = entry.baseViews?.findIndex(
      view =>
        view.name.normalize('NFC').trim().toLowerCase() ===
        selector.view.normalize('NFC').trim().toLowerCase(),
    )
    if (index === 0) return { slug: entry.slug }
    const candidate = `${entry.slug}/${sluggify(selector.view)}`
    return { slug: isFullSlug(candidate) ? candidate : entry.slug }
  }
  if (selector.kind === 'heading' || selector.kind === 'block')
    return { slug: entry.slug, anchor: selector.anchor }
  if (selector.kind === 'pdf') {
    const parameters = new URLSearchParams()
    if (selector.page) parameters.set('page', selector.page)
    if (selector.height) parameters.set('height', selector.height)
    return { slug: entry.slug, ...(parameters.size > 0 ? { anchor: `#${parameters}` } : {}) }
  }
  if (selector.kind === 'notebook')
    return { slug: entry.slug, anchor: `#${selector.cells.join('#')}` }
  // /embeddings opens a view from `?focus=N#view=…` (latent-client/share.ts)
  if (selector.kind === 'latent')
    return {
      slug: entry.slug,
      anchor: selector.view
        ? `#view=${selector.view}`
        : selector.focus !== undefined
          ? `?focus=${selector.focus}`
          : undefined,
    }
  return { slug: entry.slug, anchor: `#${selector.raw}` }
}
