import type { FullSlug } from '../path'
import type { ReferenceCatalog } from './catalog'
import type { ReferenceSyntax } from './model'
import type { ReferenceRoute } from './route'
import { referenceFields } from './request'
import { missingReferenceRoute, referenceRoute } from './route'
import { parseReference, scanReferences } from './syntax'
import { interpretReference } from './target'

const CACHE_LIMIT = 4096
const parsedReferences = new Map<string, ReferenceSyntax | null>()
const extractedReferences = new Map<string, readonly ReferenceSyntax[]>()

function cacheResult<T>(cache: Map<string, T>, key: string, result: T): T {
  if (cache.size >= CACHE_LIMIT) {
    const oldest = cache.keys().next().value
    if (oldest !== undefined) cache.delete(oldest)
  }
  cache.set(key, result)
  return result
}

export function parseLiteralReference(raw: string): ReferenceSyntax | null {
  const source = raw.trim()
  if (!source || (!source.startsWith('[[') && !source.startsWith('![[')) || !source.endsWith(']]'))
    return null
  const cached = parsedReferences.get(source)
  if (cached !== undefined) return cached
  const parsed = parseReference(source, { context: 'literal' })
  return cacheResult(parsedReferences, source, parsed.kind === 'complete' ? parsed.reference : null)
}

export function extractLiteralReferences(text: string): readonly ReferenceSyntax[] {
  if (!text.includes('[[')) return []
  const canCache = text.length <= 256
  const cached = canCache ? extractedReferences.get(text) : undefined
  if (cached) return cached
  const result = Object.freeze(scanReferences(text, { context: 'literal' }))
  return canCache ? cacheResult(extractedReferences, text, result) : result
}

export interface LiteralReferenceResolutionOptions {
  readonly catalog: ReferenceCatalog
  readonly origin?: string
}

export interface ResolvedLiteralReference extends ReferenceRoute {
  readonly implicitLabel?: string
}

export function resolveLiteralReference(
  reference: ReferenceSyntax,
  currentSlug: FullSlug,
  options?: LiteralReferenceResolutionOptions,
): ResolvedLiteralReference | null {
  const fields = referenceFields(reference)
  const origin = options?.origin ?? (currentSlug || 'index')
  if (options) {
    const resolution = options.catalog.resolve(fields.target, origin, { implicitAlias: true })
    if (resolution.kind === 'found') {
      const interpretation = interpretReference(resolution.entry, fields)
      const route = referenceRoute(resolution.entry, interpretation.selector)
      return resolution.implicitLabel
        ? { ...route, implicitLabel: resolution.implicitLabel }
        : route
    }
    if (resolution.kind !== 'missing') return null
  }
  return missingReferenceRoute(fields.target, origin, fields.fragments) ?? null
}

export function referenceLabelSource(
  reference: ReferenceSyntax,
  fallback = '',
  implicitLabel?: string,
): string {
  if (reference.label) return reference.label.raw
  if (implicitLabel !== undefined) return implicitLabel
  const fields = referenceFields(reference)
  const fragments = fields.fragments.map(fragment => fragment.replace(/^\^/, ''))
  return [fields.target, ...fragments].filter(Boolean).join(' > ') || fallback
}

export function stripReferenceFormatting(text: string): string {
  const references = extractLiteralReferences(text)
  if (references.length === 0) return text
  let result = ''
  let previous = 0
  for (const reference of references) {
    const fields = referenceFields(reference)
    result += text.slice(previous, reference.span.start) + (fields.label ?? fields.target)
    previous = reference.span.end
  }
  return result + text.slice(previous)
}
