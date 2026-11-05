import type { FullSlug } from '../path'
import type {
  ReferenceCatalog,
  ReferenceCatalogEntry,
  ReferenceCatalogInput,
  ReferenceFileKind,
  ReferenceResolution,
  ReferenceResolutionOptions,
} from './catalog-types'
import { isFlashcardPath } from '../flashcards-path'
import { getFileExtension, isFilePath, isFullSlug, slugifyFilePath } from '../path'
import { canonicalTagRoute } from '../tags'

export type * from './catalog-types'

const fold = (value: string): string => value.normalize('NFC').toLowerCase()
const basename = (value: string): string => value.slice(value.lastIndexOf('/') + 1)
const dirname = (value: string): string => value.slice(0, Math.max(0, value.lastIndexOf('/')))

/** Returns undefined when a relative traversal would leave the vault. */
export function normalizeReferencePath(value: string): string | undefined {
  const segments: string[] = []
  for (const segment of value.replaceAll('\\', '/').split('/')) {
    if (!segment || segment === '.') continue
    if (segment === '..') {
      if (segments.length === 0) return undefined
      segments.pop()
    } else segments.push(segment)
  }
  return segments.join('/')
}

export function referenceFileKind(filePath: string): ReferenceFileKind {
  if (isFlashcardPath(filePath)) return 'flashcard'
  const extension = getFileExtension(filePath)?.toLowerCase()
  if (extension === '.md' || extension === '.html') return 'note'
  if (extension === '.base') return 'base'
  if (extension === '.canvas') return 'canvas'
  if (extension === '.ipynb') return 'notebook'
  if (extension === '.pdf') return 'pdf'
  if (['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.webp', '.avif'].includes(extension ?? ''))
    return 'image'
  if (['.mp3', '.wav', '.m4a', '.ogg', '.3gp', '.flac'].includes(extension ?? '')) return 'audio'
  if (['.mp4', '.webm', '.ogv', '.mov', '.mkv'].includes(extension ?? '')) return 'video'
  if (
    [
      '.py',
      '.go',
      '.java',
      '.c',
      '.cpp',
      '.cxx',
      '.cu',
      '.cuh',
      '.h',
      '.hpp',
      '.ts',
      '.tsx',
      '.yaml',
      '.yml',
      '.rs',
      '.m',
      '.sql',
      '.sh',
      '.txt',
    ].includes(extension ?? '')
  )
    return 'code'
  return 'unknown'
}

function identityKeys(entry: ReferenceCatalogEntry): string[] {
  const keys = [entry.path]
  if (['note', 'base', 'canvas'].includes(entry.kind))
    keys.push(entry.path.replace(/\.[^./]+$/, ''))
  return keys
}

/** Candidate changes invalidate the same keys consulted during target lookup. */
export function referenceLookupKeys(entry: ReferenceCatalogEntry): readonly string[] {
  return [
    ...identityKeys(entry).flatMap(key => [
      `path:${key}`,
      `fold:${fold(key)}`,
      `name:${fold(basename(key))}`,
    ]),
    `route:${fold(entry.slug)}`,
    ...entry.aliases.map(alias => `alias:${fold(alias)}`),
  ]
}

function distance(origin: string, candidate: string): number {
  const a = dirname(origin).split('/').filter(Boolean)
  const b = dirname(candidate).split('/').filter(Boolean)
  let shared = 0
  while (shared < a.length && shared < b.length && fold(a[shared]) === fold(b[shared])) shared++
  return a.length + b.length - shared * 2
}

function decodeTarget(target: string, decodeUri: boolean | undefined): string {
  if (!decodeUri) return target
  try {
    return decodeURIComponent(target)
  } catch {
    return target
  }
}

/** A parser's default source slug is distinct from a publisher's output address. */
export function referencePublicationSlug(path: string, inputSlug?: FullSlug): FullSlug | undefined {
  const sourceSlug = isFilePath(path) ? slugifyFilePath(path) : isFullSlug(path) ? path : undefined
  if (!sourceSlug) return inputSlug
  if (inputSlug !== undefined && inputSlug !== sourceSlug) return inputSlug
  const published =
    referenceFileKind(path) === 'notebook' && isFilePath(path)
      ? slugifyFilePath(path, true)
      : referenceFileKind(path) === 'note' && getFileExtension(path)?.toLowerCase() === '.md'
        ? canonicalTagRoute(sourceSlug)
        : sourceSlug
  return isFullSlug(published) ? published : sourceSlug
}

export function createReferenceCatalog(inputs: readonly ReferenceCatalogInput[]): ReferenceCatalog {
  const paths = new Set<string>()
  const entries: ReferenceCatalogEntry[] = []
  for (const input of inputs) {
    const sourcePath = normalizeReferencePath(input.path)
    if (!sourcePath || paths.has(sourcePath)) continue
    const slug = referencePublicationSlug(sourcePath, input.slug)
    if (!slug) continue
    paths.add(sourcePath)
    entries.push(
      Object.freeze({
        path: sourcePath,
        slug,
        kind: referenceFileKind(sourcePath),
        aliases: Object.freeze([...(input.aliases ?? [])]),
        ...(input.headings
          ? {
              headings: Object.freeze(
                input.headings.map(heading =>
                  Object.freeze({
                    ...heading,
                    ...(heading.path ? { path: Object.freeze([...heading.path]) } : {}),
                  }),
                ),
              ),
            }
          : {}),
        ...(input.blocks ? { blocks: Object.freeze([...input.blocks]) } : {}),
        ...(input.baseViews
          ? { baseViews: Object.freeze(input.baseViews.map(view => Object.freeze({ ...view }))) }
          : {}),
      }),
    )
  }
  entries.sort((a, b) => (a.path < b.path ? -1 : a.path > b.path ? 1 : 0))
  const exact = new Map<string, ReferenceCatalogEntry[]>()
  const normalized = new Map<string, ReferenceCatalogEntry[]>()
  const names = new Map<string, ReferenceCatalogEntry[]>()
  const aliases = new Map<string, ReferenceCatalogEntry[]>()
  const routes = new Map<string, ReferenceCatalogEntry[]>()
  function add(
    index: Map<string, ReferenceCatalogEntry[]>,
    key: string,
    entry: ReferenceCatalogEntry,
  ) {
    const values = index.get(key) ?? []
    if (!values.includes(entry)) values.push(entry)
    index.set(key, values)
  }
  for (const entry of entries) {
    for (const key of identityKeys(entry)) {
      add(exact, key, entry)
      add(normalized, fold(key), entry)
      add(names, fold(basename(key)), entry)
    }
    add(routes, fold(entry.slug), entry)
    for (const alias of entry.aliases) add(aliases, fold(alias), entry)
  }

  function resolve(
    target: string,
    origin: string,
    options: ReferenceResolutionOptions = {},
  ): ReferenceResolution {
    const lookupKeys = new Set<string>()
    const original = target
    if (/^(?:[a-z][a-z\d+.-]*:|\/\/)/i.test(target))
      return { kind: 'external', target, lookupKeys: [] }
    target = decodeTarget(target, options.decodeUri).trim().replaceAll('\\', '/')
    const source = normalizeReferencePath(origin) ?? origin
    const relative = target.startsWith('./') || target.startsWith('../')
    const explicit = relative || target.startsWith('/') || target.includes('/')
    const sought = normalizeReferencePath(
      !target ? source : relative ? `${dirname(source)}/${target}` : target,
    )
    if (sought === undefined)
      return { kind: 'unsupported', target: original, reason: 'outside-vault', lookupKeys: [] }
    function result(candidates: readonly ReferenceCatalogEntry[]): ReferenceResolution | undefined {
      if (candidates.length === 0) return undefined
      const keys = Object.freeze([...lookupKeys])
      return candidates.length === 1
        ? { kind: 'found', entry: candidates[0], lookupKeys: keys }
        : {
            kind: 'ambiguous',
            target: original,
            candidates: Object.freeze([...candidates]),
            lookupKeys: keys,
          }
    }
    function direct(key: string): ReferenceResolution | undefined {
      lookupKeys.add(`path:${key}`)
      lookupKeys.add(`fold:${fold(key)}`)
      return result(exact.get(key) ?? []) ?? result(normalized.get(fold(key)) ?? [])
    }
    const matched = direct(sought)
    if (matched) return matched
    // Published destinations remain a valid adapter input, after source identity lookup.
    lookupKeys.add(`route:${fold(sought)}`)
    const routed = result(routes.get(fold(sought)) ?? [])
    if (routed) return routed
    if (!explicit) {
      lookupKeys.add(`name:${fold(sought)}`)
      const candidates = names.get(fold(sought)) ?? []
      if (candidates.length > 0) {
        const nearest = Math.min(...candidates.map(entry => distance(source, entry.path)))
        const local = result(candidates.filter(entry => distance(source, entry.path) === nearest))
        if (local) return local
      }
      if (options.aliases) {
        lookupKeys.add(`alias:${fold(sought)}`)
        const aliased = result(aliases.get(fold(sought)) ?? [])
        if (aliased) return aliased
      }
    }
    if (options.implicitAlias) {
      const last = target.lastIndexOf('/') + 1
      const space = target.indexOf(' ', last)
      if (space >= 0) {
        const prefix = target.slice(0, space)
        const implicit = resolve(prefix, origin, {
          ...options,
          implicitAlias: false,
          decodeUri: false,
        })
        for (const key of implicit.lookupKeys) lookupKeys.add(key)
        if (implicit.kind === 'found')
          return {
            ...implicit,
            implicitLabel: target.slice(space + 1).trim(),
            lookupKeys: Object.freeze([...lookupKeys]),
          }
        if (implicit.kind === 'ambiguous')
          return { ...implicit, target: original, lookupKeys: Object.freeze([...lookupKeys]) }
      }
    }
    return { kind: 'missing', target: original, lookupKeys: Object.freeze([...lookupKeys]) }
  }
  return Object.freeze({ entries: Object.freeze(entries), resolve })
}
