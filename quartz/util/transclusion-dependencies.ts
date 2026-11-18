import { resolve } from 'node:path'
import type { ProcessedContent } from '../plugins/vfile'
import type { ChangeEvent } from '../types/plugin'
import type { ReferenceDependency } from './references/dependencies'
import { referenceFileKind, referenceLookupKeys } from './references/catalog'
import { referenceCatalogFromFiles } from './references/context'

function embedsDocument(dependency: ReferenceDependency): boolean {
  return (
    dependency.embed &&
    !['external', 'unsupported'].includes(dependency.status) &&
    (!dependency.resolvedPath ||
      ['note', 'base', 'canvas', 'notebook', 'flashcard'].includes(
        referenceFileKind(dependency.resolvedPath),
      ))
  )
}

export function transclusionDependencies(
  content: ProcessedContent[],
): { source: string; consumer: string }[] {
  const edges = new Map<string, { source: string; consumer: string }>()
  for (const [, file] of content) {
    const consumer = file.data.slug
    if (!consumer) continue
    for (const dependency of file.data.referenceDependencies ?? []) {
      if (!embedsDocument(dependency)) continue
      const source = dependency.resolvedSlug ?? dependency.target
      if (!source) continue
      edges.set(JSON.stringify([source, consumer]), { source, consumer })
    }
  }
  return [...edges.values()]
}

export function transclusionChangeEvents(
  content: ProcessedContent[],
  events: ChangeEvent[],
  contentDirectory: string,
): ChangeEvent[] {
  const bySlug = new Map(content.map(([, file]) => [file.data.slug, file]))
  const bases = content
    .flatMap(([, file]) => (file.data.bases && file.data.slug ? [file.data.slug] : []))
    .sort((a, b) => b.length - a.length)
  const dependents = new Map<string, Set<string>>()
  const affected = new Set<string>()
  const touchedPaths = new Set<string>()
  const lookupKeys = new Set(
    referenceCatalogFromFiles(
      events.flatMap(event =>
        [event.file?.data, event.previousFile?.data].filter(file => file !== undefined),
      ),
    ).entries.flatMap(referenceLookupKeys),
  )

  for (const event of events) {
    if (event.file?.data.slug) affected.add(event.file.data.slug)
    if (event.previousFile?.data.slug) affected.add(event.previousFile.data.slug)
    touchedPaths.add(resolve(event.path))
    touchedPaths.add(resolve(contentDirectory, event.path))
  }

  for (const [, file] of content) {
    const slug = file.data.slug
    if (!slug) continue
    if (
      file.data.referenceDependencies?.some(
        dependency =>
          embedsDocument(dependency) && dependency.lookupKeys.some(key => lookupKeys.has(key)),
      )
    )
      affected.add(slug)
    if (
      file.data.codeDependencies?.some(
        dependency =>
          touchedPaths.has(resolve(dependency)) ||
          touchedPaths.has(resolve(contentDirectory, dependency)),
      )
    ) {
      affected.add(slug)
    }
  }
  for (const { source, consumer } of transclusionDependencies(content)) {
    const parents = dependents.get(source) ?? new Set<string>()
    parents.add(consumer)
    dependents.set(source, parents)
  }

  // Base queries and backlinks can change when any file in their input corpus changes.
  if (events.some(event => event.file || event.previousFile)) {
    for (const base of bases) {
      if (dependents.has(base)) affected.add(base)
    }
    for (const [, file] of content) {
      if (file.data.slug && file.data.embeddedSearchQueries?.length) affected.add(file.data.slug)
    }
  }

  // Set iteration includes additions, so nested embeds propagate through cycles once.
  for (const slug of affected) {
    for (const dependent of dependents.get(slug) ?? []) affected.add(dependent)
  }

  const emitted = new Set(
    events.flatMap(event => (event.file?.data.slug ? [event.file.data.slug] : [])),
  )
  const expanded = [...events]
  for (const [slug, file] of bySlug) {
    if (!slug || !affected.has(slug) || emitted.has(slug)) continue
    const path = file.data.relativePath ?? file.data.filePath
    if (path) expanded.push({ type: 'change', path, file })
  }
  return expanded
}
