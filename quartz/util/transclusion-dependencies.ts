import { resolve } from 'node:path'
import { visit } from 'unist-util-visit'
import type { ProcessedContent } from '../plugins/vfile'
import type { ChangeEvent } from '../types/plugin'
import { hasHastClass, readTranscludeTarget } from './transclude-props'

export function transclusionDependencies(
  content: ProcessedContent[],
): { source: string; consumer: string }[] {
  const slugs = new Set(content.map(([, file]) => file.data.slug))
  const bases = content
    .flatMap(([, file]) => (file.data.bases && file.data.slug ? [file.data.slug] : []))
    .sort((a, b) => b.length - a.length)
  const edges = new Map<string, { source: string; consumer: string }>()
  for (const [tree, file] of content) {
    const consumer = file.data.slug
    if (!consumer) continue
    visit(tree, 'element', node => {
      if (node.tagName !== 'blockquote' || !hasHastClass(node, 'transclude')) return
      const target = readTranscludeTarget(node)?.targetSlug
      if (!target) return
      const source = slugs.has(target)
        ? target
        : (bases.find(base => target.startsWith(`${base}/`)) ?? target)
      edges.set(JSON.stringify([source, consumer]), { source, consumer })
    })
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
