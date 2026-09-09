import { resolve } from 'node:path'
import { visit } from 'unist-util-visit'
import type { ProcessedContent } from '../plugins/vfile'
import type { ChangeEvent } from '../types/plugin'
import { hasHastClass, readTranscludeTarget } from './transclude-props'

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

  for (const [tree, file] of content) {
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
    visit(tree, 'element', node => {
      if (node.tagName !== 'blockquote' || !hasHastClass(node, 'transclude')) return
      const target = readTranscludeTarget(node)?.targetSlug
      if (!target) return
      const owner = bySlug.has(target)
        ? target
        : (bases.find(base => target.startsWith(`${base}/`)) ?? target)
      const parents = dependents.get(owner) ?? new Set<string>()
      parents.add(slug)
      dependents.set(owner, parents)
    })
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
