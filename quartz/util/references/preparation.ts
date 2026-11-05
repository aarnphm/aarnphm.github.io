import type { MarkdownContent, ProcessedContent } from '../../plugins/vfile'
import type { ChangeEvent } from '../../types/plugin'
import type { BuildCtx } from '../ctx'
import type { FilePath } from '../path'
import type { ReferenceRequestRecord } from './dependencies'
import { referenceCatalogFor, updateReferenceCatalog } from './context'
import { documentReferenceRequests, referenceResolutionSignature } from './markdown'

/** The parse coordinator owns ordering; feature emitters own source preparation and caching. */
export async function prepareReferenceContent(
  ctx: BuildCtx,
  markdown: readonly MarkdownContent[],
  cached: ReadonlyMap<FilePath, ProcessedContent>,
  begin: boolean,
): Promise<void> {
  const owners = ctx.cfg.plugins.emitters.filter(emitter => emitter.prepareReferences)
  if (owners.length === 0) return
  if (begin) {
    ctx.referencePreparationBuildId = ctx.buildId
    ctx.previousReferenceContent = ctx.referenceContent
    ctx.referenceContent = new Map()
  }
  const prepared = new Map(ctx.referenceContent)
  const attempted = new Set(prepared.keys())
  let requests: ReferenceRequestRecord[] = [
    ...markdown.flatMap(([tree, file]) => documentReferenceRequests(ctx, tree, file)),
    ...[...cached.values()].flatMap(([, file]) => file.data.referenceDependencies ?? []),
  ]
  // Every wave either discovers a finite catalog target or refreshes a selector after its
  // target supplied metadata. Cycles cannot grow the source set beyond the catalog.
  for (let wave = 0; wave <= ctx.allFiles.length; wave += 1) {
    const paths = new Set<FilePath>()
    for (const request of requests) {
      const result = referenceCatalogFor(ctx).resolve(request.target, request.origin, request)
      if (result.kind !== 'found' || result.entry.kind !== 'notebook') continue
      const source = ctx.allFiles.find(file => file === result.entry.path)
      if (source && !attempted.has(source)) paths.add(source)
    }
    if (paths.size === 0) {
      for (const [source, [, file]] of prepared) {
        if (referenceResolutionSignature(ctx, file) !== file.data.referenceResolutionSignature)
          paths.add(source)
      }
      if (paths.size === 0) return
    }
    for (const source of paths) attempted.add(source)
    requests = []
    for (const owner of owners) {
      const pages = (await owner.prepareReferences?.(ctx, [...paths])) ?? []
      updateReferenceCatalog(
        ctx,
        pages.map(([, file]) => file.data),
      )
      for (const page of pages) {
        if (!ctx.cfg.plugins.filters.every(filter => filter.shouldPublish(ctx, page))) continue
        const source = page[1].data.relativePath
        if (source) prepared.set(source, page)
        requests.push(...(page[1].data.referenceDependencies ?? []))
      }
    }
    ctx.referenceContent = new Map(prepared)
  }
  throw new Error('Reference preparation did not reach a stable catalog')
}

/** Non-Markdown source events carry the same identities as ordinary parsed documents. */
export function referenceChangeEvents(ctx: BuildCtx, events: ChangeEvent[]): ChangeEvent[] {
  return events.map(event => {
    const current = ctx.referenceContent?.get(event.path)?.[1]
    const previous = ctx.previousReferenceContent?.get(event.path)?.[1]
    return current || previous
      ? { ...event, file: current ?? previous, previousFile: previous }
      : event
  })
}

export async function ensureReferenceContent(
  ctx: BuildCtx,
  content: ProcessedContent[],
): Promise<void> {
  if (ctx.referencePreparationBuildId === ctx.buildId) return
  const cached = new Map<FilePath, ProcessedContent>()
  for (const item of content) {
    const source = item[1].data.relativePath
    if (source) cached.set(source, item)
  }
  await prepareReferenceContent(ctx, [], cached, true)
}
