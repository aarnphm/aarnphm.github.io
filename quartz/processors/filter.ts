import { ProcessedContent } from '../plugins/vfile'
import { BuildCtx } from '../util/ctx'
import { PerfTimer } from '../util/perf'

export type FilterContentResult = { published: ProcessedContent[]; removed: ProcessedContent[] }

export function filterContentResult(
  ctx: BuildCtx,
  content: ProcessedContent[],
): FilterContentResult {
  const { cfg, argv } = ctx
  const perf = new PerfTimer()
  const initialLength = content.length
  const removed: ProcessedContent[] = []

  for (const plugin of cfg.plugins.filters) {
    const published: ProcessedContent[] = []
    const filtered: ProcessedContent[] = []

    for (const item of content) {
      if (plugin.shouldPublish(ctx, item)) {
        published.push(item)
      } else {
        filtered.push(item)
      }
    }

    if (argv.verbose) {
      for (const file of filtered) {
        console.log(`[filter:${plugin.name}] ${file[1].data.slug}`)
      }
    }

    removed.push(...filtered)
    content = published
  }

  console.log(
    `[filter] Filtered out ${initialLength - content.length} files in ${perf.timeSince()}`,
  )
  return { published: content, removed }
}

export function filterContent(ctx: BuildCtx, content: ProcessedContent[]): ProcessedContent[] {
  return filterContentResult(ctx, content).published
}
