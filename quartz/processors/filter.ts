import { BuildCtx } from "../util/ctx"
import { PerfTimer } from "../util/perf"
import { HtmlContent } from "../plugins/vfile"

export function filterContent(ctx: BuildCtx, content: HtmlContent[]): HtmlContent[] {
  const { cfg, argv } = ctx
  const perf = new PerfTimer()
  const initialLength = content.length
  for (const plugin of cfg.plugins.filters) {
    const updatedContent = content.filter((item) => plugin.shouldPublish(ctx, item))

    if (argv.verbose) {
      const diff = content.filter((x) => !updatedContent.includes(x))
      for (const file of diff) {
        console.log(`[filter:${plugin.name}] ${file[1].data.slug}`)
      }
    }

    content = updatedContent
  }

  console.log(
    `[filter] Filtered out ${initialLength - content.length} files in ${perf.timeSince()}`,
  )
  return content
}
