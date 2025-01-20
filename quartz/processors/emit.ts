import { PerfTimer } from "../util/perf"
import { getStaticResourcesFromPlugins } from "../plugins"
import { HtmlContent } from "../plugins/vfile"
import { QuartzLogger } from "../util/log"
import { trace } from "../util/trace"
import { BuildCtx } from "../util/ctx"
import { styleText } from "node:util"

type ParsedContents = {
  all: HtmlContent[]
  delta?: HtmlContent[]
}

type EmitOptions = {
  ctx: BuildCtx
  contents: ParsedContents
  incremental?: boolean
}

export async function emitContent({ ctx, contents, incremental }: EmitOptions) {
  incremental ||= false
  const { argv, cfg } = ctx
  const { all, delta } = contents
  const perf = new PerfTimer()
  const log = new QuartzLogger(ctx.argv.verbose)

  log.start(styleText("blue", `[emit] Emitting output files`))

  let emittedFiles = 0
  const staticResources = getStaticResourcesFromPlugins(ctx)
  for (const emitter of cfg.plugins.emitters) {
    if (emitter.skipDuringServe) {
      if (argv.verbose)
        console.log(styleText("yellow", `[emit:${emitter.name}] Skip during serve time`))
      continue
    }

    const emitterPerf = new PerfTimer()
    try {
      const emitted = await emitter.emit(
        ctx,
        incremental && emitter.requiresFullContent !== undefined && emitter.requiresFullContent
          ? all
          : (delta ?? all),
        staticResources,
      )
      emittedFiles += emitted.length

      if (ctx.argv.verbose) {
        for (const file of emitted) {
          console.log(`[emit:${emitter.name}] ${file}`)
        }
        console.log(
          styleText(
            "cyan",
            `[emit:${emitter.name}] Emit ${emitted.length} files in ${emitterPerf.timeSince()}`,
          ),
        )
      }
    } catch (err) {
      trace(`Failed to emit from plugin \`${emitter.name}\``, err as Error)
    }
  }

  log.end(
    styleText(
      "blue",
      `[emit] Emitted ${emittedFiles} files to \`${argv.output}\` in ${perf.timeSince()}`,
    ),
  )
}
