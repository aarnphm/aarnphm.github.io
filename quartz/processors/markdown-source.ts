import type { BuildCtx } from '../util/ctx'

export function transformMarkdownSource(ctx: BuildCtx, source: string): string {
  for (const plugin of ctx.cfg.plugins.transformers) {
    if (plugin.textTransform) source = plugin.textTransform(ctx, source)
  }
  return source
}
