import { build, type Plugin } from 'esbuild'
import type { BuildCtx } from '../../../util/ctx'
import type { FilePath } from '../../../util/path'
import { shouldHashAssets } from '../../../util/asset-manifest'
import {
  clientHighlighterEntry,
  externalizeClientHighlighter,
} from '../../../util/inline-script-bundler'
import { ReloadableWeakMap } from '../../../util/module-caches'
import { clientHighlighterPath } from './asset-paths'
import { resolveAssetPath, writeRawAsset } from './asset-writer'

type PreparedHighlighter = { file: FilePath; plugin: Plugin }
const pending = new ReloadableWeakMap<
  BuildCtx,
  { identity: string; value: Promise<PreparedHighlighter> }
>()

/** Every client receives the same registered URL, including independent notebook emission. */
export function prepareClientHighlighter(ctx: BuildCtx): Promise<PreparedHighlighter> {
  const identity = `${ctx.buildId}\0${ctx.argv.output}\0${shouldHashAssets(ctx)}`
  const existing = pending.get(ctx)
  if (existing?.identity === identity) return existing.value
  const value = emitClientHighlighter(ctx).catch(error => {
    if (pending.get(ctx)?.value === value) pending.delete(ctx)
    throw error
  })
  pending.set(ctx, { identity, value })
  return value
}

async function emitClientHighlighter(ctx: BuildCtx): Promise<PreparedHighlighter> {
  const result = await build({
    entryPoints: [clientHighlighterEntry],
    bundle: true,
    minify: true,
    platform: 'browser',
    format: 'esm',
    write: false,
  })
  const file = await writeRawAsset(ctx, clientHighlighterPath, result.outputFiles[0].text)
  return {
    file,
    plugin: externalizeClientHighlighter(`/${resolveAssetPath(ctx, clientHighlighterPath)}`),
  }
}
