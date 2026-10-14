import { build } from 'esbuild'
import type { BuildCtx } from '../../../util/ctx'
import { writeRawAsset } from './asset-writer'

export const sharedTrailPath = 'static/scripts/shared-trail.html'

export async function writeSharedTrailAsset(ctx: BuildCtx) {
  const worker = await build({
    entryPoints: ['quartz/components/visitor-trail/tree-worker.ts'],
    bundle: true,
    minify: true,
    platform: 'browser',
    format: 'iife',
    write: false,
  })
  const result = await build({
    entryPoints: ['quartz/components/visitor-trail/frame.ts'],
    bundle: true,
    minify: true,
    platform: 'browser',
    format: 'iife',
    write: false,
    define: { TRAIL_TREE_WORKER_SOURCE: JSON.stringify(worker.outputFiles[0].text) },
  })
  const code = result.outputFiles[0].text.replaceAll('</script', '<\\/script')
  return writeRawAsset(
    ctx,
    sharedTrailPath,
    `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta http-equiv="Content-Security-Policy" content="style-src 'self' 'unsafe-inline'"><title>Shared trail</title></head><body><script>${code}</script></body></html>`,
  )
}
