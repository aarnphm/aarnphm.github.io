import { build } from 'esbuild'
import { compileAsync } from 'sass-embedded'
import type { BuildCtx } from '../../../util/ctx'
import { minifyStylesheet } from '../../../util/minify-stylesheet'
import { joinSegments } from '../../../util/path'
import { resolveAssetPath, writeRawAsset } from './asset-writer'

export async function writeSharedTrailAsset(ctx: BuildCtx) {
  const styles = await compileAsync('quartz/components/styles/trail-surface.scss', {
    charset: false,
    style: 'expanded',
  })
  const worker = await build({
    entryPoints: ['quartz/components/visitor-trail/tree-worker.ts'],
    bundle: true,
    minify: true,
    platform: 'browser',
    format: 'iife',
    write: false,
  })
  const navigation = await build({
    entryPoints: ['quartz/components/visitor-trail/navigation-worker.ts'],
    bundle: true,
    minify: true,
    platform: 'browser',
    format: 'iife',
    write: false,
  })
  const moss = await build({
    entryPoints: ['quartz/components/visitor-trail/moss-worker.ts'],
    bundle: true,
    minify: true,
    platform: 'browser',
    format: 'iife',
    write: false,
  })
  const entries = [
    { path: 'static/scripts/trail-trees.worker.js', source: worker.outputFiles[0].text },
    { path: 'static/scripts/trail-navigation.worker.js', source: navigation.outputFiles[0].text },
    { path: 'static/scripts/trail-moss.worker.js', source: moss.outputFiles[0].text },
  ]
  const workerFiles = await Promise.all(
    entries.map(entry => writeRawAsset(ctx, entry.path, entry.source)),
  )
  const reference = (path: string) => JSON.stringify(joinSegments('/', resolveAssetPath(ctx, path)))
  const result = await build({
    entryPoints: ['quartz/components/visitor-trail/frame.ts'],
    bundle: true,
    minify: true,
    platform: 'browser',
    format: 'iife',
    write: false,
    define: {
      TRAIL_TREE_WORKER_URL: reference('static/scripts/trail-trees.worker.js'),
      TRAIL_NAVIGATION_WORKER_URL: reference('static/scripts/trail-navigation.worker.js'),
      TRAIL_MOSS_WORKER_URL: reference('static/scripts/trail-moss.worker.js'),
      TRAIL_SCENE_STYLES: JSON.stringify(minifyStylesheet('trail-surface.css', styles.css)),
    },
  })
  return [
    ...workerFiles,
    await writeRawAsset(ctx, 'static/scripts/shared-trail.js', result.outputFiles[0].text),
  ]
}
