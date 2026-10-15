import { build } from 'esbuild'
import { compileAsync } from 'sass-embedded'
import type { BuildCtx } from '../../../util/ctx'
import { minifyStylesheet } from '../../../util/minify-stylesheet'
import { writeRawAsset } from './asset-writer'

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
  const result = await build({
    entryPoints: ['quartz/components/visitor-trail/frame.ts'],
    bundle: true,
    minify: true,
    platform: 'browser',
    format: 'iife',
    write: false,
    define: {
      TRAIL_TREE_WORKER_SOURCE: JSON.stringify(worker.outputFiles[0].text),
      TRAIL_NAVIGATION_WORKER_SOURCE: JSON.stringify(navigation.outputFiles[0].text),
      TRAIL_SCENE_STYLES: JSON.stringify(minifyStylesheet('trail-surface.css', styles.css)),
    },
  })
  return writeRawAsset(ctx, 'static/scripts/shared-trail.js', result.outputFiles[0].text)
}
