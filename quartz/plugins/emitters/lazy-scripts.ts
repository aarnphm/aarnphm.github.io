import { build } from 'esbuild'
import type { QuartzEmitterPlugin } from '../../types/plugin'
import { FullSlug, joinSegments, QUARTZ } from '../../util/path'
import { staticScriptsDir } from './component-resources/asset-paths'
import { write } from './helpers'

export type LazyScript = {
  /** emitted as `static/scripts/<name>.js`; a page-script loader imports it relative to `import.meta.url` */
  name: string
  /** entry point, relative to the `quartz` folder */
  entry: string
}

export type Options = { scripts: LazyScript[] }

/**
 * Bundles a script under `static/scripts/` instead of into the page scripts every page loads,
 * so a heavy module only some pages need stays off the rest. The filename stays unhashed:
 * ComponentResources bundles the loaders before this emitter runs, so they cannot learn a hash.
 */
export const LazyScripts: QuartzEmitterPlugin<Options> = opts => {
  const scripts = opts?.scripts ?? []
  return {
    name: 'LazyScripts',
    async emit(ctx) {
      return Promise.all(
        scripts.map(async ({ name, entry }) => {
          const result = await build({
            entryPoints: [joinSegments(QUARTZ, entry)],
            bundle: true,
            minify: true,
            platform: 'browser',
            format: 'esm',
            write: false,
          })
          return write({
            ctx,
            slug: joinSegments(staticScriptsDir, name) as FullSlug,
            ext: '.js',
            content: result.outputFiles[0].text,
          })
        }),
      )
    },
    async *partialEmit() {},
  }
}
