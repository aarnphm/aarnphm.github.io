import { build, type Plugin } from 'esbuild'
import { readFile } from 'node:fs/promises'
import { devFeatureEnabled, type DevFeature } from '../../dev-features'
import type { QuartzEmitterPlugin } from '../../types/plugin'
import { FullSlug, joinSegments, QUARTZ } from '../../util/path'
import { staticScriptsDir } from './component-resources/asset-paths'
import { applyScriptAssetReferences } from './component-resources/script-assets'
import { write } from './helpers'

export type LazyScript = {
  /** emitted as `static/scripts/<name>.js`; a page-script loader imports it relative to `import.meta.url` */
  name: string
  /** entry point, relative to the `quartz` folder */
  entry: string
  devFeature?: DevFeature
}

export type Options = { scripts: LazyScript[] }

/**
 * pseudocode.js (a lazy script's pseudo fences) requires KaTeX itself, which would add ~500kB to a
 * script that typesets its math elsewhere; the require sits in a try, so an empty module is enough.
 * Its sources also assign two undeclared variables, which a strict-mode bundle turns into
 * ReferenceErrors; they are declared here.
 */
const pseudocodeBundle: Plugin = {
  name: 'pseudocode-bundle',
  setup(b) {
    const inPseudocode = /[\\/]node_modules[\\/]pseudocode[\\/]/
    b.onResolve({ filter: /^(katex|mathjax)$/ }, args =>
      inPseudocode.test(args.importer) ? { path: args.path, namespace: 'no-math' } : undefined,
    )
    b.onLoad({ filter: /.*/, namespace: 'no-math' }, args => ({
      contents: `throw new Error('${args.path} is not bundled')`,
      loader: 'js',
    }))
    b.onLoad({ filter: /[\\/]pseudocode[\\/]src[\\/][^\\/]+\.js$/ }, async args => ({
      contents: `var attrVal, ifCond;\n${await readFile(args.path, 'utf8')}`,
      loader: 'js',
    }))
  },
}

/**
 * Bundles a script under `static/scripts/` instead of into the page scripts every page loads,
 * so a heavy module only some pages need stays off the rest. The filename stays unhashed:
 * ComponentResources bundles the loaders before this emitter runs, so they cannot learn a hash.
 */
export const LazyScripts: QuartzEmitterPlugin<Options> = opts => {
  const scripts = opts?.scripts ?? []
  return {
    name: 'LazyScripts',
    async *emit(ctx) {
      const active = scripts.filter(
        script => !script.devFeature || devFeatureEnabled(ctx.argv, script.devFeature),
      )
      yield* await Promise.all(
        active.map(async ({ name, entry }) => {
          const result = await build({
            entryPoints: [joinSegments(QUARTZ, entry)],
            bundle: true,
            minify: true,
            platform: 'browser',
            format: 'esm',
            write: false,
            plugins: [pseudocodeBundle],
          })

          return write({
            ctx,
            slug: joinSegments(staticScriptsDir, name) as FullSlug,
            ext: '.js',
            // a lazy script may spawn the site's workers by their bare names
            content: applyScriptAssetReferences(ctx, result.outputFiles[0].text),
          })
        }),
      )
    },
    async *partialEmit() {},
  }
}
