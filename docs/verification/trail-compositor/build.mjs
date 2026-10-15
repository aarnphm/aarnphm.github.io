// Bundles the shared trail frame and the homepage trail script the way the Quartz emitter does
// (quartz/plugins/emitters/component-resources/shared-trail-assets.ts), next to a stub homepage.
// Usage: node build.mjs [out=site] [--verify] [--repo <checkout>]
// --verify injects verify-patch.mjs into the compositor bundle: every region-painted frame is
// compared against a full repaint of the same frame.
import { build } from 'esbuild'
import { Features, transform } from 'lightningcss'
import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { compileAsync } from 'sass-embedded'
import { injectVerifier } from './verify-patch.mjs'

const here = path.dirname(fileURLToPath(import.meta.url))
const args = process.argv.slice(2)
const flag = name => (args.includes(name) ? args[args.indexOf(name) + 1] : undefined)
// --repo builds another checkout, such as a baseline worktree, with this harness.
const repo = path.resolve(flag('--repo') ?? path.resolve(here, '../../..'))
const verify = args.includes('--verify')
const out = path.resolve(
  here,
  args.find((arg, index) => !arg.startsWith('--') && args[index - 1] !== '--repo') ?? 'site',
)
const source = file => path.join(repo, file)
const minify = process.env.MINIFY !== '0'
const nodePaths = [
  path.join(repo, 'node_modules'),
  path.resolve(here, '../../../node_modules'),
  path.join(here, 'node_modules'),
]
const patches = {
  name: 'verify',
  setup(plugin) {
    if (!verify) return
    plugin.onLoad({ filter: /scene-compositor\.ts$/ }, ({ path: file }) => ({
      contents: injectVerifier(readFileSync(file, 'utf8')),
      loader: 'ts',
    }))
  },
}
const bundle = (entry, options = {}) =>
  build({
    entryPoints: [source(entry)],
    bundle: true,
    minify,
    keepNames: !minify,
    platform: 'browser',
    format: 'iife',
    write: false,
    nodePaths,
    logLevel: 'error',
    ...options,
  })

mkdirSync(out, { recursive: true })
const styles = await compileAsync(source('quartz/components/styles/trail-surface.scss'), {
  charset: false,
  style: 'expanded',
})
const css = transform({
  filename: 'trail-surface.css',
  code: Buffer.from(styles.css),
  minify: true,
  targets: { chrome: 109 << 16, safari: (15 << 16) | (6 << 8), firefox: 102 << 16 },
  include: Features.MediaQueries,
}).code.toString()
const worker = await bundle('quartz/components/visitor-trail/tree-worker.ts')
const navigation = await bundle('quartz/components/visitor-trail/navigation-worker.ts')
// Baselines from before the scenery worker never read this define.
const scenery = await bundle('quartz/components/visitor-trail/scenery-worker.ts').catch(() => null)
const frame = await bundle('quartz/components/visitor-trail/frame.ts', {
  plugins: [patches],
  define: {
    TRAIL_TREE_WORKER_SOURCE: JSON.stringify(worker.outputFiles[0].text),
    TRAIL_NAVIGATION_WORKER_SOURCE: JSON.stringify(navigation.outputFiles[0].text),
    TRAIL_SCENERY_WORKER_SOURCE: JSON.stringify(scenery?.outputFiles[0].text ?? ''),
    TRAIL_SCENE_STYLES: JSON.stringify(css),
  },
})
writeFileSync(path.join(out, 'shared-trail.js'), frame.outputFiles[0].text)
const page = await bundle('quartz/components/scripts/homepage-trail.inline.ts', { format: 'esm' })
writeFileSync(path.join(out, 'homepage-trail.js'), page.outputFiles[0].text)
const pageStyles = await compileAsync(source('quartz/components/styles/homepage-trail.scss'), {
  charset: false,
  loadPaths: [source('quartz/styles')],
})
writeFileSync(path.join(out, 'homepage-trail.css'), pageStyles.css)
copyFileSync(path.join(here, 'index.html'), path.join(out, 'index.html'))
console.log(`built ${out}${verify ? ' with the pixel verifier' : ''}`)
