import { build, type Plugin } from 'esbuild'
import { createRequire } from 'node:module'
import path from 'node:path'
import type { BuildCtx } from '../../../util/ctx'
import { isFullSlug, type FilePath } from '../../../util/path'
import { write } from '../helpers'
import { codeSearchWorkerEntry, staticScriptsDir } from './asset-paths'
import { writeAssetBundleOutput } from './asset-writer'

/** Adapt wasm-bindgen's native WASM module import without rewriting published files. */
export function codeSearchPlugins(): Plugin[] {
  const resolve = createRequire(import.meta.url).resolve
  const root = path.dirname(resolve('@ast-grep/wasm'))
  return [
    {
      name: 'code-search-wasm',
      setup(build) {
        build.initialOptions.define = { ...build.initialOptions.define, process: 'undefined' }
        build.onResolve({ filter: /^(fs\/promises|module)$/ }, args => ({
          path: args.path,
          external: true,
        }))
        build.onResolve({ filter: /^@ast-grep\/wasm$/ }, () => ({
          path: 'matcher',
          namespace: 'code-search',
        }))
        build.onLoad({ filter: /.*/, namespace: 'code-search' }, () => ({
          resolveDir: root,
          contents: `
          import * as bindings from './wasm_bg.js';
          import url from './wasm_bg.wasm';
          const response = await fetch(new URL(url, import.meta.url));
          if (!response.ok) throw new Error('AST matcher download failed: ' + response.status);
          const { instance } = await WebAssembly.instantiate(await response.arrayBuffer(), { './wasm_bg.js': bindings });
          bindings.__wbg_set_wasm(instance.exports);
          instance.exports.__wbindgen_start();
          export * from './wasm_bg.js';
        `,
        }))
        build.onResolve({ filter: /^web-tree-sitter$/ }, args => {
          if (args.importer === path.join(root, 'wasm_bg.js')) {
            return { path: path.resolve('quartz/workers/code-search-ast-parser.ts') }
          }
        })
      },
    },
  ]
}

export async function writeCodeSearchAssets(ctx: BuildCtx): Promise<FilePath[]> {
  const result = await build({
    entryPoints: { 'code-search.worker': codeSearchWorkerEntry },
    outdir: path.join(ctx.argv.output, staticScriptsDir),
    bundle: true,
    minify: true,
    platform: 'browser',
    format: 'esm',
    target: 'esnext',
    splitting: true,
    entryNames: '[name]',
    chunkNames: 'chunks/[name]-[hash]',
    assetNames: 'assets/[name]-[hash]',
    loader: { '.wasm': 'file' },
    plugins: codeSearchPlugins(),
    write: false,
  })
  return Promise.all(
    result.outputFiles.map(async output => {
      if (!output.path.endsWith('.wasm')) return writeAssetBundleOutput(ctx, output)
      const relative = path.relative(ctx.argv.output, output.path).split(path.sep).join('/')
      const slug = relative.slice(0, -5)
      if (!isFullSlug(slug)) throw new Error(`Invalid code search asset path: ${relative}`)
      return write({ ctx, slug, ext: '.wasm', content: Buffer.from(output.contents) })
    }),
  )
}
