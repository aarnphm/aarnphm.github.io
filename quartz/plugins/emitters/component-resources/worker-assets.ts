import { build as bundle } from 'esbuild'
import { globby } from 'globby'
import fs from 'node:fs/promises'
import { createRequire } from 'node:module'
import path from 'path'
import type { BuildCtx } from '../../../util/ctx'
import type { FilePath, FullSlug } from '../../../util/path'
import { workerEntryPattern } from '../../../util/workers'
import { write } from '../helpers'
import {
  collaborativeCommentsClientEntry,
  lexicalWorkerEntry,
  lexicalWorkerPath,
  semanticWorkerEntry,
  staticScriptsDir,
} from './asset-paths'
import { writeRawAsset, writeAssetBundleOutput } from './asset-writer'

const speechWorkerEntry = 'quartz/workers/speech.worker.ts'
export async function writeSpeechRuntimeAssets(ctx: BuildCtx): Promise<FilePath[]> {
  const resolve = createRequire(import.meta.url).resolve
  const wasm = await fs.readFile(resolve('onnxruntime-web/ort-wasm-simd-threaded.wasm'))
  return [
    await write({
      ctx,
      slug: 'static/speech/ort-wasm-simd-threaded' as FullSlug,
      ext: '.wasm',
      content: wasm,
    }),
    await writeGenericWorkerAsset(ctx, speechWorkerEntry),
  ]
}

export async function writeCollaborativeCommentsAssets(ctx: BuildCtx): Promise<FilePath[]> {
  const outdir = path.join(ctx.argv.output, staticScriptsDir)
  const client = await bundle({
    entryPoints: { 'collaborative-comments.client': collaborativeCommentsClientEntry },
    bundle: true,
    minify: true,
    platform: 'browser',
    format: 'esm',
    splitting: true,
    outdir,
    entryNames: '[name]',
    chunkNames: 'chunks/[name]-[hash]',
    write: false,
  })
  return await Promise.all(client.outputFiles.map(output => writeAssetBundleOutput(ctx, output)))
}

export async function writeSemanticWorkerAssets(ctx: BuildCtx): Promise<FilePath[]> {
  const outdir = path.join(ctx.argv.output, staticScriptsDir)
  const worker = await bundle({
    entryPoints: { 'semantic.worker': semanticWorkerEntry },
    bundle: true,
    minify: true,
    platform: 'browser',
    format: 'esm',
    splitting: true,
    outdir,
    entryNames: '[name]',
    chunkNames: 'chunks/[name]-[hash]',
    write: false,
  })
  return await Promise.all(worker.outputFiles.map(output => writeAssetBundleOutput(ctx, output)))
}

export async function writeLexicalWorkerAsset(ctx: BuildCtx): Promise<FilePath> {
  const result = await bundle({
    entryPoints: [lexicalWorkerEntry],
    bundle: true,
    minify: true,
    platform: 'browser',
    format: 'esm',
    write: false,
  })
  return writeRawAsset(ctx, lexicalWorkerPath, result.outputFiles[0].text)
}

async function writeGenericWorkerAsset(ctx: BuildCtx, src: string): Promise<FilePath> {
  const result = await bundle({
    entryPoints: [src],
    bundle: true,
    minify: true,
    platform: 'browser',
    format: 'esm',
    write: false,
  })
  const code = result.outputFiles[0].text
  const name = path.basename(src).replace(/\.ts$/, '')
  return write({ ctx, slug: name as FullSlug, ext: '.js', content: code })
}

export async function* writeGenericWorkerAssets(ctx: BuildCtx): AsyncGenerator<FilePath> {
  const workerFiles = (await globby([workerEntryPattern])).filter(
    src => src !== semanticWorkerEntry && src !== speechWorkerEntry && src !== lexicalWorkerEntry,
  )
  for (const src of workerFiles) {
    yield writeGenericWorkerAsset(ctx, src)
  }
}
