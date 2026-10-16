import assert from 'node:assert/strict'
import { mkdir, mkdtemp, rm, stat, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import test from 'node:test'
import type { QuartzConfig } from '../cfg'
import type { BuildCtx } from './ctx'
import type { FilePath } from './path'
import {
  cleanOutputExcept,
  emitOutputAsset,
  getPreservedOutputFiles,
  preservedOutputAssets,
  pruneOutputAssetManifest,
  recordOutputAsset,
  removeOutputAsset,
  replaceOutputAssets,
  resetOutputAssetClaims,
  type OutputAssetClaim,
} from './output-assets'

const testConfig = {
  configuration: {
    pageTitle: 'test',
    enableSPA: true,
    enablePopovers: true,
    analytics: null,
    ignorePatterns: [],
    defaultDateType: 'created',
    theme: {} as QuartzConfig['configuration']['theme'],
    locale: 'en-US',
  },
  plugins: { transformers: [], filters: [], emitters: [] },
} satisfies QuartzConfig

function ctx(root: string): BuildCtx {
  return {
    buildId: 'test',
    argv: {
      directory: path.join(root, 'content'),
      verbose: false,
      output: path.join(root, 'public'),
      serve: false,
      watch: false,
      port: 8080,
      wsPort: 3001,
      force: false,
    },
    cfg: testConfig,
    allSlugs: [],
    allFiles: [],
    incremental: false,
  }
}

async function source(root: string, relativePath: string, content: string): Promise<FilePath> {
  const file = path.join(root, 'content', relativePath) as FilePath
  await mkdir(path.dirname(file), { recursive: true })
  await writeFile(file, content)
  return file
}

function claim(buildCtx: BuildCtx, sourcePath: FilePath, outputPath: string): OutputAssetClaim {
  return {
    owner: 'content-asset',
    source: sourcePath,
    output: path.join(buildCtx.argv.output, outputPath) as FilePath,
  }
}

test('cleanOutputExcept preserves unchanged claimed assets and removes generated files', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'quartz-output-assets-clean-'))
  try {
    resetOutputAssetClaims()
    const buildCtx = ctx(root)
    const imageSource = await source(root, 'media/a.png', 'asset')
    const imageClaim = claim(buildCtx, imageSource, 'media/a.png')
    const imageOutput = await emitOutputAsset(buildCtx, imageClaim)
    const generated = path.join(buildCtx.argv.output, 'index.html')
    const stale = path.join(buildCtx.argv.output, 'media/old.png')
    await mkdir(path.dirname(generated), { recursive: true })
    await mkdir(path.dirname(stale), { recursive: true })
    await writeFile(generated, 'generated')
    await writeFile(stale, 'stale')

    const preserved = await preservedOutputAssets(buildCtx.outputAssetManifest!, [imageClaim])
    await cleanOutputExcept(buildCtx.argv.output, preserved)

    assert.equal((await stat(imageOutput)).isFile(), true)
    await assert.rejects(stat(generated))
    await assert.rejects(stat(stale))
  } finally {
    resetOutputAssetClaims()
    await rm(root, { recursive: true, force: true })
  }
})

test('renamed output asset removes the old output and records the new output', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'quartz-output-assets-rename-'))
  try {
    resetOutputAssetClaims()
    const buildCtx = ctx(root)
    const oldSource = await source(root, 'media/old.png', 'old')
    const oldClaim = claim(buildCtx, oldSource, 'media/old.png')
    const oldOutput = await emitOutputAsset(buildCtx, oldClaim)
    const newSource = await source(root, 'media/new.png', 'new asset')
    const newClaim = claim(buildCtx, newSource, 'media/new.png')

    const preserved = await preservedOutputAssets(buildCtx.outputAssetManifest!, [newClaim])
    await cleanOutputExcept(buildCtx.argv.output, preserved)
    await emitOutputAsset(buildCtx, newClaim)
    pruneOutputAssetManifest(buildCtx.outputAssetManifest!, [newClaim])

    await assert.rejects(stat(oldOutput))
    const newOutput = path.join(buildCtx.argv.output, 'media/new.png')
    assert.equal((await stat(newOutput)).size, Buffer.byteLength('new asset'))
    assert.equal(buildCtx.outputAssetManifest!.size, 1)
  } finally {
    resetOutputAssetClaims()
    await rm(root, { recursive: true, force: true })
  }
})

test('preserves notebook-runtime and notebook-asset claims across hard rebuilds', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'quartz-output-assets-notebook-'))
  try {
    resetOutputAssetClaims()
    const buildCtx = ctx(root)
    const runtimeFile = path.join(
      buildCtx.argv.output,
      'static/scripts/notebook-runtimes/manifest.json',
    ) as FilePath
    const inlineAssetFile = path.join(
      buildCtx.argv.output,
      'static/notebook-assets/deadbeef.png',
    ) as FilePath
    const htmlFile = path.join(buildCtx.argv.output, 'notes/page.html')
    await mkdir(path.dirname(runtimeFile), { recursive: true })
    await mkdir(path.dirname(inlineAssetFile), { recursive: true })
    await mkdir(path.dirname(htmlFile), { recursive: true })
    await writeFile(runtimeFile, '{"version":1,"runtimes":{}}')
    await writeFile(inlineAssetFile, 'png-bytes')
    await writeFile(htmlFile, '<html></html>')

    recordOutputAsset('notebook-runtime', [runtimeFile])
    recordOutputAsset('notebook-asset', [inlineAssetFile])

    assert.equal(getPreservedOutputFiles().has(runtimeFile), true)
    assert.equal(getPreservedOutputFiles().has(inlineAssetFile), true)

    const preserved = await preservedOutputAssets(new Map(), [])
    await cleanOutputExcept(buildCtx.argv.output, preserved)

    assert.equal((await stat(runtimeFile)).isFile(), true)
    assert.equal((await stat(inlineAssetFile)).isFile(), true)
    await assert.rejects(stat(htmlFile))

    replaceOutputAssets('notebook-runtime', [])
    assert.equal(getPreservedOutputFiles().has(runtimeFile), false)
    await removeOutputAsset('notebook-asset', [inlineAssetFile])
    assert.equal(getPreservedOutputFiles().has(inlineAssetFile), false)
  } finally {
    resetOutputAssetClaims()
    await rm(root, { recursive: true, force: true })
  }
})
