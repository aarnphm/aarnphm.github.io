import assert from 'node:assert'
import { mkdtemp, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'
import { cachedTikzSvg, type TikzRenderOptions } from './tikz'

const options: TikzRenderOptions = {
  showConsole: false,
  disableSanitize: false,
  disableOptimize: true,
}

test('caches tikz svg renders by source and options', async t => {
  const cacheDir = await mkdtemp(path.join(os.tmpdir(), 'quartz-tikz-cache-'))
  t.after(() => rm(cacheDir, { recursive: true, force: true }))
  let renders = 0

  const render = async () => {
    renders += 1
    return `<svg>${renders}</svg>`
  }

  const first = await cachedTikzSvg('source', options, render, cacheDir)
  const second = await cachedTikzSvg('source', options, render, cacheDir)

  assert.strictEqual(first, '<svg>1</svg>')
  assert.strictEqual(second, '<svg>1</svg>')
  assert.strictEqual(renders, 1)
})

test('separates tikz cache entries by render options', async t => {
  const cacheDir = await mkdtemp(path.join(os.tmpdir(), 'quartz-tikz-cache-'))
  t.after(() => rm(cacheDir, { recursive: true, force: true }))
  let renders = 0

  const render = async () => {
    renders += 1
    return `<svg>${renders}</svg>`
  }

  const sanitized = await cachedTikzSvg('source', options, render, cacheDir)
  const unsanitized = await cachedTikzSvg(
    'source',
    { ...options, disableSanitize: true },
    render,
    cacheDir,
  )

  assert.strictEqual(sanitized, '<svg>1</svg>')
  assert.strictEqual(unsanitized, '<svg>2</svg>')
  assert.strictEqual(renders, 2)
})
