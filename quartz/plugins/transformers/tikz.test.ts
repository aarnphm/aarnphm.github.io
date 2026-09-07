import assert from 'node:assert'
import { mkdtemp, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'
import { cachedTikzSvg, normalizeComputerModernText, type TikzRenderOptions } from './tikz'

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

test('caches tikz render failures in memory', async t => {
  const cacheDir = await mkdtemp(path.join(os.tmpdir(), 'quartz-tikz-cache-'))
  t.after(() => rm(cacheDir, { recursive: true, force: true }))
  let renders = 0

  const render = async () => {
    renders += 1
    throw new Error(`render failed ${renders}`)
  }

  await assert.rejects(cachedTikzSvg('broken source', options, render, cacheDir), /render failed 1/)
  await assert.rejects(
    cachedTikzSvg('broken source', options, render, cacheDir),
    /cached render failure: render failed 1/,
  )
  assert.strictEqual(renders, 1)
})

test('normalizes Computer Modern set operators in tikz text', () => {
  const svg = [
    '<svg>',
    '<text font-family="cmsy10">[</text>',
    '<text font-family="cmsy10">\\</text>',
    '</svg>',
  ].join('')

  assert.strictEqual(
    normalizeComputerModernText(svg),
    [
      '<svg>',
      '<text font-family="serif">∪</text>',
      '<text font-family="serif">∩</text>',
      '</svg>',
    ].join(''),
  )
})

test('normalizes absolute-value bars and the emitted leq glyph in tikz labels', () => {
  // node-tikzjax emits these font-encoded characters for $|x| \\leq L$.
  const svg = [
    '<svg>',
    '<text font-family="cmsy10" x="0">j</text>',
    '<text font-family="cmmi10" x="3">x</text>',
    '<text font-family="cmsy10" x="9">&#106;</text>',
    '<text font-family="cmsy10" x="15">∙</text>',
    '<text font-family="cmmi10" x="25">L</text>',
    '<text font-family="cmbsy10">&#x2219;j</text>',
    '</svg>',
  ].join('')

  assert.strictEqual(
    normalizeComputerModernText(svg),
    [
      '<svg>',
      '<text font-family="serif" x="0">|</text>',
      '<text font-family="serif" x="3" font-style="italic">x</text>',
      '<text font-family="serif" x="9">|</text>',
      '<text font-family="serif" x="15">≤</text>',
      '<text font-family="serif" x="25" font-style="italic">L</text>',
      '<text font-family="serif">≤|</text>',
      '</svg>',
    ].join(''),
  )
})

test('preserves literal j and dot characters outside Computer Modern symbol fonts', () => {
  assert.strictEqual(
    normalizeComputerModernText(
      '<svg><text font-family="cmr10">j</text><text font-family="cmmi10">j</text><text font-family="serif">j∙</text></svg>',
    ),
    '<svg><text font-family="serif">j</text><text font-family="serif" font-style="italic">j</text><text font-family="serif">j∙</text></svg>',
  )
})
