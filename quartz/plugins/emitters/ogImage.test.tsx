import assert from 'node:assert/strict'
import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'
import sharp from 'sharp'
import type { QuartzConfig } from '../../cfg'
import type { BuildCtx } from '../../util/ctx'
import { isFullSlug, type FullSlug } from '../../util/path'
import { defaultProcessedContent } from '../vfile'
import { CustomOgImages, createOgImageGenerator } from './ogImage'

const fullSlug = (value: string): FullSlug => {
  if (!isFullSlug(value)) throw new Error(`invalid slug: ${value}`)
  return value
}

const colors = {
  light: '#fff',
  lightgray: '#ddd',
  gray: '#999',
  darkgray: '#555',
  dark: '#111',
  secondary: '#ced697',
  tertiary: '#fcc192',
  highlight: '#dad8ce',
  textHighlight: '#f1d67e',
}

const config: QuartzConfig = {
  configuration: {
    pageTitle: "Aaron's notes",
    enableSPA: true,
    enablePopovers: true,
    analytics: null,
    ignorePatterns: [],
    defaultDateType: 'created',
    baseUrl: 'aarnphm.xyz',
    locale: 'en-US',
    theme: {
      cdnCaching: false,
      fontOrigin: 'local',
      typography: { header: 'Test Header', body: 'Test Body', code: 'Test Code' },
      colors: { lightMode: colors, darkMode: colors },
    },
  },
  plugins: { transformers: [], filters: [], emitters: [] },
}

test('generates a social image for a synthetic triathlon subpage', async () => {
  const output = await fs.mkdtemp(path.join(os.tmpdir(), 'triathlon-og-'))
  const slug = fullSlug('triathlon/analytics')
  const ctx: BuildCtx = {
    buildId: 'triathlon-og-test',
    argv: {
      directory: 'content',
      verbose: false,
      output,
      serve: false,
      watch: false,
      port: 0,
      wsPort: 0,
      force: true,
    },
    cfg: config,
    allSlugs: [slug],
    allFiles: [],
    incremental: false,
  }
  const [, file] = defaultProcessedContent({
    slug,
    text: 'Generated training analytics across 154 activities.',
    frontmatter: {
      title: 'triathlon · analytics',
      pageLayout: 'default',
      description: 'Generated training analytics across 154 activities.',
      generatedSocialImage: true,
    },
  })

  try {
    const generateOgImage = await createOgImageGenerator(ctx)
    const generated = await generateOgImage(file.data)
    const metadata = await sharp(await fs.readFile(generated)).metadata()

    assert.equal(generated, path.join(output, 'triathlon/analytics-og-image.webp'))
    assert.equal(metadata.format, 'webp')
    assert.equal(metadata.width, 1200)
    assert.equal(metadata.height, 630)
  } finally {
    await fs.rm(output, { recursive: true, force: true })
  }
})

test('reuses cached og image when volatile published or modified dates change across builds', async () => {
  const output = await fs.mkdtemp(path.join(os.tmpdir(), 'og-cache-stability-'))
  const slug = fullSlug('thoughts/cache-stability-probe')
  const ctx: BuildCtx = {
    buildId: 'og-cache-stability',
    argv: {
      directory: 'content',
      verbose: false,
      output,
      serve: false,
      watch: false,
      port: 0,
      wsPort: 0,
      force: true,
    },
    cfg: config,
    allSlugs: [slug],
    allFiles: [],
    incremental: false,
  }

  const created = new Date('2026-10-01T12:00:00.000Z')
  const [, firstBuildFile] = defaultProcessedContent({
    slug,
    text: 'Deterministic social card cache test body.',
    dates: {
      created,
      modified: new Date('2026-10-18T01:00:00.111Z'),
      published: new Date('2026-10-18T01:00:00.222Z'),
    },
    frontmatter: {
      title: 'cache stability probe',
      pageLayout: 'default',
      description: 'Deterministic social card cache test body.',
      tags: ['perf', 'cache', 'og', 'ignored-fourth-tag'],
    },
  })

  const [, secondBuildFile] = defaultProcessedContent({
    slug,
    text: 'Deterministic social card cache test body.',
    dates: {
      created: new Date('2026-10-01T18:45:12.987Z'),
      modified: new Date('2026-10-19T05:12:34.567Z'),
      published: new Date('2026-10-19T05:12:34.890Z'),
    },
    frontmatter: {
      title: 'cache stability probe',
      pageLayout: 'default',
      description: 'Deterministic social card cache test body.',
      tags: ['perf', 'cache', 'og', 'different-fourth-tag'],
      aliases: ['probe-alias'],
    },
  })

  try {
    const cacheDir = path.join('quartz', '.quartz-cache', 'og-images')
    const generateOgImage = await createOgImageGenerator(ctx)
    const firstPath = await generateOgImage(firstBuildFile.data)
    const firstBytes = await fs.readFile(firstPath)
    const cacheEntriesAfterFirst = new Set(await fs.readdir(cacheDir))

    const secondCtx: BuildCtx = {
      ...ctx,
      argv: { ...ctx.argv, output: path.join(output, 'second') },
    }
    const generateSecond = await createOgImageGenerator(secondCtx)
    const secondPath = await generateSecond(secondBuildFile.data)
    const secondBytes = await fs.readFile(secondPath)
    const cacheEntriesAfterSecond = await fs.readdir(cacheDir)
    const newlyCreatedCacheFiles = cacheEntriesAfterSecond.filter(
      entry => !cacheEntriesAfterFirst.has(entry),
    )

    assert.deepEqual(newlyCreatedCacheFiles, [])
    assert.deepEqual(secondBytes, firstBytes)
  } finally {
    await fs.rm(output, { recursive: true, force: true })
  }
})

test('CustomOgImages.emit generates multiple social images concurrently and skips explicit socialImage', async () => {
  const output = await fs.mkdtemp(path.join(os.tmpdir(), 'og-concurrent-emit-'))
  const slugs = [
    fullSlug('thoughts/concurrent-a'),
    fullSlug('thoughts/concurrent-b'),
    fullSlug('thoughts/concurrent-c'),
    fullSlug('thoughts/custom-social'),
  ]
  const ctx: BuildCtx = {
    buildId: 'og-concurrent-emit',
    argv: {
      directory: 'content',
      verbose: false,
      output,
      serve: false,
      watch: false,
      port: 0,
      wsPort: 0,
      force: true,
    },
    cfg: config,
    allSlugs: slugs,
    allFiles: [],
    incremental: false,
  }

  const content = [
    defaultProcessedContent({
      slug: slugs[0],
      text: 'Shared concurrent card body.',
      frontmatter: {
        title: 'concurrent card',
        pageLayout: 'default',
        description: 'Shared concurrent card body.',
      },
    }),
    defaultProcessedContent({
      slug: slugs[1],
      text: 'Shared concurrent card body.',
      frontmatter: {
        title: 'concurrent card',
        pageLayout: 'default',
        description: 'Shared concurrent card body.',
      },
    }),
    defaultProcessedContent({
      slug: slugs[2],
      text: 'Distinct concurrent card body with extra words.',
      frontmatter: {
        title: 'distinct card',
        pageLayout: 'default',
        description: 'Distinct concurrent card body.',
      },
    }),
    defaultProcessedContent({
      slug: slugs[3],
      text: 'Has explicit socialImage.',
      frontmatter: {
        title: 'explicit social image',
        pageLayout: 'default',
        socialImage: 'https://aarnphm.xyz/static/custom.png',
      },
    }),
  ]

  try {
    const emitter = CustomOgImages()
    const emitted: string[] = []
    const outputIter = await emitter.emit(ctx, content, { css: [], js: [], additionalHead: [] })
    if (Symbol.asyncIterator in outputIter) {
      for await (const file of outputIter) {
        emitted.push(file)
      }
    } else {
      emitted.push(...outputIter)
    }

    assert.equal(emitted.length, 3)
    for (const file of emitted) {
      const metadata = await sharp(await fs.readFile(file)).metadata()
      assert.equal(metadata.format, 'webp')
    }
  } finally {
    await fs.rm(output, { recursive: true, force: true })
  }
})
