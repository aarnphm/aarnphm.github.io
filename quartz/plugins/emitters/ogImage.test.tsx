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
import { createOgImageGenerator } from './ogImage'

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
