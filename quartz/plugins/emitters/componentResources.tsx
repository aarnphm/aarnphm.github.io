import { transform as transpile, build as bundle } from 'esbuild'
import { globby } from 'globby'
import { Features, transform } from 'lightningcss'
import fs from 'node:fs/promises'
import path from 'path'
import type { QuartzMdxComponent } from '../../components/mdx/registry'
import type { StaticResources } from '../../util/resources'
import { getMdxComponents } from '../../components/mdx/registry'
// @ts-ignore
import notFoundScript from '../../components/scripts/404.inline'
//@ts-ignore
import audioScript from '../../components/scripts/audio.inline'
// @ts-ignore
import baseMapScript from '../../components/scripts/base-map.inline'
// @ts-ignore
import pseudoScript from '../../components/scripts/clipboard-pseudo.inline'
// @ts-ignore
import clipboardScript from '../../components/scripts/clipboard.inline'
// @ts-ignore
import collaborativeCommentsScript from '../../components/scripts/collaborative-comments.inline'
// @ts-ignore
import collapseHeaderScript from '../../components/scripts/collapse-header.inline'
// @ts-ignore
import markerScript from '../../components/scripts/marker.inline'
// @ts-ignore
import petScript from '../../components/scripts/pet.inline'
// @ts-ignore
import popoverScript from '../../components/scripts/popover.inline'
//@ts-ignore
import protectedScript from '../../components/scripts/protected.inline'
// @ts-ignore
import spaRouterScript from '../../components/scripts/spa.inline'
// @ts-ignore
import transcludeScript from '../../components/scripts/transclude.inline.ts'
import audioStyle from '../../components/styles/audio.scss'
import clipboardStyle from '../../components/styles/clipboard.scss'
import collapseHeaderStyle from '../../components/styles/collapseHeader.inline.scss'
import popoverStyle from '../../components/styles/popover.scss'
import pseudoStyle from '../../components/styles/pseudocode.scss'
import '../../components/mdx'
import baseStyles from '../../styles/base.scss'
import customStyles from '../../styles/custom.scss'
import { QuartzComponent } from '../../types/component'
import { QuartzEmitterPlugin } from '../../types/plugin'
import {
  assetManifestRecord,
  assetPath,
  assetSlugForContent,
  contentHashSlug,
  registerExtractedStaticResource,
  resolveAsset,
} from '../../util/asset-manifest'
import { BuildCtx } from '../../util/ctx'
import { FilePath, FullSlug, isFullSlug, joinSegments } from '../../util/path'
import {
  splitCssBundles,
  splitJsBundles,
  componentCssResourceKey,
  staticCssBundleKey,
  staticCssBundleSlug,
  staticJsBundleKey,
  staticJsBundleSlug,
} from '../../util/resource-bundles'
import { googleFontHref, joinStyles, processGoogleFonts } from '../../util/theme'
import { isWorkerEntryPath, workerEntryPattern } from '../../util/workers'
import { write } from './helpers'

const name = 'ComponentResources'
const collaborativeCommentsClientEntry =
  'quartz/components/scripts/collaborative-comments.client.ts'
const notebookRuntimeInlineEntry = 'quartz/components/scripts/notebook-runtime.inline.ts'
const notebookRuntimeClientEntry = 'quartz/components/scripts/notebook-runtime.client.ts'
const notebookRuntimeWorkerEntry = 'quartz/components/scripts/notebook-runtime.pyodide.js'
const notebookRuntimeBootstrapEntry = 'quartz/components/scripts/notebook-runtime.pyodide.py'
const notebookRuntimeMlBridgeEntry = 'quartz/components/scripts/notebook-runtime.ml.js'
const emojiAssetSourceDir = 'quartz/util/emojimap'
const notebookRuntimeAssetEntries = new Set([
  notebookRuntimeClientEntry,
  notebookRuntimeWorkerEntry,
  notebookRuntimeBootstrapEntry,
  notebookRuntimeMlBridgeEntry,
  'quartz/util/notebook-runtime.ts',
  'quartz/components/scripts/notebook-code-editor.ts',
])

type ComponentResources = {
  css: string[]
  componentCss: string[]
  beforeDOMLoaded: string[]
  afterDOMLoaded: string[]
}

export function normalizeResource(resource: string | string[] | undefined): string[] {
  if (!resource) return []
  if (Array.isArray(resource)) return resource
  return [resource]
}

function getComponentResources(ctx: BuildCtx): ComponentResources {
  const allComponents: Set<QuartzComponent | QuartzMdxComponent> = new Set()
  for (const emitter of ctx.cfg.plugins.emitters) {
    const components = emitter.getQuartzComponents?.(ctx) ?? []
    for (const component of components) {
      allComponents.add(component)
    }
  }
  for (const component of getMdxComponents()) {
    allComponents.add(component)
  }

  const componentResources = {
    css: new Set<string>(),
    beforeDOMLoaded: new Set<string>(),
    afterDOMLoaded: new Set<string>(),
  }

  for (const component of allComponents) {
    const { css, beforeDOMLoaded, afterDOMLoaded } = component
    const normalizedCss = normalizeResource(css)
    const normalizedBeforeDOMLoaded = normalizeResource(beforeDOMLoaded)
    const normalizedAfterDOMLoaded = normalizeResource(afterDOMLoaded)

    normalizedCss.forEach(c => componentResources.css.add(c))
    normalizedBeforeDOMLoaded.forEach(b => componentResources.beforeDOMLoaded.add(b))
    normalizedAfterDOMLoaded.forEach(a => componentResources.afterDOMLoaded.add(a))
  }

  return {
    css: [...componentResources.css],
    componentCss: [...componentResources.css],
    beforeDOMLoaded: [...componentResources.beforeDOMLoaded],
    afterDOMLoaded: [...componentResources.afterDOMLoaded],
  }
}

async function joinScripts(scripts: string[]): Promise<string> {
  // wrap with iife to prevent scope collision
  const script = scripts.map(script => `(function () {${script}})();`).join('\n')

  // minify with esbuild
  const res = await transpile(script, { minify: true })

  return res.code
}

function notebookRuntimeInlineResourceIndex(componentResources: ComponentResources): number {
  return componentResources.afterDOMLoaded.findIndex(
    script =>
      script.includes('notebookRuntimeScriptUrl') && script.includes('data-notebook-runtime-data'),
  )
}

async function refreshNotebookRuntimeInlineResource(componentResources: ComponentResources) {
  const index = notebookRuntimeInlineResourceIndex(componentResources)
  if (index < 0) return
  componentResources.afterDOMLoaded[index] = await fs.readFile(notebookRuntimeInlineEntry, 'utf8')
}

async function currentComponentResources(ctx: BuildCtx): Promise<ComponentResources> {
  const componentResources = getComponentResources(ctx)
  await refreshNotebookRuntimeInlineResource(componentResources)
  addGlobalPageResources(ctx, componentResources)
  return componentResources
}

async function writeAssetBundleOutput(ctx: BuildCtx, outputFile: { path: string; text: string }) {
  const rel = path.relative(ctx.argv.output, outputFile.path).split(path.sep).join('/')
  const ext = path.extname(rel) as `.${string}`
  const logicalSlug = rel.slice(0, -ext.length)
  const slug = rel.includes('/chunks/')
    ? (logicalSlug as FullSlug)
    : assetSlugForContent(ctx, logicalSlug, ext, outputFile.text)
  return write({ ctx, slug, ext, content: outputFile.text })
}

async function writeAssetManifest(ctx: BuildCtx): Promise<FilePath> {
  return write({
    ctx,
    slug: 'static/scripts/asset-manifest' as FullSlug,
    ext: '.json',
    content: JSON.stringify(assetManifestRecord(ctx)),
  })
}

function minifyStylesheet(filename: string, stylesheet: string) {
  return transform({
    filename,
    code: Buffer.from(stylesheet),
    minify: true,
    targets: {
      safari: (15 << 16) | (6 << 8),
      ios_saf: (15 << 16) | (6 << 8),
      edge: 115 << 16,
      firefox: 102 << 16,
      chrome: 109 << 16,
    },
    include: Features.MediaQueries,
  }).code.toString()
}

async function* writeStaticResourceBundles(ctx: BuildCtx, resources: StaticResources) {
  for (const part of splitCssBundles(resources.css, [collapseHeaderStyle])) {
    if (part.type !== 'bundle') continue
    const content = minifyStylesheet('resource-style.css', part.content)
    const slug = contentHashSlug(staticCssBundleSlug, content)
    registerExtractedStaticResource(ctx, staticCssBundleKey(part.content), assetPath(slug, '.css'))
    yield write({ ctx, slug, ext: '.css', content })
  }

  for (const loadTime of ['beforeDOMReady', 'afterDOMReady'] as const) {
    const leadingInline =
      loadTime === 'afterDOMReady' ? [transcludeScript, collapseHeaderScript] : []
    for (const part of splitJsBundles(resources.js, loadTime, leadingInline)) {
      if (part.type !== 'bundle') continue
      const content = await joinScripts(part.scripts)
      const slug = contentHashSlug(staticJsBundleSlug(part.loadTime), content)
      registerExtractedStaticResource(
        ctx,
        staticJsBundleKey(part.loadTime, part.scripts),
        assetPath(slug, '.js'),
      )
      yield write({ ctx, slug, ext: '.js', content })
    }
  }
}

async function* writeComponentStyles(ctx: BuildCtx, resources: ComponentResources) {
  for (const stylesheet of resources.componentCss) {
    const content = minifyStylesheet('component.css', `@layer quartz-base {\n${stylesheet}\n}`)
    const slug = contentHashSlug('component', content)
    registerExtractedStaticResource(
      ctx,
      componentCssResourceKey(stylesheet),
      assetPath(slug, '.css'),
    )
    yield write({ ctx, slug, ext: '.css', content })
  }
}

async function writeAfterDomLoadedScripts(ctx: BuildCtx, scripts: string[]) {
  const entries = await Promise.all(
    scripts.map(async script => {
      const content = await joinScripts([script])
      const slug = contentHashSlug('static/scripts/script', content)
      return {
        filename: assetPath(slug, '.js'),
        file: await write({ ctx, slug, ext: '.js', content }),
      }
    }),
  )

  const postscript = entries.map(({ filename }) => `await import("./${filename}");`).join('\n')

  return { postscript, files: entries.map(({ file }) => file) }
}

async function writeNotebookRuntimeAssets(ctx: BuildCtx): Promise<FilePath[]> {
  const outdir = path.join(ctx.argv.output, 'static/scripts')
  const worker = await bundle({
    entryPoints: [notebookRuntimeWorkerEntry],
    bundle: true,
    minify: true,
    platform: 'browser',
    format: 'esm',
    outfile: path.join(outdir, 'notebook-runtime.worker.js'),
    define: { 'globalThis.process': 'undefined' },
    external: ['fs'],
    loader: { '.py': 'text' },
    write: false,
  })
  const workerFiles = await Promise.all(
    worker.outputFiles.map(output => writeAssetBundleOutput(ctx, output)),
  )
  const workerName = path.basename(resolveAsset(ctx, 'static/scripts/notebook-runtime.worker.js'))
  const client = await bundle({
    entryPoints: { 'notebook-runtime.client': notebookRuntimeClientEntry },
    bundle: true,
    minify: true,
    platform: 'browser',
    format: 'esm',
    splitting: true,
    outdir,
    entryNames: '[name]',
    chunkNames: 'chunks/[name]-[hash]',
    loader: { '.html': 'text' },
    write: false,
  })
  const clientOutputs = client.outputFiles.map(output => ({
    ...output,
    text: output.text.replaceAll('notebook-runtime.worker.js', workerName),
  }))
  const clientFiles = await Promise.all(
    clientOutputs.map(output => writeAssetBundleOutput(ctx, output)),
  )
  return [...workerFiles, ...clientFiles]
}

async function writeCollaborativeCommentsAssets(ctx: BuildCtx): Promise<FilePath[]> {
  const outdir = path.join(ctx.argv.output, 'static/scripts')
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

async function writeEmojiAssets(ctx: BuildCtx): Promise<FilePath[]> {
  const files = await globby([`${emojiAssetSourceDir}/**/*.json`])
  return await Promise.all(
    files.map(async file => {
      const rel = path.relative(emojiAssetSourceDir, file).split(path.sep).join('/')
      const slug = joinSegments('static', 'scripts', 'emoji', rel.slice(0, -'.json'.length))
      if (!isFullSlug(slug)) throw new Error(`invalid emoji asset slug ${slug}`)
      const content = await fs.readFile(file)
      return await write({
        ctx,
        slug: assetSlugForContent(ctx, slug, '.json', content),
        ext: '.json',
        content,
      })
    }),
  )
}

function assetBasename(ctx: BuildCtx, logicalPath: string): string {
  return path.basename(resolveAsset(ctx, logicalPath))
}

function resolveComponentResourceAssets(ctx: BuildCtx, componentResources: ComponentResources) {
  componentResources.afterDOMLoaded = componentResources.afterDOMLoaded.map(script =>
    script
      .replaceAll(
        'notebook-runtime.client.js',
        assetBasename(ctx, 'static/scripts/notebook-runtime.client.js'),
      )
      .replaceAll(
        'collaborative-comments.client.js',
        assetBasename(ctx, 'static/scripts/collaborative-comments.client.js'),
      ),
  )
}

function isNotebookRuntimeAssetChange(changePath: string): boolean {
  return notebookRuntimeAssetEntries.has(changePath)
}

function isNotebookRuntimePageScriptChange(changePath: string): boolean {
  return changePath === notebookRuntimeInlineEntry
}

function isCollaborativeCommentsAssetChange(changePath: string): boolean {
  return (
    changePath === collaborativeCommentsClientEntry ||
    changePath === 'quartz/components/scripts/markdown-editor.ts' ||
    changePath === 'quartz/components/scripts/search-index.ts' ||
    changePath.startsWith('quartz/components/multiplayer/') ||
    changePath.startsWith('quartz/functional/')
  )
}

function isEmojiAssetChange(changePath: string): boolean {
  return changePath.startsWith(`${emojiAssetSourceDir}/`) && changePath.endsWith('.json')
}

function addGlobalPageResources(ctx: BuildCtx, componentResources: ComponentResources) {
  const cfg = ctx.cfg.configuration
  const resolvedPetScript = petScript.replace(
    '__QUARTZ_PETS_DEFAULT_ENABLED__',
    ctx.argv.watch || ctx.argv.serve ? '0' : '1',
  )

  // popovers
  if (cfg.enablePopovers) {
    componentResources.afterDOMLoaded.push(popoverScript)
    componentResources.css.push(popoverStyle)
  }

  componentResources.beforeDOMLoaded.push(markerScript)

  componentResources.css.push(clipboardStyle, pseudoStyle, audioStyle)
  componentResources.afterDOMLoaded.push(
    clipboardScript,
    pseudoScript,
    protectedScript,
    audioScript,
    baseMapScript,
    collaborativeCommentsScript,
    resolvedPetScript,
  )

  if (cfg.analytics?.provider === 'plausible') {
    const plausibleHost = cfg.analytics.host ?? 'https://plausible.io'
    componentResources.afterDOMLoaded.push(`
      const plausibleScript = document.createElement("script")
      plausibleScript.src = "${plausibleHost}/js/script.outbound-links.manual.js"
      plausibleScript.setAttribute("data-domain", [location.hostname, "stream.aarnphm.xyz"].join(','))
      plausibleScript.setAttribute("data-api", "/_plausible/event")
      plausibleScript.defer = true
      plausibleScript.onload = () => {
        window.plausible = window.plausible || function () { (window.plausible.q = window.plausible.q || []).push(arguments); };
        plausible('pageview')
        document.addEventListener('nav', () => {
          plausible('pageview')
        })
      }

      document.head.appendChild(plausibleScript)
    `)
  }

  componentResources.afterDOMLoaded.push(notFoundScript, spaRouterScript)
}

export const ComponentResources: QuartzEmitterPlugin = () => {
  return {
    name,
    async *emit(ctx, _content, resources) {
      const cfg = ctx.cfg.configuration
      // component specific scripts and styles
      const componentResources = await currentComponentResources(ctx)
      const notebookRuntimeFiles = await writeNotebookRuntimeAssets(ctx)
      const collaborativeCommentsFiles = await writeCollaborativeCommentsAssets(ctx)
      const emojiFiles = await writeEmojiAssets(ctx)
      resolveComponentResourceAssets(ctx, componentResources)
      let googleFontsStyleSheet = ''
      if (cfg.theme.fontOrigin === 'local') {
        // let the user do it themselves in css
      } else if (cfg.theme.fontOrigin === 'googleFonts' && !cfg.theme.cdnCaching) {
        const response = await fetch(googleFontHref(ctx.cfg.configuration.theme))
        googleFontsStyleSheet = await response.text()

        if (!cfg.baseUrl) {
          throw new Error(
            'baseUrl must be defined when using Google Fonts without cfg.theme.cdnCaching',
          )
        }

        const { processedStylesheet, fontFiles } = await processGoogleFonts(
          googleFontsStyleSheet,
          cfg.baseUrl,
        )
        googleFontsStyleSheet = processedStylesheet

        // Download and save font files
        for (const fontFile of fontFiles) {
          const res = await fetch(fontFile.url)
          if (!res.ok) {
            throw new Error(`failed to fetch font ${fontFile.filename}`)
          }

          const buf = await res.arrayBuffer()
          yield write({
            ctx,
            slug: joinSegments('static', 'fonts', fontFile.filename) as FullSlug,
            ext: `.${fontFile.extension}`,
            content: Buffer.from(buf),
          })
        }
      }

      yield* writeComponentStyles(ctx, componentResources)

      const componentCss = new Set(componentResources.componentCss)
      const quartzBase = joinStyles(
        ctx.cfg.configuration.theme,
        googleFontsStyleSheet,
        ...componentResources.css.filter(css => !componentCss.has(css)),
        baseStyles,
      )
      const stylesheet = `@layer quartz-base {\n${quartzBase}\n}\n${customStyles}`
      const [prescript, postscriptResult] = await Promise.all([
        joinScripts(componentResources.beforeDOMLoaded),
        writeAfterDomLoadedScripts(ctx, componentResources.afterDOMLoaded),
      ])
      const { postscript, files: postscriptFiles } = postscriptResult

      const manifest = {
        name: cfg.pageTitle,
        short_name: cfg.baseUrl,
        icons: [
          { src: '/android-chrome-192x192.png', sizes: '192x192', type: 'image/png' },
          { src: '/android-chrome-512x512.png', sizes: '512x512', type: 'image/png' },
        ],
        theme_color: cfg.theme.colors['lightMode'].light,
        background_color: cfg.theme.colors['lightMode'].light,
        display: 'standalone',
        lang: cfg.locale,
        dir: 'auto',
      }

      const stylesheetContent = minifyStylesheet('index.css', stylesheet)
      yield write({
        ctx,
        slug: assetSlugForContent(ctx, 'index', '.css', stylesheetContent),
        ext: '.css',
        content: stylesheetContent,
      })

      yield* writeStaticResourceBundles(ctx, resources)

      yield write({
        ctx,
        slug: assetSlugForContent(ctx, 'prescript', '.js', prescript),
        ext: '.js',
        content: prescript,
      })

      yield write({
        ctx,
        slug: assetSlugForContent(ctx, 'postscript', '.js', postscript),
        ext: '.js',
        content: postscript,
      })

      for (const file of postscriptFiles) {
        yield file
      }

      yield write({
        ctx,
        slug: 'site' as FullSlug,
        ext: '.webmanifest',
        content: JSON.stringify(manifest),
      })

      const workerFiles = await globby([workerEntryPattern])
      for (const src of workerFiles) {
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
        yield write({ ctx, slug: name as FullSlug, ext: '.js', content: code })
      }

      for (const file of notebookRuntimeFiles) {
        yield file
      }

      for (const file of collaborativeCommentsFiles) {
        yield file
      }

      for (const file of emojiFiles) {
        yield file
      }

      yield writeAssetManifest(ctx)
    },
    async *partialEmit(ctx, _content, resources, changeEvents) {
      const componentResources = await currentComponentResources(ctx)
      yield* writeComponentStyles(ctx, componentResources)
      yield* writeStaticResourceBundles(ctx, resources)

      if (changeEvents.some(changeEvent => isNotebookRuntimeAssetChange(changeEvent.path))) {
        for (const file of await writeNotebookRuntimeAssets(ctx)) {
          yield file
        }
      }

      if (changeEvents.some(changeEvent => isNotebookRuntimePageScriptChange(changeEvent.path))) {
        resolveComponentResourceAssets(ctx, componentResources)
        const [prescript, postscriptResult] = await Promise.all([
          joinScripts(componentResources.beforeDOMLoaded),
          writeAfterDomLoadedScripts(ctx, componentResources.afterDOMLoaded),
        ])
        const { postscript, files: postscriptFiles } = postscriptResult

        yield write({
          ctx,
          slug: assetSlugForContent(ctx, 'prescript', '.js', prescript),
          ext: '.js',
          content: prescript,
        })
        yield write({
          ctx,
          slug: assetSlugForContent(ctx, 'postscript', '.js', postscript),
          ext: '.js',
          content: postscript,
        })

        for (const file of postscriptFiles) {
          yield file
        }
      }

      if (changeEvents.some(changeEvent => isCollaborativeCommentsAssetChange(changeEvent.path))) {
        for (const file of await writeCollaborativeCommentsAssets(ctx)) {
          yield file
        }
      }

      if (changeEvents.some(changeEvent => isEmojiAssetChange(changeEvent.path))) {
        for (const file of await writeEmojiAssets(ctx)) {
          yield file
        }
      }

      for (const changeEvent of changeEvents) {
        if (!isWorkerEntryPath(changeEvent.path)) continue
        if (changeEvent.type === 'delete') {
          const name = path.basename(changeEvent.path).replace(/\.ts$/, '')
          const dest = joinSegments(ctx.argv.output, `${name}.js`)
          await fs.unlink(dest)
          continue
        }
        const result = await bundle({
          entryPoints: [changeEvent.path],
          bundle: true,
          minify: true,
          platform: 'browser',
          format: 'esm',
          write: false,
        })
        const code = result.outputFiles[0].text
        const name = path.basename(changeEvent.path).replace(/\.ts$/, '')
        yield write({ ctx, slug: name as FullSlug, ext: '.js', content: code })
      }

      yield writeAssetManifest(ctx)
    },
    externalResources: ({ cfg }) => ({
      additionalHead: [
        <link rel="manifest" href={`https://${cfg.configuration.baseUrl}/site.webmanifest`} />,
        <link rel="shortcut icon" href={`https://${cfg.configuration.baseUrl}/favicon.ico`} />,
        <link
          rel="icon"
          type="image/png"
          sizes="32x32"
          href={`https://${cfg.configuration.baseUrl}/favicon-32x32.png`}
        />,
        <link
          rel="icon"
          type="image/png"
          sizes="16x16"
          href={`https://${cfg.configuration.baseUrl}/favicon-16x16.png`}
        />,
        <link
          rel="apple-touch-icon"
          sizes="180x180"
          href={`https://${cfg.configuration.baseUrl}/apple-touch-icon.png`}
        />,
        <link
          rel="android-chrome"
          sizes="192x192"
          href={`https://${cfg.configuration.baseUrl}/android-chrome-192x192.png`}
        />,
        <link
          rel="android-chrome"
          sizes="512x512"
          href={`https://${cfg.configuration.baseUrl}/android-chrome-512x512.png`}
        />,
      ],
    }),
  }
}
