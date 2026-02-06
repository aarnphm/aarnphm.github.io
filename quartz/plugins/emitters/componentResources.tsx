import { transform as transpile, build as bundle } from 'esbuild'
import { globby } from 'globby'
import { Features, transform } from 'lightningcss'
import fs from 'node:fs/promises'
import path from 'path'
import type { QuartzMdxComponent } from '../../components/mdx/registry'
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
import multiplayerScript from '../../components/scripts/collaborative-comments.inline'
// @ts-ignore
import markerScript from '../../components/scripts/marker.inline'
// @ts-ignore
import popoverScript from '../../components/scripts/popover.inline'
//@ts-ignore
import protectedScript from '../../components/scripts/protected.inline'
// @ts-ignore
import spaRouterScript from '../../components/scripts/spa.inline'
import audioStyle from '../../components/styles/audio.scss'
import clipboardStyle from '../../components/styles/clipboard.scss'
import popoverStyle from '../../components/styles/popover.scss'
import pseudoStyle from '../../components/styles/pseudocode.scss'
import styles from '../../styles/custom.scss'
import '../../components/mdx'
import { QuartzComponent } from '../../types/component'
import { QuartzEmitterPlugin } from '../../types/plugin'
import { BuildCtx } from '../../util/ctx'
import { FullSlug, joinSegments } from '../../util/path'
import { googleFontHref, joinStyles, processGoogleFonts } from '../../util/theme'
import { write } from './helpers'

const name = 'ComponentResources'

type ComponentResources = { css: string[]; beforeDOMLoaded: string[]; afterDOMLoaded: string[] }

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

function addGlobalPageResources(ctx: BuildCtx, componentResources: ComponentResources) {
  const cfg = ctx.cfg.configuration

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
    multiplayerScript,
  )

  if (cfg.analytics?.provider === 'plausible') {
    const plausibleHost = cfg.analytics.host ?? 'https://plausible.io'
    componentResources.afterDOMLoaded.push(`
      const plausibleScript = document.createElement("script")
      plausibleScript.src = "${plausibleHost}/js/script.outbound-links.manual.js"
      plausibleScript.setAttribute("data-domain", [location.hostname, "notes.aarnphm.xyz", "stream.aarnphm.xyz"].join(','))
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
    async *emit(ctx) {
      const cfg = ctx.cfg.configuration
      // component specific scripts and styles
      const componentResources = getComponentResources(ctx)
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

      // important that this goes *after* component scripts
      // as the "nav" event gets triggered here and we should make sure
      // that everyone else had the chance to register a listener for it
      addGlobalPageResources(ctx, componentResources)

      const stylesheet = joinStyles(
        ctx.cfg.configuration.theme,
        googleFontsStyleSheet,
        ...componentResources.css,
        styles,
      )
      const [prescript, postscript] = await Promise.all([
        joinScripts(componentResources.beforeDOMLoaded),
        joinScripts(componentResources.afterDOMLoaded),
      ])

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

      yield write({
        ctx,
        slug: 'index' as FullSlug,
        ext: '.css',
        content: transform({
          filename: 'index.css',
          code: Buffer.from(stylesheet),
          minify: true,
          targets: {
            safari: (15 << 16) | (6 << 8), // 15.6
            ios_saf: (15 << 16) | (6 << 8), // 15.6
            edge: 115 << 16,
            firefox: 102 << 16,
            chrome: 109 << 16,
          },
          include: Features.MediaQueries,
        }).code.toString(),
      })

      yield write({ ctx, slug: 'prescript' as FullSlug, ext: '.js', content: prescript })

      yield write({ ctx, slug: 'postscript' as FullSlug, ext: '.js', content: postscript })

      yield write({
        ctx,
        slug: 'site' as FullSlug,
        ext: '.webmanifest',
        content: JSON.stringify(manifest),
      })

      const workerFiles = await globby(['quartz/**/*.worker.ts'])
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
    },
    async *partialEmit(ctx, _content, _resources, changeEvents) {
      for (const changeEvent of changeEvents) {
        if (!changeEvent.path.endsWith('.worker.ts')) continue
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
