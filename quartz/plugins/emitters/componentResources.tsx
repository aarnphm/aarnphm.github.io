import { FullSlug, joinSegments } from "../../util/path"
import { QuartzEmitterPlugin } from "../types"
// @ts-ignore
import spaRouterScript from "../../components/scripts/spa.inline"
//@ts-ignore
import insightsScript from "../../components/scripts/insights.inline"
// @ts-ignore
import popoverScript from "../../components/scripts/popover.inline"
import styles from "../../styles/custom.scss"
import popoverStyle from "../../components/styles/popover.scss"
// @ts-ignore
import clipboardScript from "../../components/scripts/clipboard.inline"
import clipboardStyle from "../../components/styles/clipboard.scss"
// @ts-ignore
import pseudoScript from "../../components/scripts/clipboard-pseudo.inline"
import pseudoStyle from "../../components/styles/pseudocode.scss"
import { BuildCtx } from "../../util/ctx"
import { QuartzComponent } from "../../components/types"
import { googleFontHref, joinStyles, processGoogleFonts } from "../../util/theme"
import { Features, transform } from "lightningcss"
import { transform as transpile } from "esbuild"
import { write } from "./helpers"

const name = "ComponentResources"

type ComponentResources = {
  css: string[]
  beforeDOMLoaded: string[]
  afterDOMLoaded: string[]
}

function getComponentResources(ctx: BuildCtx): ComponentResources {
  const allComponents: Set<QuartzComponent> = new Set()
  for (const emitter of ctx.cfg.plugins.emitters) {
    const components = emitter.getQuartzComponents?.(ctx) ?? []
    for (const component of components) {
      allComponents.add(component)
    }
  }

  const componentResources = {
    css: new Set<string>(),
    beforeDOMLoaded: new Set<string>(),
    afterDOMLoaded: new Set<string>(),
  }

  for (const component of allComponents) {
    const { css, beforeDOMLoaded, afterDOMLoaded } = component
    if (css) {
      componentResources.css.add(css)
    }
    if (beforeDOMLoaded) {
      componentResources.beforeDOMLoaded.add(beforeDOMLoaded)
    }
    if (afterDOMLoaded) {
      componentResources.afterDOMLoaded.add(afterDOMLoaded)
    }
  }

  return {
    css: [...componentResources.css],
    beforeDOMLoaded: [...componentResources.beforeDOMLoaded],
    afterDOMLoaded: [...componentResources.afterDOMLoaded],
  }
}

async function joinScripts(scripts: string[]): Promise<string> {
  // wrap with iife to prevent scope collision
  const script = scripts.map((script) => `(function () {${script}})();`).join("\n")

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

  componentResources.css.push(clipboardStyle, pseudoStyle)
  componentResources.afterDOMLoaded.push(clipboardScript, pseudoScript)

  if (cfg.analytics?.provider === "plausible") {
    const plausibleHost = cfg.analytics.host ?? "https://plausible.io"
    componentResources.afterDOMLoaded.push(`
      const plausibleScript = document.createElement("script")
      plausibleScript.src = "${plausibleHost}/js/script.outbound-links.manual.js"
      plausibleScript.setAttribute("data-domain", [location.hostname, "notes.aarnphm.xyz", "tinymorph.aarnphm.xyz"].join(','))
      plausibleScript.setAttribute("spa-preserve", true)
      plausibleScript.defer = true
      document.head.appendChild(plausibleScript)

      window.plausible = window.plausible || function() { (window.plausible.q = window.plausible.q || []).push(arguments) }

      document.addEventListener("nav", () => {
        plausible("pageview")
      })
    `)
  }

  componentResources.afterDOMLoaded.push(insightsScript, spaRouterScript)
}

export const ComponentResources: QuartzEmitterPlugin = () => {
  return {
    name,
    async *emit(ctx, _content, _resources) {
      const cfg = ctx.cfg.configuration
      // component specific scripts and styles
      const componentResources = getComponentResources(ctx)
      let googleFontsStyleSheet = ""
      if (cfg.theme.fontOrigin === "local") {
        // let the user do it themselves in css
      } else if (cfg.theme.fontOrigin === "googleFonts" && !cfg.theme.cdnCaching) {
        const response = await fetch(googleFontHref(ctx.cfg.configuration.theme))
        googleFontsStyleSheet = await response.text()

        if (!cfg.baseUrl) {
          throw new Error(
            "baseUrl must be defined when using Google Fonts without cfg.theme.cdnCaching",
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
            slug: joinSegments("static", "fonts", fontFile.filename) as FullSlug,
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
          { src: "/android-chrome-192x192.png", sizes: "192x192", type: "image/png" },
          { src: "/android-chrome-512x512.png", sizes: "512x512", type: "image/png" },
        ],
        theme_color: cfg.theme.colors["lightMode"].light,
        background_color: cfg.theme.colors["lightMode"].light,
        display: "standalone",
        lang: cfg.locale,
        dir: "auto",
      }

      yield write({
        ctx,
        slug: "index" as FullSlug,
        ext: ".css",
        content: transform({
          filename: "index.css",
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
      }),
        yield write({
          ctx,
          slug: "prescript" as FullSlug,
          ext: ".js",
          content: prescript,
        }),
        yield write({
          ctx,
          slug: "postscript" as FullSlug,
          ext: ".js",
          content: postscript,
        })
      yield write({
        ctx,
        slug: "site" as FullSlug,
        ext: ".webmanifest",
        content: JSON.stringify(manifest),
      })
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
