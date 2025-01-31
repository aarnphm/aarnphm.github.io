import { FilePath, FullSlug, joinSegments } from "../../util/path"
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
import { googleFontHref, joinStyles } from "../../util/theme"
import { Features, transform } from "lightningcss"
import { transform as transpile } from "esbuild"
import { write } from "./helpers"
import DepGraph from "../../depgraph"
import { SocialImageOptions, getSatoriFont, defaultImageOptions } from "../../util/og"
import satori, { SatoriOptions } from "satori"
import { QuartzPluginData } from "../vfile"
import sharp from "sharp"
import { unescapeHTML } from "../../util/escape"
import { i18n } from "../../i18n"
import { styleText } from "node:util"

const name = "ComponentResources"

type ComponentResources = {
  css: string[]
  beforeDOMLoaded: string[]
  afterDOMLoaded: string[]
}

function getComponentResources(ctx: BuildCtx): ComponentResources {
  const allComponents: Set<QuartzComponent> = new Set()
  for (const emitter of ctx.cfg.plugins.emitters) {
    const components = emitter.getQuartzComponents(ctx)
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
      plausibleScript.setAttribute("data-domain", location.hostname)
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

async function generateOgImage(
  ctx: BuildCtx,
  fonts: SatoriOptions["fonts"],
  opts: SocialImageOptions,
  title: string,
  description: string,
  fileData: QuartzPluginData,
  fileName: string,
) {
  const svg = await satori(
    opts.Component(ctx.cfg.configuration, fileData, opts, title, description, fonts),
    {
      width: opts.width,
      height: opts.height,
      fonts,
      graphemeImages: {
        "🚧": "https://cdnjs.cloudflare.com/ajax/libs/twemoji/14.0.2/svg/1f6a7.svg",
      },
    },
  )

  const content = await sharp(Buffer.from(svg)).webp({ quality: 70 }).toBuffer()

  return await write({
    ctx,
    slug: joinSegments("static", "social-images", fileName) as FullSlug,
    ext: `.webp`,
    content,
  })
}

interface Options {
  fontOrigin: "googleFonts" | "local"
}

const defaultOptions: Options = {
  fontOrigin: "googleFonts",
}

export const ComponentResources: QuartzEmitterPlugin<Options> = (opts?: Partial<Options>) => {
  let fonts: Promise<SatoriOptions["fonts"]>
  let imageOptions: SocialImageOptions

  const { fontOrigin } = { ...defaultOptions, ...opts }
  return {
    name,
    getQuartzComponents: () => [],
    async getDependencyGraph(_ctx, _content, _resources) {
      return new DepGraph<FilePath>()
    },
    async emit(ctx, content, _resources): Promise<FilePath[]> {
      const promises: Promise<FilePath>[] = []
      const cfg = ctx.cfg.configuration
      // component specific scripts and styles
      const componentResources = getComponentResources(ctx)
      let googleFontsStyleSheet = ""
      if (fontOrigin === "local") {
        // let the user do it themselves in css
      } else if (fontOrigin === "googleFonts" && !cfg.theme.cdnCaching) {
        let match

        const fontSourceRegex = /url\((https:\/\/fonts.gstatic.com\/s\/[^)]+\.(woff2|ttf))\)/g

        googleFontsStyleSheet = await (
          await fetch(googleFontHref(ctx.cfg.configuration.theme))
        ).text()

        while ((match = fontSourceRegex.exec(googleFontsStyleSheet)) !== null) {
          // match[0] is the `url(path)`, match[1] is the `path`
          const url = match[1]
          // the static name of this file.
          const [filename, ext] = url.split("/").pop()!.split(".")

          googleFontsStyleSheet = googleFontsStyleSheet.replace(
            url,
            `https://${cfg.baseUrl}/static/fonts/${filename}.ttf`,
          )

          promises.push(
            fetch(url)
              .then((res) => {
                if (!res.ok) {
                  throw new Error(`Failed to fetch font`)
                }
                return res.arrayBuffer()
              })
              .then((buf) =>
                write({
                  ctx,
                  slug: joinSegments("static", "fonts", filename) as FullSlug,
                  ext: `.${ext}`,
                  content: Buffer.from(buf),
                }),
              ),
          )
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

      promises.push(
        write({
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
        write({
          ctx,
          slug: "prescript" as FullSlug,
          ext: ".js",
          content: prescript,
        }),
        write({
          ctx,
          slug: "postscript" as FullSlug,
          ext: ".js",
          content: postscript,
        }),
        write({
          ctx,
          slug: "site" as FullSlug,
          ext: ".webmanifest",
          content: JSON.stringify(manifest),
        }),
      )

      if (cfg.generateSocialImages && !ctx.argv.serve) {
        if (ctx.argv.verbose)
          console.log(styleText("blue", `[emit:${name}] Generating social images...`))

        if (!imageOptions) {
          if (typeof cfg.generateSocialImages !== "boolean") {
            imageOptions = { ...defaultImageOptions, ...cfg.generateSocialImages }
          } else {
            imageOptions = defaultImageOptions
          }
        }

        if (!fonts) fonts = getSatoriFont(cfg)
        const fontData = await fonts

        const ogs = [...content]
          .filter(([_, file]) => !file.data.slug!.includes("university"))
          .map(([_, file]) => {
            const slug = file.data.slug!
            const fileName = slug.replaceAll("/", "-")
            const title = file.data.frontmatter?.title ?? i18n(cfg.locale).propertyDefaults.title
            const description = unescapeHTML(
              file.data.frontmatter?.description ??
                file.data.description?.trim() ??
                i18n(cfg.locale).propertyDefaults.description,
            )

            return generateOgImage(
              ctx,
              fontData,
              imageOptions,
              title,
              description,
              file.data,
              fileName,
            )
          })
        promises.push(...ogs)
      } else {
        if (ctx.argv.verbose)
          console.log(
            styleText("yellow", `[emit:${name}] Skipping OG generations during serve time.`),
          )
      }

      return Promise.all(promises)
    },
  }
}
