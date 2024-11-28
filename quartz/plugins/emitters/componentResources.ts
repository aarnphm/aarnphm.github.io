import { FilePath, FullSlug, joinSegments } from "../../util/path"
import { QuartzEmitterPlugin } from "../types"
// @ts-ignore
import spaRouterScript from "../../components/scripts/spa.inline"
// @ts-ignore
import popoverScript from "../../components/scripts/popover.inline"
// @ts-ignore
import fontsScript from "../../components/scripts/fonts.inline"
import styles from "../../styles/custom.scss"
import popoverStyle from "../../components/styles/popover.scss"
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
import chalk from "chalk"
import EventEmitter from "events"

const NAME = "ComponentResources"

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

  if (cfg.analytics?.provider === "google") {
    const tagId = cfg.analytics.tagId
    componentResources.afterDOMLoaded.push(`
      const gtagScript = document.createElement("script")
      gtagScript.src = "https://www.googletagmanager.com/gtag/js?id=${tagId}"
      gtagScript.async = true
      document.head.appendChild(gtagScript)

      window.dataLayer = window.dataLayer || [];
      function gtag() { dataLayer.push(arguments); }
      gtag("js", new Date());
      gtag("config", "${tagId}", { send_page_view: false });

      document.addEventListener("nav", () => {
        gtag("event", "page_view", {
          page_title: document.title,
          page_location: location.href,
        });
      });`)
  } else if (cfg.analytics?.provider === "plausible") {
    const plausibleHost = cfg.analytics.host ?? "https://plausible.io"
    componentResources.afterDOMLoaded.push(`
      const plausibleScript = document.createElement("script")
      plausibleScript.src = "${plausibleHost}/js/script.manual.js"
      plausibleScript.setAttribute("data-domain", location.hostname)
      plausibleScript.defer = true
      document.head.appendChild(plausibleScript)

      window.plausible = window.plausible || function() { (window.plausible.q = window.plausible.q || []).push(arguments) }

      document.addEventListener("nav", () => {
        plausible("pageview")
      })
    `)
  } else if (cfg.analytics?.provider === "umami") {
    componentResources.afterDOMLoaded.push(`
      const umamiScript = document.createElement("script")
      umamiScript.src = "${cfg.analytics.host ?? "https://analytics.umami.is"}/script.js"
      umamiScript.setAttribute("data-website-id", "${cfg.analytics.websiteId}")
      umamiScript.async = true

      document.head.appendChild(umamiScript)
    `)
  } else if (cfg.analytics?.provider === "goatcounter") {
    componentResources.afterDOMLoaded.push(`
      const goatcounterScript = document.createElement("script")
      goatcounterScript.src = "${cfg.analytics.scriptSrc ?? "https://gc.zgo.at/count.js"}"
      goatcounterScript.async = true
      goatcounterScript.setAttribute("data-goatcounter",
        "https://${cfg.analytics.websiteId}.${cfg.analytics.host ?? "goatcounter.com"}/count")
      document.head.appendChild(goatcounterScript)
    `)
  } else if (cfg.analytics?.provider === "posthog") {
    componentResources.afterDOMLoaded.push(`
      const posthogScript = document.createElement("script")
      posthogScript.innerHTML= \`!function(t,e){var o,n,p,r;e.__SV||(window.posthog=e,e._i=[],e.init=function(i,s,a){function g(t,e){var o=e.split(".");2==o.length&&(t=t[o[0]],e=o[1]),t[e]=function(){t.push([e].concat(Array.prototype.slice.call(arguments,0)))}}(p=t.createElement("script")).type="text/javascript",p.async=!0,p.src=s.api_host+"/static/array.js",(r=t.getElementsByTagName("script")[0]).parentNode.insertBefore(p,r);var u=e;for(void 0!==a?u=e[a]=[]:a="posthog",u.people=u.people||[],u.toString=function(t){var e="posthog";return"posthog"!==a&&(e+="."+a),t||(e+=" (stub)"),e},u.people.toString=function(){return u.toString(1)+".people (stub)"},o="capture identify alias people.set people.set_once set_config register register_once unregister opt_out_capturing has_opted_out_capturing opt_in_capturing reset isFeatureEnabled onFeatureFlags getFeatureFlag getFeatureFlagPayload reloadFeatureFlags group updateEarlyAccessFeatureEnrollment getEarlyAccessFeatures getActiveMatchingSurveys getSurveys onSessionId".split(" "),n=0;n<o.length;n++)g(u,o[n]);e._i.push([i,s,a])},e.__SV=1)}(document,window.posthog||[]);
      posthog.init('${cfg.analytics.apiKey}',{api_host:'${cfg.analytics.host ?? "https://app.posthog.com"}'})\`
      document.head.appendChild(posthogScript)
    `)
  } else if (cfg.analytics?.provider === "tinylytics") {
    const siteId = cfg.analytics.siteId
    componentResources.afterDOMLoaded.push(`
      const tinylyticsScript = document.createElement("script")
      tinylyticsScript.src = "https://tinylytics.app/embed/${siteId}.js"
      tinylyticsScript.defer = true
      document.head.appendChild(tinylyticsScript)
    `)
  } else if (cfg.analytics?.provider === "cabin") {
    componentResources.afterDOMLoaded.push(`
      const cabinScript = document.createElement("script")
      cabinScript.src = "${cfg.analytics.host ?? "https://scripts.cabin.dev"}/cabin.js"
      cabinScript.defer = true
      cabinScript.async = true
      document.head.appendChild(cabinScript)
    `)
  }

  if (cfg.enableSPA) {
    componentResources.afterDOMLoaded.push(spaRouterScript)
  } else {
    componentResources.afterDOMLoaded.push(`
      window.spaNavigate = (url, _) => window.location.assign(url)
      window.addCleanup = () => {}
      const event = new CustomEvent("nav", { detail: { url: document.body.dataset.slug } })
      document.dispatchEvent(event)
    `)
  }

  componentResources.beforeDOMLoaded.push(fontsScript)
}

interface OgTask {
  title: string
  description: string
  fileData: QuartzPluginData
  fileDir: string
  fileName: string
  extension: string
}

class OgImageQueue extends EventEmitter {
  private queue: OgTask[] = []
  private processing = false
  private completed = 0
  private total = 0
  private progressBar = ""

  constructor(
    private ctx: BuildCtx,
    private fonts: SatoriOptions["fonts"],
    private opts: SocialImageOptions,
  ) {
    super()
  }

  add(task: OgTask) {
    this.queue.push(task)
    this.total++
  }

  process() {
    if (this.processing) return []
    this.processing = true

    const fps: Promise<FilePath>[] = []
    const batchSize = this.ctx.argv.concurrency ?? 10

    // Add event listener for progress
    if (this.ctx.argv.verbose) {
      this.on("progress", (completed, total) => {
        const percent = Math.round((completed / total) * 100)
        this.progressBar = `[emit:${NAME}] Generating OG images: ${completed}/${total} (${percent}%)`

        // Only write newline before first progress message
        process.stdout.write(`\r${this.progressBar}`)

        // Write newline when complete
        if (completed === total) {
          process.stdout.write("\n")
        }
      })
    }

    while (this.queue.length > 0) {
      const batch = this.queue.splice(0, batchSize)
      batch.map(async ({ title, description, fileData, fileDir, fileName, extension }) => {
        try {
          const component = this.opts.Component(
            this.ctx.cfg.configuration,
            fileData,
            this.opts,
            title,
            description,
            this.fonts,
          )
          const svg = await satori(component, {
            width: this.opts.width,
            height: this.opts.height,
            fonts: this.fonts,
            graphemeImages: {
              "🚧": "https://cdnjs.cloudflare.com/ajax/libs/twemoji/14.0.2/svg/1f6a7.svg",
            },
          })

          const content = await sharp(Buffer.from(svg)).webp({ quality: 40 }).toBuffer()

          fps.push(
            write({
              ctx: this.ctx,
              slug: joinSegments("static", fileDir, fileName) as FullSlug,
              ext: `.${extension}`,
              content,
            }),
          )

          this.completed++
          if (this.ctx.argv.verbose) {
            this.emit("progress", this.completed, this.total)
          }
        } catch (error) {
          console.error(
            chalk.red(`\n[emit:${NAME}] Failed to generate social image for "${title}":`, error),
          )
        }
      })
    }
    return fps
  }
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
    name: NAME,
    getQuartzComponents() {
      return []
    },
    async getDependencyGraph(ctx, content, _resources) {
      // This emitter adds static resources to the `resources` parameter. One
      // important resource this emitter adds is the code to start a websocket
      // connection and listen to rebuild messages, which triggers a page reload.
      // The resources parameter with the reload logic is later used by the
      // ContentPage emitter while creating the final html page. In order for
      // the reload logic to be included, and so for partial rebuilds to work,
      // we need to run this emitter for all markdown files.
      const graph = new DepGraph<FilePath>()

      for (const [_tree, file] of content) {
        const sourcePath = file.data.filePath!
        const slug = file.data.slug!
        graph.addEdge(sourcePath, joinSegments(ctx.argv.output, slug + ".html") as FilePath)
      }

      return graph
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
        ...componentResources.css,
        googleFontsStyleSheet,
        ...componentResources.css,
        styles,
      )
      const [prescript, postscript] = await Promise.all([
        joinScripts(componentResources.beforeDOMLoaded),
        joinScripts(componentResources.afterDOMLoaded),
      ])

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
      )

      if (cfg.generateSocialImages && !ctx.argv.serve) {
        if (!imageOptions) {
          if (typeof cfg.generateSocialImages !== "boolean") {
            imageOptions = { ...defaultImageOptions, ...cfg.generateSocialImages }
          } else {
            imageOptions = defaultImageOptions
          }
        }

        if (!fonts) fonts = getSatoriFont(cfg)
        const fontData = await fonts

        const queue = new OgImageQueue(ctx, fontData, imageOptions)

        for (const [_, file] of content) {
          const slug = file.data.slug!
          const fileName = slug.replaceAll("/", "-")

          const title = file.data.frontmatter?.title ?? i18n(cfg.locale).propertyDefaults.title
          const description = unescapeHTML(
            file.data.frontmatter?.description ??
              file.data.description?.trim() ??
              i18n(cfg.locale).propertyDefaults.description,
          )

          queue.add({
            title,
            description,
            fileData: file.data,
            fileDir: "social-images",
            fileName,
            extension: "webp",
          })
        }
        // Start processing in background
        if (ctx.argv.verbose) {
          console.log(
            chalk.blue(`[emit:${NAME}] Starting social image generation in background...`),
          )
        }
        promises.push(...queue.process())
      }

      return await Promise.all(promises)
    },
  }
}
