import satori, { SatoriOptions } from "satori"
import { GlobalConfiguration } from "../../cfg"
import { QuartzEmitterPlugin } from "../types"
import { i18n } from "../../i18n"
import { FilePath, FullSlug, joinSegments } from "../../util/path"
import { write } from "./helpers"
import sharp from "sharp"
import { defaultImageOptions, getSatoriFont, SocialImageOptions } from "../../util/og"
import { HtmlContent, QuartzPluginData } from "../vfile"
import { unescapeHTML } from "../../util/escape"
import { BuildCtx } from "../../util/ctx"
import { styleText } from "node:util"

export interface InstagramOptions {
  height: number
  width: number
  Component: SocialImageOptions["Component"]
}

function chunk<T>(arr: T[], size: number): T[][] {
  return Array.from({ length: Math.ceil(arr.length / size) }, (_, i) =>
    arr.slice(i * size, i * size + size),
  )
}
async function processChunk(
  items: HtmlContent[],
  ctx: BuildCtx,
  cfg: GlobalConfiguration,
  opts: InstagramOptions,
  fonts: SatoriOptions["fonts"],
): Promise<FilePath[]> {
  return Promise.all(
    items.map(async ([_, file]) => {
      const slug = file.data.slug!
      const fileName = slug.replaceAll("/", "-")
      const title = file.data.frontmatter?.title ?? i18n(cfg.locale).propertyDefaults.title
      const description = unescapeHTML(
        file.data.frontmatter?.description ??
          file.data.description?.trim() ??
          i18n(cfg.locale).propertyDefaults.description,
      )

      const component = opts.Component(
        cfg,
        file.data,
        { ...defaultImageOptions, ...opts },
        title,
        description,
        fonts,
      )
      const svg = await satori(component, {
        width: opts.width,
        height: opts.height,
        fonts,
      })
      const img = await sharp(Buffer.from(svg)).png().toBuffer()
      return await write({
        ctx,
        content: img,
        slug: joinSegments("static", "instagram", fileName) as FullSlug,
        ext: ".png",
      })
    }),
  )
}

const InstagramPost: SocialImageOptions["Component"] = (
  cfg: GlobalConfiguration,
  fileData: QuartzPluginData,
  { colorScheme }: Omit<SocialImageOptions, "Component">,
  title: string,
  description: string,
  fonts: SatoriOptions["fonts"],
) => {
  return (
    <div
      style={{
        display: "flex",
        height: "100%",
        width: "100%",
        alignItems: "center",
        justifyContent: "center",
        backgroundImage: `url("https://${cfg.baseUrl}/static/og-vertical.png")`,
        backgroundRepeat: "no-repeat",
        backgroundPosition: "center center",
        backgroundSize: "100% 100%",
        position: "relative",
        fontSize: "1.1875em",
      }}
    >
      <div
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: "radial-gradient(circle at center, transparent, rgba(0, 0, 0, 0.4) 70%)",
        }}
      />
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "flex-start",
          justifyContent: "center",
          gap: "2rem",
          padding: "2rem",
          maxWidth: "85%",
        }}
      >
        <h1
          style={{
            color: cfg.theme.colors[colorScheme].light,
            fontSize: "4em",
            fontWeight: fonts[0].weight,
            fontFamily: fonts[0].name,
            margin: 0,
            lineHeight: 1.2,
          }}
        >
          {title}
        </h1>
        <p
          style={{
            color: cfg.theme.colors[colorScheme].gray,
            fontFamily: fonts[1].name,
            fontSize: "3em",
            margin: 0,
            fontStyle: "italic",
          }}
        >
          <em>{description}</em>
        </p>
        <p
          style={{
            color: cfg.theme.colors[colorScheme].light,
            fontFamily: fonts[1].name,
            fontWeight: fonts[1].weight,
            margin: 0,
            marginTop: "6rem",
            fontSize: "2em",
          }}
        >
          {fileData.abstract}
        </p>
        <p
          style={{
            color: cfg.theme.colors[colorScheme].light,
            fontSize: "1.25em",
            fontFamily: fonts[1].name,
            textDecoration: "underline",
            margin: 0,
            marginTop: "12rem",
          }}
        >
          {cfg.baseUrl}
        </p>
      </div>
    </div>
  )
}

const defaultOptions: InstagramOptions = {
  height: 1920,
  width: 1080,
  Component: InstagramPost,
}

const name = "PressKit"
export const PressKit: QuartzEmitterPlugin<Partial<InstagramOptions>> = (userOpts) => {
  const opts = { ...defaultOptions, ...userOpts }
  return {
    skipDuringServe: true,
    requiresFullContent: true,
    name,
    getQuartzComponents: () => [],
    async emit(ctx, content, _resource) {
      const { configuration } = ctx.cfg
      // Re-use OG image generation infrastructure
      if (!configuration.baseUrl) {
        console.warn(`[emit:${name}] Instagram image generation requires \`baseUrl\` to be set`)
        return []
      }

      // Filter content first
      const filteredContents = [...content].filter(
        ([_, file]) => !file.data.slug!.includes("university"),
      )
      if (filteredContents.length === 0) return []
      const fonts = await getSatoriFont(configuration, true)

      // rough heuristics: 128 gives enough time for v8 to JIT and optimize parsing code paths
      const NUM_WORKERS = 4
      const CHUNK_SIZE = Math.ceil(filteredContents.length / NUM_WORKERS)
      const chunks = chunk(filteredContents, CHUNK_SIZE)

      if (ctx.argv.verbose) console.log(styleText("blue", `[emit:${name}] Generating press kit...`))
      // Process chunks in parallel
      const results = await Promise.all(
        chunks.map((chunk) => processChunk(chunk, ctx, configuration, opts, fonts)),
      )

      return results.flat()
    },
  }
}
