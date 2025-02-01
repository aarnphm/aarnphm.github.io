import satori, { SatoriOptions } from "satori"
import { GlobalConfiguration } from "../../cfg"
import { QuartzEmitterPlugin } from "../types"
import { i18n } from "../../i18n"
import { formatDate, getDate } from "../../components/Date"
import { FilePath, FullSlug, joinSegments } from "../../util/path"
import { write } from "./helpers"
import sharp from "sharp"
import { JSX } from "preact/jsx-runtime"
import { defaultImageOptions, getSatoriFont, SocialImageOptions } from "../../util/og"
import { HtmlContent, QuartzPluginData } from "../vfile"
import { BuildCtx } from "../../util/ctx"
import { styleText } from "node:util"
import { fromHtml } from "hast-util-from-html"
import { htmlToJsx } from "../../util/jsx"

export interface PressReleaseOptions {
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
  opts: PressReleaseOptions,
  fonts: SatoriOptions["fonts"],
  directory: "instagram" | "twitter",
): Promise<FilePath[]> {
  return Promise.all(
    items.map(async ([_, file]) => {
      const slug = file.data.slug!
      const fileName = slug.replaceAll("/", "-")
      const title = file.data.frontmatter?.title ?? i18n(cfg.locale).propertyDefaults.title

      const component = opts.Component(
        cfg,
        file.data,
        { ...defaultImageOptions, ...opts },
        title,
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
        slug: joinSegments("static", directory, fileName) as FullSlug,
        ext: ".png",
      })
    }),
  )
}

const TwitterPost: SocialImageOptions["Component"] = (
  cfg: GlobalConfiguration,
  fileData: QuartzPluginData,
  { colorScheme }: Omit<SocialImageOptions, "Component">,
  title: string,
  fonts: SatoriOptions["fonts"],
) => {
  let created: string | undefined
  let reading: string | undefined
  if (fileData.dates) {
    created = formatDate(getDate(cfg, fileData)!, cfg.locale)
  }
  const { locale } = cfg
  reading = i18n(locale).components.contentMeta.readingTime({
    minutes: Math.ceil(fileData.readingTime?.minutes!),
    words: Math.ceil(fileData.readingTime?.words!),
  })

  const Li = [created, reading]

  return (
    <div
      style={{
        position: "relative",
        display: "flex",
        flexDirection: "row",
        alignItems: "flex-start",
        height: "100%",
        width: "100%",
        background: cfg.theme.colors[colorScheme].light,
        backgroundSize: "100% 100%",
      }}
    >
      <div
        style={{
          display: "flex",
          height: "100%",
          width: "100%",
          flexDirection: "column",
          justifyContent: "flex-start",
          alignItems: "flex-start",
          gap: "1.5rem",
          paddingTop: "6rem",
          paddingBottom: "6rem",
          marginLeft: "4rem",
          fontFamily: fonts[0].name,
          maxWidth: "85%",
        }}
      >
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            textAlign: "left",
          }}
        >
          <h2
            style={{
              color: cfg.theme.colors[colorScheme].dark,
              fontSize: "3rem",
              fontWeight: fonts[0].weight,
            }}
          >
            {title}
          </h2>
          <ul
            style={{
              color: cfg.theme.colors[colorScheme].gray,
              gap: "1rem",
              fontSize: "1.5rem",
              fontFamily: fonts[1].name,
              fontStyle: "italic",
            }}
          >
            {Li.map((item, index) => {
              if (item) {
                return (
                  <li key={index} style={{ fontStyle: "italic" }}>
                    {item}
                  </li>
                )
              }
            })}
          </ul>
        </div>
        <p
          style={{
            color: cfg.theme.colors[colorScheme].dark,
            fontSize: "2rem",
            overflow: "hidden",
            marginTop: "4rem",
            textOverflow: "ellipsis",
            display: "-webkit-box",
            WebkitLineClamp: 7,
            WebkitBoxOrient: "vertical",
            lineClamp: 7,
          }}
        >
          <Abstract {...getAbstractProps(fileData.abstract!)} />
        </p>
      </div>
    </div>
  )
}

type Props = {
  children: JSX.Element
}

const getAbstractProps = (abstract: string): Props =>
  htmlToJsx("" as FilePath, fromHtml(abstract, { fragment: true })).props

function Abstract({ children }: Props) {
  return <span>{children}</span>
}

const InstagramPost: SocialImageOptions["Component"] = (
  cfg: GlobalConfiguration,
  fileData: QuartzPluginData,
  { colorScheme }: Omit<SocialImageOptions, "Component">,
  title: string,
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
            color: cfg.theme.colors[colorScheme].light,
            fontFamily: fonts[1].name,
            fontSize: "3em",
            margin: 0,
            fontStyle: "italic",
          }}
        >
          <em>{fileData.description}</em>
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
          <Abstract {...getAbstractProps(fileData.abstract!)} />
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

const defaultInstagramOptions: PressReleaseOptions = {
  height: 1920,
  width: 1080,
  Component: InstagramPost,
}

const defaultTwitterOptions: PressReleaseOptions = {
  height: 900,
  width: 900,
  Component: TwitterPost,
}

interface PressKitOptions {
  twitter: PressReleaseOptions
  instagram: PressReleaseOptions
}

const name = "PressKit"
export const PressKit: QuartzEmitterPlugin<Partial<PressKitOptions>> = (userOpts) => {
  const instagramOptions = { ...defaultInstagramOptions, ...userOpts?.instagram }
  const twitterOpts = { ...defaultTwitterOptions, ...userOpts?.twitter }
  return {
    skipDuringServe: true,
    requiresFullContent: true,
    name,
    getQuartzComponents: () => [],
    async emit(ctx, content, _resource) {
      const { configuration } = ctx.cfg
      // Re-use OG image generation infrastructure
      if (!configuration.baseUrl) {
        console.warn(`[emit:${name}] Skip PressKit generation ('baseUrl' is missing)`)
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

      // Process both platforms and all chunks in parallel
      const [instagram, twitter] = await Promise.all([
        Promise.all(
          chunks.map((chunk) =>
            processChunk(chunk, ctx, configuration, instagramOptions, fonts, "instagram"),
          ),
        ),
        Promise.all(
          chunks.map((chunk) =>
            processChunk(chunk, ctx, configuration, twitterOpts, fonts, "twitter"),
          ),
        ),
      ])

      return [...instagram.flat(), ...twitter.flat()]
    },
  }
}
