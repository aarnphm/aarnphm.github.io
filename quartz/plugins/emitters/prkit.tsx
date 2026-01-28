import type { SatoriOptions } from "satori/wasm"
import { fromHtml } from "hast-util-from-html"
import { JSX } from "preact"
import satori from "satori"
import sharpLib from "sharp"
import { GlobalConfiguration } from "../../cfg"
import { formatDate, getDate } from "../../components/Date"
import { i18n } from "../../i18n"
import { QuartzEmitterPlugin } from "../../types/plugin"
import { BuildCtx } from "../../util/ctx"
import { loadEmoji, getIconCode } from "../../util/emoji"
import { htmlToJsx } from "../../util/jsx"
import { getSatoriFonts } from "../../util/og"
import { FilePath, FullSlug, joinSegments } from "../../util/path"
import { ThemeKey } from "../../util/theme"
import { ProcessedContent, QuartzPluginData } from "../vfile"
import { write } from "./helpers"

type PressReleaseComponent = (
  cfg: GlobalConfiguration,
  fileData: QuartzPluginData,
  opts: PressReleaseOptions,
  title: string,
  fonts: NonNullable<SatoriOptions["fonts"]>,
) => JSX.Element

export interface PressReleaseOptions {
  height: number
  width: number
  colorScheme: ThemeKey
  Component: PressReleaseComponent
}

function chunk<T>(arr: T[], size: number): T[][] {
  return Array.from({ length: Math.ceil(arr.length / size) }, (_, i) =>
    arr.slice(i * size, i * size + size),
  )
}
async function processChunk(
  items: ProcessedContent[],
  ctx: BuildCtx,
  cfg: GlobalConfiguration,
  opts: PressReleaseOptions,
  fonts: NonNullable<SatoriOptions["fonts"]>,
  directory: "instagram" | "twitter",
): Promise<FilePath[]> {
  return Promise.all(
    items.map(async ([_, file]) => {
      const slug = file.data.slug!
      const fileName = slug.replaceAll("/", "-")
      const title = file.data.frontmatter?.title ?? i18n(cfg.locale).propertyDefaults.title

      const component = opts.Component(cfg, file.data, opts, title, fonts)
      const svg = await satori(component, {
        width: opts.width,
        height: opts.height,
        fonts,
        // `code` will be the detected language code, `emoji` if it's an Emoji, or `unknown` if not able to tell.
        // `segment` will be the content to render.
        loadAdditionalAsset: async (languageCode: string, segment: string) => {
          if (languageCode === "emoji") {
            return `data:image/svg+xml;base64,${btoa(await loadEmoji(getIconCode(segment)))}`
          }
          return languageCode
        },
      })
      const img = await sharpLib(Buffer.from(svg)).png().toBuffer()
      return await write({
        ctx,
        content: img,
        slug: joinSegments("static", directory, fileName) as FullSlug,
        ext: ".png",
      })
    }),
  )
}

const TwitterPost: PressReleaseComponent = (
  cfg: GlobalConfiguration,
  fileData: QuartzPluginData,
  { colorScheme },
  title: string,
  fonts: NonNullable<SatoriOptions["fonts"]>,
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

  const metaItems: string[] = []
  if (created) metaItems.push(created)
  if (reading) metaItems.push(reading)

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
            {metaItems.map((item, index) => (
              <li key={index} style={{ fontStyle: "italic" }}>
                {item}
              </li>
            ))}
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
          }}
        >
          {fileData.abstract && <Abstract {...getAbstractProps(fileData.abstract)} />}
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

const InstagramPost: PressReleaseComponent = (
  cfg: GlobalConfiguration,
  fileData: QuartzPluginData,
  { colorScheme },
  title: string,
  fonts: NonNullable<SatoriOptions["fonts"]>,
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
          background: "radial-gradient(circle at center, transparent, rgba(0, 0, 0, 0.2) 70%)",
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
            fontSize: "2.5em",
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
            fontSize: "1.5em",
          }}
        >
          {fileData.abstract && <Abstract {...getAbstractProps(fileData.abstract)} />}
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
          {cfg.baseUrl}/{fileData.slug}
        </p>
      </div>
    </div>
  )
}

const defaultInstagramOptions: PressReleaseOptions = {
  height: 1920,
  width: 1080,
  colorScheme: "lightMode",
  Component: InstagramPost,
}

const defaultTwitterOptions: PressReleaseOptions = {
  height: 900,
  width: 900,
  colorScheme: "lightMode",
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
    name,
    getQuartzComponents() {
      return []
    },
    async *partialEmit() {},
    async *emit(ctx, content, _resource) {
      if (ctx.argv.watch && !ctx.argv.force) return []

      const { configuration } = ctx.cfg
      // Re-use OG image generation infrastructure
      if (!configuration.baseUrl) {
        console.warn(`[emit:${name}] Skip PressKit generation ('baseUrl' is missing)`)
        return
      }

      // Filter content first
      const filteredContents = [...content].filter(
        ([_, file]) =>
          !file.data.slug!.includes("university") && file.data.slug!.includes("posts/"),
      )
      if (filteredContents.length === 0) return
      const headerFont = configuration.theme.typography.header
      const bodyFont = configuration.theme.typography.body
      const fonts = (await getSatoriFonts(configuration, headerFont, bodyFont)) as NonNullable<
        SatoriOptions["fonts"]
      >

      // rough heuristics: 128 gives enough time for v8 to JIT and optimize parsing code paths
      const NUM_WORKERS = 4
      const CHUNK_SIZE = Math.ceil(filteredContents.length / NUM_WORKERS)
      const chunks = chunk(filteredContents, CHUNK_SIZE)

      const platforms: Array<["instagram" | "twitter", PressReleaseOptions]> = [
        ["instagram", instagramOptions],
        ["twitter", twitterOpts],
      ]

      for (const [platform, opts] of platforms) {
        for (const chunkItems of chunks) {
          const results = await processChunk(chunkItems, ctx, configuration, opts, fonts, platform)
          for (const filePath of results) {
            yield filePath
          }
        }
      }
    },

    externalResources: (ctx) => {
      if (!ctx.cfg.configuration.baseUrl) return {}

      const baseUrl = ctx.cfg.configuration.baseUrl
      return {
        additionalHead: [
          (pageData) => {
            return (
              <>
                {!pageData.slug!.includes("university") && (
                  <>
                    <meta
                      name="pr:twitter"
                      content={`https://${baseUrl}/static/twitter/${pageData.slug!.replaceAll("/", "-")}.png`}
                    />
                    <meta
                      name="pr:instagram"
                      content={`https://${baseUrl}/static/instagram/${pageData.slug!.replaceAll("/", "-")}.png`}
                    />
                  </>
                )}
              </>
            )
          },
        ],
      }
    },
  }
}
