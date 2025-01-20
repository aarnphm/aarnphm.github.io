import { FontWeight, SatoriOptions } from "satori/wasm"
import { GlobalConfiguration } from "../cfg"
import { JSXInternal } from "preact/src/jsx"
import { joinSegments } from "./path"
import { QuartzPluginData } from "../plugins/vfile"
import { formatDate, getDate } from "../components/Date"
import { i18n } from "../i18n"
import { ThemeKey } from "./theme"

const headerFont = joinSegments("static", "GT-Sectra-Display-Regular.woff")
const bodyFont = joinSegments("static", "GT-Sectra-Book.woff")

export async function getSatoriFont(cfg: GlobalConfiguration): Promise<SatoriOptions["fonts"]> {
  const headerWeight: FontWeight = 700
  const bodyWeight: FontWeight = 400

  const url = new URL(`https://${cfg.baseUrl ?? "example.com"}`)

  const fetchFonts = async (font: string) => {
    const res = await fetch(`${url.toString()}/${font}`)
    const data = await res.arrayBuffer()
    return data
  }

  const header = await fetchFonts(headerFont)
  const body = await fetchFonts(bodyFont)

  return [
    { name: cfg.theme.typography.header, data: header, weight: headerWeight, style: "normal" },
    { name: cfg.theme.typography.body, data: body, weight: bodyWeight, style: "normal" },
  ]
}

/**
 * Get the `.ttf` file of a google font
 * @param fontName name of google font
 * @param weight what font weight to fetch font
 * @returns `.ttf` file of google font
 */
async function fetchTtf(fontName: string, weight: FontWeight): Promise<ArrayBuffer> {
  try {
    // Get css file from google fonts
    const cssResponse = await fetch(
      `https://fonts.googleapis.com/css2?family=${fontName}:wght@${weight}`,
    )
    const css = await cssResponse.text()

    // Extract .ttf url from css file
    const urlRegex = /url\((https:\/\/fonts.gstatic.com\/s\/.*?.ttf)\)/g
    const match = urlRegex.exec(css)

    if (!match) {
      throw new Error("Could not fetch font")
    }

    // Retrieve font data as ArrayBuffer
    const fontResponse = await fetch(match[1])

    // fontData is an ArrayBuffer containing the .ttf file data (get match[1] due to google fonts response format, always contains link twice, but second entry is the "raw" link)
    const fontData = await fontResponse.arrayBuffer()

    return fontData
  } catch (error) {
    throw new Error(`Error fetching font: ${error}`)
  }
}

export type SocialImageOptions = {
  /**
   * What color scheme to use for image generation (uses colors from config theme)
   */
  colorScheme: ThemeKey
  /**
   * Height to generate image with in pixels (should be around 630px)
   */
  height: number
  /**
   * Width to generate image with in pixels (should be around 1200px)
   */
  width: number
  /**
   * Whether to use the auto generated image for the root path ("/", when set to false) or the default og image (when set to true).
   */
  excludeRoot: boolean
  /**
   * JSX to use for generating image. See satori docs for more info (https://github.com/vercel/satori)
   * @param cfg global quartz config
   * @param userOpts options that can be set by user
   * @param title title of current page
   * @param description description of current page
   * @param fonts global font that can be used for styling
   * @returns prepared jsx to be used for generating image
   */
  Component: (
    cfg: GlobalConfiguration,
    fileData: QuartzPluginData,
    opts: Options,
    title: string,
    description: string,
    fonts: SatoriOptions["fonts"],
  ) => JSXInternal.Element
}

export type Options = Omit<SocialImageOptions, "Component">

export type ImageOptions = {
  /**
   * what title to use as header in image
   */
  title: string
  /**
   * what description to use as body in image
   */
  description: string
  /**
   * what fileName to use when writing to disk
   */
  fileName: string
  /**
   * what directory to store image in
   */
  fileDir: string
  /**
   * what file extension to use (should be `webp` unless you also change sharp conversion)
   */
  extension: string
  /**
   * header + body font to be used when generating satori image (as promise to work around sync in component)
   */
  fonts: Promise<SatoriOptions["fonts"]>
  /**
   * `GlobalConfiguration` of quartz (used for theme/typography)
   */
  cfg: GlobalConfiguration
  /**
   * full file data of current page
   */
  fileData: QuartzPluginData
}

export const og: SocialImageOptions["Component"] = (
  cfg: GlobalConfiguration,
  fileData: QuartzPluginData,
  { colorScheme }: Options,
  title: string,
  description: string,
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
        backgroundImage: `url("https://${cfg.baseUrl}/static/og-image.jpeg")`,
        backgroundSize: "100% 100%",
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
          height: "100%",
          width: "100%",
          flexDirection: "column",
          justifyContent: "flex-start",
          alignItems: "flex-start",
          gap: "1.5rem",
          paddingTop: "4rem",
          paddingBottom: "4rem",
          marginLeft: "4rem",
        }}
      >
        <img
          src={`https://${cfg.baseUrl}/static/icon.jpeg`}
          style={{
            position: "relative",
            backgroundClip: "border-box",
            borderRadius: "6rem",
          }}
          width={80}
          height={80}
        />
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            textAlign: "left",
            fontFamily: fonts[0].name,
          }}
        >
          <h2
            style={{
              color: cfg.theme.colors[colorScheme].light,
              fontSize: "3rem",
              fontWeight: 700,
              marginRight: "4rem",
              fontFamily: fonts[0].name,
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
            }}
          >
            {Li.map((item, index) => {
              if (item) {
                return <li key={index}>{item}</li>
              }
            })}
          </ul>
        </div>
        <p
          style={{
            color: cfg.theme.colors[colorScheme].light,
            fontSize: "1.5rem",
            overflow: "hidden",
            marginRight: "8rem",
            textOverflow: "ellipsis",
            display: "-webkit-box",
            WebkitLineClamp: 7,
            WebkitBoxOrient: "vertical",
            lineClamp: 7,
            fontFamily: fonts[1].name,
          }}
        >
          {description}
        </p>
      </div>
    </div>
  )
}

export const defaultImageOptions: SocialImageOptions = {
  colorScheme: "lightMode",
  height: 630,
  width: 1200,
  excludeRoot: true,
  Component: og,
}
