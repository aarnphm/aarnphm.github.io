import satori, { FontWeight, SatoriOptions } from "satori/wasm"
import { GlobalConfiguration } from "../cfg"
import { JSXInternal } from "preact/src/jsx"
import { joinSegments } from "./path"

const headerFont = joinSegments("static", "GT-Sectra-Display-Regular.woff")
const bodyFont = joinSegments("static", "GT-Sectra-Book.woff")

export async function getSatoriFont(cfg: GlobalConfiguration): Promise<SatoriOptions["fonts"]> {
  const headerWeight: FontWeight = 700
  const bodyWeight: FontWeight = 400

  const url = new URL(`https://${cfg.baseUrl ?? "example.com"}`)

  const headerBuffer = await fetch(`${url.toString()}/${headerFont}`).then((res) =>
    res.arrayBuffer(),
  )
  const bodyBuffer = await fetch(`${url.toString()}/${bodyFont}`).then((res) => res.arrayBuffer())

  return [
    {
      name: cfg.theme.typography.header,
      data: headerBuffer,
      weight: headerWeight,
      style: "normal",
    },
    { name: cfg.theme.typography.body, data: bodyBuffer, weight: bodyWeight, style: "normal" },
  ]
}

export type SocialImageOptions = {
  /**
   * What color scheme to use for image generation (uses colors from config theme)
   */
  colorScheme: "lightMode" | "darkMode"
  /**
   * Height to generate image with in pixels (should be around 630px)
   */
  height: number
  /**
   * Width to generate image with in pixels (should be around 1200px)
   */
  width: number
  /**
   * JSX to use for generating image. See satori docs for more info (https://github.com/vercel/satori)
   * @param cfg global quartz config
   * @param userOpts options that can be set by user
   * @param title title of current page
   * @param description description of current page
   * @param fonts global font that can be used for styling
   * @returns prepared jsx to be used for generating image
   */
  imageStructure: (
    cfg: GlobalConfiguration,
    opts: Options,
    title: string,
    description: string,
    fonts: SatoriOptions["fonts"],
  ) => JSXInternal.Element
}

export type Options = Omit<SocialImageOptions, "imageStructure">

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
}

export const og: SocialImageOptions["imageStructure"] = (
  cfg: GlobalConfiguration,
  { colorScheme }: Options,
  title: string,
  description: string,
  fonts: SatoriOptions["fonts"],
) => {
  const fontBreakpoint = 22
  const useSmallerFont = title.length > fontBreakpoint

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "row",
        justifyContent: "flex-start",
        alignItems: "center",
        height: "100%",
        width: "100%",
        backgroundImage: `url("https://${cfg.baseUrl}/static/og-image.png")`,
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          height: "100%",
          width: "100%",
          backgroundColor: cfg.theme.colors[colorScheme].light,
          flexDirection: "column",
          gap: "2.5rem",
          paddingTop: "2rem",
          paddingBottom: "2rem",
        }}
      >
        <p
          style={{
            color: cfg.theme.colors[colorScheme].dark,
            fontSize: useSmallerFont ? 70 : 82,
            marginLeft: "4rem",
            textAlign: "center",
            marginRight: "4rem",
            fontFamily: fonts[0].name,
          }}
        >
          {title}
        </p>
        <p
          style={{
            color: cfg.theme.colors[colorScheme].dark,
            fontSize: 44,
            marginLeft: "8rem",
            marginRight: "8rem",
            lineClamp: 3,
            fontFamily: fonts[1].name,
          }}
        >
          {description}
        </p>
      </div>
    </div>
  )
}
