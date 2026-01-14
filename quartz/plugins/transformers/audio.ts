import { QuartzTransformerPlugin } from "../../types/plugin"
import { Element, Root } from "hast"
import { visit } from "unist-util-visit"
import { h, s } from "hastscript"
import { svgOptions } from "../../components/svg"

export interface Options {
  /**
   * enable custom audio player for all audio embeds
   * default: true
   */
  enableCustomPlayer: boolean
}

const defaultOptions: Options = {
  enableCustomPlayer: true,
}

/**
 * parse timestamp from various formats:
 * - "1m30s" -> 90
 * - "90s" -> 90
 * - "1:30" -> 90
 * - "90" -> 90
 */
function parseTimestamp(fragment: string): number | undefined {
  if (!fragment) return undefined

  // remove #t= prefix if present
  const clean = fragment.replace(/^#t=/, "")

  // format: 1m30s or 2h15m30s
  const complexMatch = clean.match(/(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?/)
  if (complexMatch) {
    const hours = parseInt(complexMatch[1] || "0")
    const minutes = parseInt(complexMatch[2] || "0")
    const seconds = parseInt(complexMatch[3] || "0")
    const total = hours * 3600 + minutes * 60 + seconds
    if (total > 0) return total
  }

  // format: 1:30 or 1:30:45
  if (clean.includes(":")) {
    const parts = clean.split(":").map((p) => parseInt(p))
    if (parts.length === 2) {
      // mm:ss
      return parts[0] * 60 + parts[1]
    } else if (parts.length === 3) {
      // hh:mm:ss
      return parts[0] * 3600 + parts[1] * 60 + parts[2]
    }
  }

  // format: just seconds
  const seconds = parseInt(clean)
  if (!isNaN(seconds) && seconds > 0) return seconds

  return undefined
}

/**
 * check if an audio element should be enhanced with custom player
 */
function shouldEnhanceAudio(node: Element): boolean {
  return (
    node.type === "element" &&
    node.tagName === "audio" &&
    typeof node.properties?.src === "string" &&
    node.properties.src.length > 0
  )
}

/**
 * create Substack-style audio player structure
 */
function createAudioPlayer(audioNode: Element): Element {
  const src = audioNode.properties.src as string
  const metadata = audioNode.properties["data-metadata"]
    ? JSON.parse(audioNode.properties["data-metadata"] as string)
    : {}

  // parse timestamp from src if present (e.g., audio.mp3#t=1m30s)
  let startTime: number | undefined
  if (src.includes("#t=")) {
    const [cleanSrc, fragment] = src.split("#t=")
    startTime = parseTimestamp(fragment)
    // update audio src to clean version
    audioNode.properties.src = cleanSrc
  }

  // create play/pause SVG icons
  const playIcon = s(
    "svg",
    {
      ...svgOptions,
      fill: "var(--dark)",
      stroke: "var(--dark)",
      class: "play-icon",
      viewBox: "0 -6 24 24",
      "aria-hidden": "true",
    },
    [
      s("path", {
        d: "M5.04688 18.5527C5.4375 18.5527 5.76953 18.3965 6.16016 18.1719L17.5469 11.5898C18.3574 11.1113 18.6406 10.7988 18.6406 10.2812C18.6406 9.76367 18.3574 9.45117 17.5469 8.98242L6.16016 2.39063C5.76953 2.16602 5.4375 2.01953 5.04688 2.01953C4.32422 2.01953 3.875 2.56641 3.875 3.41602V17.1465C3.875 17.9961 4.32422 18.5527 5.04688 18.5527Z",
        stroke: "none",
      }),
    ],
  )

  const pauseIcon = s(
    "svg",
    {
      ...svgOptions,
      fill: "var(--dark)",
      stroke: "var(--dark)",
      class: "pause-icon",
      viewBox: "0 -6 24 24",
      "aria-hidden": "true",
    },
    [
      s("path", {
        d: "M5.29883 17.9082H7.52539C8.375 17.9082 8.82422 17.459 8.82422 16.5996V3.29883C8.82422 2.41016 8.375 2 7.52539 2H5.29883C4.44922 2 4 2.44922 4 3.29883V16.5996C4 17.459 4.44922 17.9082 5.29883 17.9082ZM12.3984 17.9082H14.6152C15.4746 17.9082 15.9141 17.459 15.9141 16.5996V3.29883C15.9141 2.41016 15.4746 2 14.6152 2H12.3984C11.5391 2 11.0898 2.44922 11.0898 3.29883V16.5996C11.0898 17.459 11.5391 17.9082 12.3984 17.9082Z",
        stroke: "none",
      }),
    ],
  )

  // create play/pause button
  const playButton = h(
    "button.audio-play-button",
    {
      type: "button",
      ariaLabel: "Play",
      ariaPressed: "false",
    },
    [h("span.audio-icon-play", [playIcon]), h("span.audio-icon-pause", [pauseIcon])],
  )

  // create progress controls
  const progressWrapper = h(".audio-progress-wrapper", [
    h("span.audio-time-current", "0:00"),
    h(".audio-progress-container", [
      h(".audio-progress-track", [h(".audio-progress-bar")]),
      h(".audio-playhead"),
    ]),
    h("span.audio-time-remaining", "0:00"),
  ])

  // create speed control
  const speedButton = h(
    "button.audio-speed",
    {
      type: "button",
      ariaLabel: "Playback speed",
      ["data-speed"]: "1",
    },
    "1×",
  )

  const controlsContainer = h(".audio-controls", [progressWrapper])
  const extrasContainer = h(".audio-extras", [speedButton])

  // ensure audio element has necessary attributes
  audioNode.properties.preload = "metadata"
  audioNode.properties.controls = false // hide native controls

  // create main container
  const container = h(
    ".audio-embed",
    {
      ["data-audio-embed"]: "",
      ...(startTime !== undefined ? { ["data-start-time"]: startTime.toString() } : {}),
      ...(metadata && Object.keys(metadata).length > 0
        ? { ["data-metadata"]: JSON.stringify(metadata) }
        : {}),
    },
    [playButton, controlsContainer, extrasContainer, audioNode],
  )

  return container
}

export const Audio: QuartzTransformerPlugin<Partial<Options> | undefined> = (userOpts) => {
  const opts = { ...defaultOptions, ...userOpts }

  return {
    name: "Audio",
    htmlPlugins() {
      return [
        () => {
          return (tree: Root, _file) => {
            if (!opts.enableCustomPlayer) return

            visit(tree, "element", (node: Element, index, parent) => {
              if (shouldEnhanceAudio(node)) {
                const enhanced = createAudioPlayer(node)
                if (parent && typeof index === "number") {
                  parent.children[index] = enhanced
                }
              }
            })
          }
        },
      ]
    },
  }
}
