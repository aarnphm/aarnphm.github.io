import { QuartzTransformerPlugin } from "../types"
import {
  FullSlug,
  RelativeURL,
  SimpleSlug,
  TransformOptions,
  stripSlashes,
  simplifySlug,
  splitAnchor,
  transformLink,
} from "../../util/path"
import path from "path"
import { visit } from "unist-util-visit"
import isAbsoluteUrl from "is-absolute-url"
import { Element } from "hast"
import { filterEmbedTwitter, twitterUrlRegex } from "./twitter"
import { h, s } from "hastscript"
import {
  anthropicSvg,
  bentomlSvg,
  ycSvg,
  bskySvg,
  doiSvg,
  githubSvg,
  substackSvg,
  svgOptions,
  twitterSvg,
  openaiSvg,
  hfSvg,
  obsidianSvg,
  youtubeSvg,
  gwernSvg,
} from "../../components/svg"
import type { FrontmatterLink } from "./frontmatter"

interface Options {
  enableArxivEmbed: boolean
  enableRawEmbed: boolean
  enableIndicatorHook: boolean
  /** How to resolve Markdown paths */
  markdownLinkResolution: TransformOptions["strategy"]
  /** Strips folders from a link so that it looks nice */
  prettyLinks: boolean
  openLinksInNewTab: boolean
  lazyLoad: boolean
  externalLinkIcon: boolean
}

const defaultOptions: Options = {
  enableArxivEmbed: false,
  enableRawEmbed: false,
  enableIndicatorHook: true,
  markdownLinkResolution: "absolute",
  prettyLinks: true,
  openLinksInNewTab: false,
  lazyLoad: false,
  externalLinkIcon: true,
}

const ALLOWED_EXTENSIONS = [
  ".py",
  ".go",
  ".java",
  ".c",
  ".cpp",
  ".cxx",
  ".cu",
  ".cuh",
  ".h",
  ".hpp",
  ".ts",
  ".tsx",
  ".yaml",
  ".yml",
  ".rs",
  ".m",
  ".sql",
  ".sh",
  ".txt",
]

/**
 * Match arXiv URLs such as:
 *  - https://arxiv.org/abs/1712.05877
 *  - https://arxiv.org/pdf/1712.05877.pdf
 *  - https://arxiv.org/html/1712.05877
 *  - https://arxiv.org/pdf/1712.05877v1.pdf
 */
const ARXIV_URL_REGEX =
  /^https?:\/\/arxiv\.org\/(?:abs|pdf|html)[\/\w.-]*?(\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?(?:[?#].*)?$/i

export function extractArxivId(url: string): string | null {
  try {
    const urlObj = new URL(url)
    if (!urlObj.hostname.includes("arxiv.org")) return null

    const match = url.match(ARXIV_URL_REGEX)
    return match ? match[1] : null
  } catch {
    return null
  }
}

interface LinkContext {
  classes: string[]
  dest: RelativeURL
  ext: string
  isExternal: boolean
  node: Element
}

export const CrawlLinks: QuartzTransformerPlugin<Partial<Options>> = (userOpts) => {
  const opts = { ...defaultOptions, ...userOpts }
  return {
    name: "LinkProcessing",
    htmlPlugins(ctx) {
      const { cfg } = ctx
      return [
        () => {
          return (tree, file) => {
            const curSlug = simplifySlug(file.data.slug!)
            const outgoing: Set<SimpleSlug> = new Set()

            const transformOptions: TransformOptions = {
              strategy: opts.markdownLinkResolution,
              allSlugs: ctx.allSlugs,
            }

            const shouldRewriteLinks = ({ tagName, properties }: Element) =>
              tagName === "a" && Boolean(properties.href) && typeof properties.href === "string"

            // rewrite all links
            //@ts-ignore
            visit(
              tree,
              (node: Element) => shouldRewriteLinks(node as Element),
              (node: Element) => {
                const classes = (node.properties.className ?? []) as string[]
                // insert a span element into node.children
                let dest = node.properties.href as RelativeURL
                const ext: string = path.extname(dest).toLowerCase()

                // Check for protocol-based URLs (mailto:, tel:, etc.)
                const hasProtocol = /^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(dest)

                // Initialize context object
                const ctx: LinkContext = {
                  classes,
                  dest,
                  ext,
                  isExternal:
                    opts.enableRawEmbed && ALLOWED_EXTENSIONS.includes(ext)
                      ? true
                      : isAbsoluteUrl(dest) || hasProtocol,
                  node,
                }

                // Link type checks
                const linkTypes = {
                  isApexDomain: dest.includes(cfg.configuration.baseUrl!),
                  isCslNode: classes.includes("csl-external-link"),
                  isEmbedTwitter: filterEmbedTwitter(node),
                  isArxiv: dest.includes("arxiv.org"),
                  isWikipedia: dest.includes("wikipedia.org"),
                  isLessWrong: dest.includes("lesswrong.com"),
                  isBentoml: dest.includes("bentoml.com"),
                  isYoutube: dest.includes("youtube.com"),
                  isGwern: dest.includes("gwern.net"),
                  isNeovim: dest.includes("neovim.io"),
                  isQuartz: dest.includes("quartz.jzhao.xyz"),
                  isObsidian: dest.includes("obsidian.md"),
                  isGithub: dest.includes("github.com"),
                  isSubstack: dest.includes("substack.com"),
                  isTwitter: twitterUrlRegex.test(dest),
                  isBsky: dest.includes("bsky.app"),
                  isDoi: dest.includes("doi.org"),
                  isOpenai: dest.includes("openai.com"),
                  isHf: dest.includes("huggingface.co"),
                  isYC: dest.includes("ycombinator.com"),
                  isAnthropic:
                    dest.includes("transformer-circuits.pub") || dest.includes("anthropic.com"),
                  isGoogleDocs: dest.includes("docs.google.com"),
                  isGoogleDrive: dest.includes("drive.google.com"),
                }

                if (
                  linkTypes.isWikipedia &&
                  node.children.length === 1 &&
                  node.children[0].type === "text" &&
                  node.children[0].value === dest
                ) {
                  try {
                    const u = new URL(dest)
                    const lang = u.hostname.split(".")[0]
                    const m = u.pathname.match(/\/wiki\/(.+)/)
                    if (m) {
                      node.children[0].value = `wikipedia/${lang !== "simple" ? lang + "/" : ""}${m[1]}`
                    }
                  } catch {}
                }

                if (
                  linkTypes.isYoutube &&
                  node.children.length === 1 &&
                  node.children[0].type === "text" &&
                  node.children[0].value === dest
                ) {
                  try {
                    const u = new URL(dest)
                    const vid = u.searchParams.get("v")
                    if (vid) {
                      node.children = [{ type: "text", value: `youtube/v=${vid}` }]
                    }
                  } catch {}
                }

                if (
                  linkTypes.isGoogleDocs &&
                  node.children.length === 1 &&
                  node.children[0].type === "text" &&
                  node.children[0].value === dest
                ) {
                  try {
                    const u = new URL(dest)
                    const m = u.pathname.match(/\/d\/([^/]+)/)
                    if (m) {
                      const id = m[1]
                      let displayId = id
                      if (id.length > 10) {
                        displayId = `${id.slice(0, 3)}[...]${id.slice(-3)}`
                      }
                      node.children[0].value = `docs.google.com/${displayId}`
                    }
                  } catch {}
                }

                if (
                  linkTypes.isGoogleDrive &&
                  node.children.length === 1 &&
                  node.children[0].type === "text" &&
                  node.children[0].value === dest
                ) {
                  try {
                    const u = new URL(dest)
                    const m = u.pathname.match(/\/d\/([^/]+)/)
                    if (m) {
                      const id = m[1]
                      node.children[0].value = `drive.google.com/${id}`
                    }
                  } catch {}
                }

                // Handle special link types
                const handleArxiv = (ctx: LinkContext) => {
                  if (opts.enableArxivEmbed && linkTypes.isArxiv) {
                    ctx.classes.push("internal")
                    ctx.node.properties.dataArxivId = extractArxivId(ctx.dest)
                    return true
                  }
                  return false
                }

                const handleCdnLinks = (ctx: LinkContext) => {
                  if (ctx.isExternal && opts.enableRawEmbed) {
                    if (ALLOWED_EXTENSIONS.includes(ctx.ext) && !isAbsoluteUrl(ctx.dest)) {
                      ctx.classes.push("cdn-links")
                      ctx.dest = ctx.node.properties.href =
                        `https://aarnphm.xyz/${ctx.dest}` as RelativeURL
                    }
                  }
                }

                const createIconElement = (src: string, alt: string) =>
                  h(
                    "span",
                    { style: "white-space: nowrap;" },
                    h("img.inline-icons", {
                      src,
                      alt,
                      style:
                        "height: 8px; width: 8px; margin-left: 3px; bottom: 2px; position: relative;",
                    }),
                  )

                // Add appropriate icons based on link type
                if (!handleArxiv(ctx) && !linkTypes.isEmbedTwitter) {
                  ctx.classes.push(ctx.isExternal ? "external" : "internal")
                }

                handleCdnLinks(ctx)

                // Add appropriate icons (skip if data-skip-icons is present)
                const skipIcons =
                  ctx.node.properties.dataSkipIcons === true ||
                  ctx.node.properties.dataSkipIcons === "true"
                if (!skipIcons) {
                  if (linkTypes.isWikipedia) {
                    ctx.node.children.push(
                      createIconElement("/static/favicons/wikipedia.svg", "Wikipedia"),
                    )
                  } else if (linkTypes.isApexDomain && file.data.slug! !== "index") {
                    ctx.node.children.push(createIconElement("/static/icon.webp", "apex"))
                  } else if (linkTypes.isArxiv) {
                    ctx.node.children.push(
                      createIconElement("/static/favicons/arxiv.avif", "arXiv"),
                    )
                  } else if (linkTypes.isLessWrong) {
                    ctx.node.children.push(
                      createIconElement("/static/favicons/lesswrong.avif", "LessWrong"),
                    )
                  } else if (linkTypes.isQuartz) {
                    ctx.node.children.push(
                      createIconElement("/static/favicons/quartz.png", "Quartz"),
                    )
                  } else if (linkTypes.isNeovim) {
                    ctx.node.children.push(
                      createIconElement("/static/favicons/neovim.svg", "Neovim"),
                    )
                  } else if (linkTypes.isBentoml) {
                    ctx.node.children.push(bentomlSvg)
                  } else if (linkTypes.isYoutube) {
                    ctx.node.children.push(youtubeSvg)
                  } else if (linkTypes.isGwern) {
                    ctx.node.children.push(gwernSvg)
                  } else if (linkTypes.isObsidian) {
                    ctx.node.children.push(obsidianSvg)
                  } else if (linkTypes.isYC) {
                    ctx.node.children.push(ycSvg)
                  } else if (linkTypes.isDoi) {
                    ctx.node.children.push(doiSvg)
                  } else if (linkTypes.isHf) {
                    ctx.node.children.push(hfSvg)
                  } else if (linkTypes.isAnthropic) {
                    ctx.node.children.push(anthropicSvg)
                  } else if (linkTypes.isOpenai) {
                    ctx.node.children.push(openaiSvg)
                  } else if (linkTypes.isGithub) {
                    ctx.node.children.push(githubSvg)
                  } else if (linkTypes.isSubstack) {
                    ctx.node.children.push(substackSvg)
                  } else if (linkTypes.isTwitter) {
                    ctx.node.children.push(twitterSvg)
                  } else if (linkTypes.isBsky) {
                    ctx.node.children.push(bskySvg)
                  } else if (
                    !linkTypes.isEmbedTwitter &&
                    !linkTypes.isCslNode &&
                    !linkTypes.isArxiv &&
                    ctx.isExternal &&
                    opts.externalLinkIcon
                  ) {
                    ctx.node.children.push(
                      s(
                        "svg",
                        {
                          ...svgOptions,
                          ariaHidden: true,
                          class: "external-icon",
                          viewbox: "0 -12 24 24",
                          fill: "none",
                          stroke: "currentColor",
                          strokewidth: 1.5,
                        },
                        [s("use", { href: "#arrow-ne" })],
                      ),
                    )
                  }
                }

                // Check if the link has alias text
                if (
                  node.children.length === 1 &&
                  node.children[0].type === "text" &&
                  node.children[0].value !== dest
                ) {
                  // Add the 'alias' class if the text content is not the same as the href
                  classes.push("alias")
                }
                node.properties.className = classes

                if ((ctx.isExternal && opts.openLinksInNewTab) || [".ipynb"].includes(ext)) {
                  node.properties.target = "_blank"
                }

                // don't process external links, protocol URLs, or intra-document anchors
                const isInternal = !(isAbsoluteUrl(dest) || dest.startsWith("#") || hasProtocol)
                if (isInternal) {
                  if (ext.includes("pdf")) {
                    // we use CF middleware for fetch from Git LFS, for now
                    dest = node.properties.href = `/${dest}` as RelativeURL
                  } else {
                    dest = node.properties.href = transformLink(
                      file.data.slug!,
                      dest,
                      transformOptions,
                    )
                  }

                  // url.resolve is considered legacy
                  // WHATWG equivalent https://nodejs.dev/en/api/v18/url/#urlresolvefrom-to
                  const url = new URL(dest, "https://base.com/" + stripSlashes(curSlug, true))
                  const canonicalDest = url.pathname
                  let [destCanonical, _destAnchor] = splitAnchor(canonicalDest)
                  if (destCanonical.endsWith("/")) {
                    destCanonical += "index"
                  }

                  // need to decodeURIComponent here as WHATWG URL percent-encodes everything
                  const full = decodeURIComponent(stripSlashes(destCanonical, true)) as FullSlug
                  const simple = simplifySlug(full)
                  outgoing.add(simple)
                  node.properties["data-slug"] = full
                }

                // rewrite link internals if prettylinks is on
                if (
                  opts.prettyLinks &&
                  isInternal &&
                  node.children.length === 1 &&
                  node.children[0].type === "text" &&
                  !node.children[0].value.startsWith("#")
                ) {
                  node.children[0].value = path.basename(node.children[0].value)
                }

                // add indicator hook after handling all prettyLinks, inspired by gwern
                if (opts.enableIndicatorHook) {
                  node.children = [h("span.indicator-hook"), ...node.children]
                }
              },
            )

            const shouldTransformResources = ({ tagName, properties }: Element) =>
              ["img", "video", "audio", "iframe"].includes(tagName) &&
              Boolean(properties.src) &&
              typeof properties.src === "string"

            // transform all other resources that may use links
            visit(
              tree,
              (node) => shouldTransformResources(node as Element),
              (node) => {
                if (opts.lazyLoad) node.properties.loading = "lazy"

                if (!isAbsoluteUrl(node.properties.src)) {
                  let dest = node.properties.src as RelativeURL
                  dest = node.properties.src = transformLink(
                    file.data.slug!,
                    dest,
                    transformOptions,
                  )
                  node.properties.src = dest
                }
              },
            )

            const fmLinks = file.data.frontmatterLinks as
              | Record<string, FrontmatterLink[]>
              | undefined
            if (fmLinks) {
              for (const links of Object.values(fmLinks)) {
                for (const link of links) {
                  outgoing.add(simplifySlug(link.slug))
                }
              }
            }

            file.data.links = [...outgoing]
          }
        },
      ]
    },
  }
}

declare module "vfile" {
  interface DataMap {
    links: SimpleSlug[]
  }
}
