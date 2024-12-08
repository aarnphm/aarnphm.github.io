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
import { Root, ElementContent } from "hast"
import { filterEmbedTwitter } from "./twitter"
import { VFile } from "vfile"
import { s } from "hastscript"

interface Options {
  enableArxivEmbed: boolean
  enableRawEmbed: boolean
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
  ".js",
  ".yaml",
  ".yml",
  ".rs",
  ".m",
  ".sql",
  ".sh",
  ".txt",
]

export function extractArxivId(url: string): string | null {
  try {
    const urlObj = new URL(url)
    if (!urlObj.hostname.includes("arxiv.org")) return null

    // Match different arXiv URL patterns
    const patterns = [
      /arxiv.org\/abs\/(\d+\.\d+)/,
      /arxiv.org\/pdf\/(\d+\.\d+)(\.pdf)?/,
      /arxiv.org\/\w+\/(\d+\.\d+)/,
    ]

    for (const pattern of patterns) {
      const match = url.match(pattern)
      if (match) return match[1]
    }

    return null
  } catch (e) {
    return null
  }
}

export const CrawlLinks: QuartzTransformerPlugin<Partial<Options>> = (userOpts) => {
  const opts = { ...defaultOptions, ...userOpts }
  return {
    name: "LinkProcessing",
    htmlPlugins(ctx) {
      return [
        () => {
          return (tree: Root, file: VFile) => {
            const curSlug = simplifySlug(file.data.slug!)
            const outgoing: Set<SimpleSlug> = new Set()

            const transformOptions: TransformOptions = {
              strategy: opts.markdownLinkResolution,
              allSlugs: ctx.allSlugs,
            }

            visit(tree, "element", (node, index, parent) => {
              const classes = (node.properties.className ?? []) as string[]

              // rewrite all links
              if (
                node.tagName === "a" &&
                node.properties &&
                typeof node.properties.href === "string"
              ) {
                // insert a span element into node.children
                let dest = node.properties.href as RelativeURL
                const ext: string = path.extname(dest).toLowerCase()
                const isExternal =
                  opts.enableRawEmbed && ALLOWED_EXTENSIONS.includes(ext)
                    ? true
                    : isAbsoluteUrl(dest)

                // supports for rss feed and atom feed
                const isRSS = dest.includes("index.xml") || dest.includes("feed.xml")
                const isCslNode = classes.includes("csl-external-link")
                const isEmbedTwitter = filterEmbedTwitter(node)
                const isArxiv = node.properties.href.includes("arxiv.org")

                if (opts.enableArxivEmbed && isArxiv) {
                  classes.push("internal")
                  node.properties.dataArxivId = extractArxivId(node.properties.href)
                } else if (!isEmbedTwitter) {
                  classes.push(isExternal ? "external" : "internal")
                }

                // We will need to translate the link to external here
                if (isExternal && opts.enableRawEmbed) {
                  if (ALLOWED_EXTENSIONS.includes(ext) && !isAbsoluteUrl(dest)) {
                    classes.push("cdn-links")
                    dest = node.properties.href =
                      `https://cdn.aarnphm.xyz/assets/${dest}` as RelativeURL
                  }
                }

                if (
                  !isEmbedTwitter &&
                  !isCslNode &&
                  !isArxiv &&
                  isExternal &&
                  opts.externalLinkIcon
                ) {
                  node.children.push(
                    s(
                      "svg",
                      {
                        ariahidden: true,
                        class: "external-icon",
                        viewbox: "0 -12 24 24",
                        fill: "none",
                        stroke: "currentColor",
                        strokewidth: 1.5,
                        strokelinecap: "round",
                        strokelinejoin: "round",
                      },
                      [s("use", { href: "#arrow-ne" })],
                    ),
                  )
                }

                if (isRSS) {
                  classes.push("rss-link")
                }

                // special cases for parsing landing-links
                if (file.data.slug === "index") {
                  classes.push("landing-links")
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

                if ((isExternal && opts.openLinksInNewTab) || [".ipynb"].includes(ext)) {
                  node.properties.target = "_blank"
                }

                // don't process external links or intra-document anchors
                const isInternal = !(isAbsoluteUrl(dest) || dest.startsWith("#"))
                if (isInternal) {
                  dest = node.properties.href = transformLink(
                    file.data.slug!,
                    dest,
                    transformOptions,
                  )

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

                // add indicator spanContent after handling all prettyLinks
                const spanContent: ElementContent = {
                  properties: { className: "indicator-hook" },
                  type: "element",
                  tagName: "span",
                  children: [],
                }
                node.children = [spanContent, ...node.children]

                if (isRSS) {
                  parent!.children.splice(
                    index!,
                    1,
                    node,
                    s(
                      "svg",
                      {
                        class: "rss-icon",
                        version: "1.1",
                        xmlns: "http://www.w3.org/2000/svg",
                        viewbox: "0 0 8 8",
                        width: 8,
                        height: 8,
                        stroke: "none",
                      },
                      s("rect", { width: 8, height: 8, rx: 1.5, style: "fill:#F78422;" }),
                      s("circle", { cx: 2, cy: 6, r: 1, style: "fill:#FFFFFF;" }),
                      s("path", {
                        d: "m 1,4 a 3,3 0 0 1 3,3 h 1 a 4,4 0 0 0 -4,-4 z",
                        style: "fill:#FFFFFF;",
                      }),
                      s("path", {
                        d: "m 1,2 a 5,5 0 0 1 5,5 h 1 a 6,6 0 0 0 -6,-6 z",
                        style: "fill:#FFFFFF;",
                      }),
                    ),
                  )
                }
              }

              // transform all other resources that may use links
              if (
                ["img", "video", "audio", "iframe"].includes(node.tagName) &&
                node.properties &&
                typeof node.properties.src === "string"
              ) {
                if (opts.lazyLoad) {
                  node.properties.loading = "lazy"
                }

                if (!isAbsoluteUrl(node.properties.src)) {
                  let dest = node.properties.src as RelativeURL
                  dest = node.properties.src = transformLink(
                    file.data.slug!,
                    dest,
                    transformOptions,
                  )
                  node.properties.src = dest
                }
              }
            })

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
