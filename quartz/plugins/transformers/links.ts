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
import { SKIP, visit } from "unist-util-visit"
import isAbsoluteUrl from "is-absolute-url"
import { ElementContent, Element } from "hast"
import { filterEmbedTwitter, twitterUrlRegex } from "./twitter"
import { h, s } from "hastscript"
import {
  bskySvg,
  githubSvg,
  substackSvg,
  svgOptions,
  twitterSvg,
} from "../../components/renderPage"

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
      /arxiv.org\/html\/(\d+\.\d+)/,
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

                // Initialize context object
                const ctx: LinkContext = {
                  classes,
                  dest,
                  ext,
                  isExternal:
                    opts.enableRawEmbed && ALLOWED_EXTENSIONS.includes(ext)
                      ? true
                      : isAbsoluteUrl(dest),
                  node,
                }

                // Link type checks
                const linkTypes = {
                  isCslNode: classes.includes("csl-external-link"),
                  isEmbedTwitter: filterEmbedTwitter(node),
                  isArxiv: dest.includes("arxiv.org"),
                  isWikipedia: dest.includes("wikipedia.org"),
                  isLessWrong: dest.includes("lesswrong.com"),
                  isGithub: dest.includes("github.com"),
                  isSubstack: dest.includes("substack.com"),
                  isTwitter: twitterUrlRegex.test(dest),
                  isBsky: dest.includes("bsky.app"),
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
                        `https://cdn.aarnphm.xyz/assets/${ctx.dest}` as RelativeURL
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
                      style: "height: 1em; margin-left: 0.25em;",
                    }),
                  )

                // Add appropriate icons based on link type
                if (!handleArxiv(ctx) && !linkTypes.isEmbedTwitter) {
                  ctx.classes.push(ctx.isExternal ? "external" : "internal")
                }

                handleCdnLinks(ctx)

                // Add appropriate icons
                if (linkTypes.isWikipedia) {
                  ctx.node.children.push(
                    createIconElement("/static/favicons/wikipedia.avif", "Wikipedia"),
                  )
                } else if (linkTypes.isLessWrong) {
                  ctx.node.children.push(
                    createIconElement("/static/favicons/lesswrong.avif", "LessWrong"),
                  )
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

            file.data.links = [...outgoing]
          }
        },
        // () => {
        //   const isFootnoteRef = ({ tagName, children }: Element) => {
        //     return (
        //       tagName === "sup" &&
        //       children.length === 1 &&
        //       (children[0] as Element).tagName === "a" &&
        //       (children[0] as Element).properties.dataFootnoteRef === ""
        //     )
        //   }
        //   return (tree, _file) => {
        //     let ol: Map<string, Element[]> | undefined = undefined
        //     visit(
        //       tree,
        //       (node) =>
        //         (node as Element).tagName === "section" &&
        //         (node as Element).properties?.dataFootnotes === "",
        //       (node) => {
        //         visit(node, { tagName: "ol" }, (n) => {
        //           ol = new Map<string, Element[]>(
        //             n.children
        //               .filter((el: Element) => el.tagName === "li")
        //               .map((el: Element) => [el.properties?.id as string, el.children]),
        //           )
        //           return SKIP
        //         })
        //       },
        //     )
        //     if (ol !== undefined) {
        //       visit(
        //         tree,
        //         (node) => isFootnoteRef(node as Element),
        //         (node) => {
        //           const link = node.children[0] as Element
        //           const key = (link.properties?.href as string).replace("#", "")
        //           const sideContents = ol?.get(key)
        //           node.children = [
        //             ...node.children,
        //             h("div.sidenotes", { id: key + "-sidenotes" }, sideContents),
        //           ]
        //         },
        //       )
        //     }
        //   }
        // },
      ]
    },
  }
}

declare module "vfile" {
  interface DataMap {
    links: SimpleSlug[]
  }
}
