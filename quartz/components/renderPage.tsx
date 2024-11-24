import { render } from "preact-render-to-string"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import HeaderConstructor from "./Header"
import BodyConstructor from "./Body"
import ContentConstructor from "./pages/Content"
import MetaConstructor from "./Meta"
import SpacerConstructor from "./Spacer"
import { byDateAndAlphabetical } from "./PageList"
import { getDate, formatDate } from "./Date"
import { classNames } from "../util/lang"
import { JSResourceToScriptElement, StaticResources } from "../util/resources"
import {
  clone,
  FullSlug,
  SimpleSlug,
  RelativeURL,
  joinSegments,
  normalizeHastElement,
  resolveRelative,
} from "../util/path"
import { visit } from "unist-util-visit"
import { Root, Element, ElementContent, Node } from "hast"
import { GlobalConfiguration } from "../cfg"
import { i18n } from "../i18n"
import { JSX } from "preact"
import { headingRank } from "hast-util-heading-rank"
import type { TranscludeOptions } from "../plugins/transformers/frontmatter"
import { QuartzPluginData } from "../plugins/vfile"
// @ts-ignore
import collapseHeaderScript from "./scripts/collapse-header.inline.ts"
import collapseHeaderStyle from "./styles/collapseHeader.inline.scss"

interface RenderComponents {
  head: QuartzComponent
  header: QuartzComponent[]
  beforeBody: QuartzComponent[]
  pageBody: QuartzComponent
  afterBody: QuartzComponent[]
  left: QuartzComponent[]
  right: QuartzComponent[]
  footer: QuartzComponent
}

function headerElement(
  node: Element,
  content: ElementContent[],
  idx: number,
  endHr: boolean,
): Element {
  const buttonId = `collapsible-header-${node.properties?.id ?? idx}`

  const id = node.properties?.id ?? idx
  // indicate whether the header is collapsed or not
  const lastIdx = node.children.length > 0 ? node.children.length - 1 : 0
  const icons: Element = {
    type: "element",
    tagName: "svg",
    properties: {
      "aria-hidden": "true",
      xmlns: "http://www.w3.org/2000/svg",
      width: 18,
      height: 18,
      viewBox: "0 0 24 24",
      fill: "currentColor",
      stroke: "currentColor",
      "stroke-width": "0",
      "stroke-linecap": "round",
      "stroke-linejoin": "round",
      style: "padding-left: 0.2rem;",
      className: ["collapsed-dots"],
    },
    children: [
      {
        type: "element",
        tagName: "circle",
        properties: {
          cx: "6",
          cy: "12",
          r: "2",
        },
        children: [],
      },
      {
        type: "element",
        tagName: "circle",
        properties: {
          cx: "12",
          cy: "12",
          r: "2",
        },
        children: [],
      },
      {
        type: "element",
        tagName: "circle",
        properties: {
          cx: "18",
          cy: "12",
          r: "2",
        },
        children: [],
      },
    ],
  }
  node.children.splice(lastIdx, 0, icons)

  let className = ["collapsible-header"]
  if (endHr) {
    className.push("end-hr")
  }

  const rank = headingRank(node) as number

  return {
    type: "element",
    tagName: "div",
    properties: { className, id, "data-level": `${rank}` },
    children: [
      {
        type: "element",
        tagName: "div",
        properties: {
          className: ["header-controls"],
        },
        children: [
          // Toggle button
          {
            type: "element",
            tagName: "button",
            properties: {
              id: `${buttonId}-toggle`,
              ariaLabel: "Toggle content visibility",
              ariaExpanded: true,
              className: ["toggle-button"],
            },
            children: [
              {
                type: "element",
                tagName: "div",
                properties: {
                  className: ["toggle-icons"],
                },
                children: [
                  // default circle icon
                  {
                    type: "element",
                    tagName: "svg",
                    properties: {
                      xmlns: "http://www.w3.org/2000/svg",
                      width: 18,
                      height: 18,
                      viewBox: "0 0 24 24",
                      fill: "var(--dark)",
                      stroke: "var(--dark)",
                      "stroke-width": "2",
                      "stroke-linecap": "round",
                      "stroke-linejoin": "round",
                      className: ["circle-icon"],
                    },
                    children: [
                      {
                        type: "element",
                        tagName: "circle",
                        properties: {
                          cx: "12",
                          cy: "12",
                          r: "3",
                        },
                        children: [],
                      },
                    ],
                  },
                  // expand icon
                  {
                    type: "element",
                    tagName: "svg",
                    properties: {
                      xmlns: "http://www.w3.org/2000/svg",
                      width: 18,
                      height: 18,
                      viewBox: "0 0 24 24",
                      fill: "var(--tertiary)",
                      stroke: "var(--tertiary)",
                      "stroke-width": "2",
                      "stroke-linecap": "round",
                      "stroke-linejoin": "round",
                      className: ["expand-icon"],
                    },
                    children: [
                      {
                        type: "element",
                        tagName: "line",
                        properties: {
                          x1: "12",
                          y1: "5",
                          x2: "12",
                          y2: "19",
                        },
                        children: [],
                      },
                      {
                        type: "element",
                        tagName: "line",
                        properties: {
                          x1: "5",
                          y1: "12",
                          x2: "19",
                          y2: "12",
                        },
                        children: [],
                      },
                    ],
                  },
                  // collapse icon
                  {
                    type: "element",
                    tagName: "svg",
                    properties: {
                      xmlns: "http://www.w3.org/2000/svg",
                      width: 18,
                      height: 18,
                      viewBox: "0 0 24 24",
                      fill: "none",
                      stroke: "currentColor",
                      "stroke-width": "2",
                      "stroke-linecap": "round",
                      "stroke-linejoin": "round",
                      className: ["collapse-icon"],
                    },
                    children: [
                      {
                        type: "element",
                        tagName: "line",
                        properties: {
                          x1: "5",
                          y1: "12",
                          x2: "19",
                          y2: "12",
                        },
                        children: [],
                      },
                    ],
                  },
                ],
              },
            ],
          },
          node,
        ],
      },
      {
        type: "element",
        tagName: "div",
        properties: { className: ["collapsible-header-content-outer"] },
        children: [
          {
            type: "element",
            tagName: "div",
            properties: {
              className: ["collapsible-header-content"],
              ["data-references"]: `${buttonId}-toggle`,
              ["data-level"]: `${rank}`,
              ["data-heading-id"]: node.properties.id, // HACK: This assumes that rehype-slug already runs this target
            },
            children: content,
          },
        ],
      },
    ],
  }
}

function shouldStopWrapping(node: ElementContent) {
  if (node.type === "element") {
    if (
      node.properties?.dataReferences === "" ||
      node.properties?.dataFootnotes === "" ||
      node.properties?.dataBacklinks === ""
    ) {
      return true
    }
    if (node.tagName === "hr") {
      return true
    }
  }
  return false
}

interface StackElement {
  level: number
  element: Element
  content: ElementContent[]
}

function processHeaders(nodes: ElementContent[]): ElementContent[] {
  let result: ElementContent[] = []

  let stack: StackElement[] = []
  for (const node of nodes) {
    if (shouldStopWrapping(node)) {
      const endHr = (node as Element).tagName === "hr"
      // Close any open sections
      while (stack.length > 0) {
        const completedSection = stack.pop()!
        const wrappedElement = headerElement(
          completedSection.element,
          completedSection.content,
          0,
          endHr,
        )
        if (stack.length > 0) {
          stack[stack.length - 1].content.push(wrappedElement)
        } else {
          result.push(wrappedElement)
        }
      }
      // Add the node to the result
      result.push(node)
    } else if (node.type === "element" && headingRank(node)) {
      const level = headingRank(node) as number

      // Pop from stack until the top has level less than current
      while (stack.length > 0 && stack[stack.length - 1].level >= level) {
        const completedSection = stack.pop()!
        const wrappedElement = headerElement(
          completedSection.element,
          completedSection.content,
          0,
          false,
        )
        if (stack.length > 0) {
          stack[stack.length - 1].content.push(wrappedElement)
        } else {
          result.push(wrappedElement)
        }
      }

      // Start a new section
      stack.push({ level, element: node as Element, content: [] })
    } else {
      // Content node
      if (stack.length > 0) {
        stack[stack.length - 1].content.push(node)
      } else {
        result.push(node)
      }
    }
  }

  // Close any remaining sections
  while (stack.length > 0) {
    const completedSection = stack.pop()!
    const wrappedElement = headerElement(
      completedSection.element,
      completedSection.content,
      0,
      false,
    )
    if (stack.length > 0) {
      stack[stack.length - 1].content.push(wrappedElement)
    } else {
      result.push(wrappedElement)
    }
  }

  return result
}

function mergeReferences(root: Root, appendSuffix?: string | undefined): void {
  const finalRefs: Element[] = []
  const toRemove: Element[] = []

  // visit all references with bib to update suffix
  visit(root, "element", (node: Element) => {
    if (node.tagName === "a" && (node.properties?.href as string)?.startsWith("#bib")) {
      node.properties.href = `${node.properties.href}${appendSuffix !== undefined ? "-" + appendSuffix : ""}`
    }
  })

  // Find all reference divs and collect their entries
  visit(root, "element", (node: Element) => {
    if (
      node.type === "element" &&
      node.tagName === "section" &&
      node.properties.dataReferences == ""
    ) {
      toRemove.push(node)
      const items = (node.children as Element[]).filter((val) => val.tagName === "ul")[0] // The ul is in here
      finalRefs.push(
        ...(items.children as Element[]).map((li) => {
          li.properties.id = `${li.properties?.id}${appendSuffix ? "-" + appendSuffix : ""}`
          return li
        }),
      )
    }
  })

  // we don't want to remove the last nodes
  toRemove.pop()
  if (toRemove.length === 0) return

  // Remove all reference divs except the last one
  visit(root, "element", (node: Element, index, parent) => {
    if (toRemove.includes(node)) {
      parent!.children.splice(index as number, 1)
    }
  })

  // finally, update the final position
  visit(root, "element", (node: Element, index, parent) => {
    if (
      node.type === "element" &&
      node.tagName === "section" &&
      node.properties.dataReferences == ""
    ) {
      // @ts-ignore
      node.children[1].children = finalRefs
      parent!.children.splice(index as number, 1, node)
    }
  })
}

interface Note {
  href: string
  id: string
}

function mergeFootnotes(root: Root, appendSuffix?: string | undefined): void {
  const orderNotes: Note[] = []
  const finalRefs: Element[] = []
  const toRemove: Element[] = []

  let idx = 0
  visit(root, "element", (node: Element) => {
    if (node.type === "element" && node.tagName === "a" && node.properties.dataFootnoteRef === "") {
      orderNotes.push({ href: node.properties.href as string, id: node.properties.id as string })
      node.properties.href = `${node.properties.href}${appendSuffix !== undefined ? "-" + appendSuffix : ""}`
      node.properties.id =
        node.properties.id + `${appendSuffix !== undefined ? "-" + appendSuffix : ""}`
      visit(node, "text", (node) => {
        node.value = `${idx + 1}`
        idx++
      })
    }
  })

  visit(root, "element", (node: Element) => {
    if (
      node.type === "element" &&
      node.tagName === "section" &&
      node.properties.dataFootnotes == ""
    ) {
      toRemove.push(node)
      const items = (node.children as Element[]).filter((val) => val.tagName === "ol")[0] // The ol is in here
      finalRefs.push(...(items.children as Element[]))
    }
  })

  // we don't want to remove the last nodes
  toRemove.pop()
  if (orderNotes.length === 0) return

  // Remove all reference divs except the last one
  visit(root, "element", (node: Element, index, parent) => {
    if (toRemove.includes(node)) {
      parent!.children.splice(index as number, 1)
    }
  })

  // Sort finalRefs based on orderNotes
  const sortedRefs = (
    orderNotes
      .map(({ href }: Note) => {
        // Remove the '#' from the href to match with footnote IDs
        const noteId = href.replace("#", "")
        return finalRefs.find((ref) => {
          return ref.properties?.id === noteId
        })
      })
      .filter(Boolean) as Element[]
  ).map((ref) => {
    const transclude = ref.properties?.id
    ref.properties!.id = `${transclude}${appendSuffix ? "-" + appendSuffix : ""}`
    visit(ref, "element", (c) => {
      if (c.tagName === "a" && c.properties.dataFootnoteBackref == "") {
        c.properties.href = `${c.properties.href}${appendSuffix !== undefined ? "-" + appendSuffix : ""}`
      }
    })
    return ref
  })

  // finally, update the final position
  visit(root, "element", (node: Element) => {
    if (
      node.type === "element" &&
      node.tagName === "section" &&
      node.properties.dataFootnotes == ""
    ) {
      // HACK: The node.children will have length 4, and ol is the 3rd items
      const ol = node.children[2] as Element
      ol.children = sortedRefs
    }
  })
}

export function mergeIsomorphic(ast: Node, suffix?: string) {
  mergeReferences(ast as Root, suffix)
  mergeFootnotes(ast as Root, suffix)
}

export function pageResources(
  baseDir: FullSlug | RelativeURL,
  staticResources: StaticResources,
): StaticResources {
  const contentIndexPath = joinSegments(baseDir, "static/contentIndex.json")
  const contentIndexScript = `const fetchData = fetch("${contentIndexPath}").then(data => data.json())`

  return {
    css: [
      { content: joinSegments(baseDir, "index.css") },
      {
        content: collapseHeaderStyle,
        inline: true,
      },
      ...staticResources.css,
    ],
    js: [
      {
        src: joinSegments(baseDir, "prescript.js"),
        loadTime: "beforeDOMReady",
        contentType: "external",
      },
      {
        loadTime: "beforeDOMReady",
        contentType: "inline",
        spaPreserve: true,
        script: contentIndexScript,
      },
      {
        script: collapseHeaderScript,
        loadTime: "beforeDOMReady",
        contentType: "inline",
      },
      ...staticResources.js,
      {
        src: joinSegments(baseDir, "postscript.js"),
        loadTime: "afterDOMReady",
        moduleType: "module",
        contentType: "external",
      },
    ],
  }
}

const defaultTranscludeOptions: TranscludeOptions = { dynalist: true, title: true }

export function transcludeFinal(
  root: Root,
  { cfg, allFiles, fileData }: QuartzComponentProps,
  userOpts?: Partial<TranscludeOptions>,
): Root {
  // NOTE: return early these cases, we probably don't want to transclude them anw
  if (fileData.frontmatter?.poem || fileData.frontmatter?.menu) return root

  // hierarchy of transclusion: frontmatter > userOpts > defaultOpts
  const slug = fileData.slug as FullSlug
  let opts: TranscludeOptions
  if (userOpts) {
    opts = { ...defaultTranscludeOptions, ...userOpts }
  } else {
    opts = defaultTranscludeOptions
  }

  if (fileData.frontmatter?.transclude) {
    opts = { ...opts, ...fileData.frontmatter?.transclude }
  }

  const { dynalist } = opts

  const anchor = (href: string, url: string, description: string): Element => {
    const metadata: Element[] = [
      {
        type: "element",
        tagName: "li",
        properties: {
          style: {
            "font-style": "italic",
            color: "var(--gray)",
          },
        },
        children: [{ type: "text", value: `url: ${url}` }],
      },
    ]

    if (description !== "undefined") {
      metadata.push({
        type: "element",
        tagName: "li",
        properties: {},
        children: [
          {
            type: "element",
            tagName: "span",
            properties: {
              style: { "text-decoration": "underline" },
            },
            children: [{ type: "text", value: `description` }],
          },
          { type: "text", value: `: ${description}` },
        ],
      })
    }

    return {
      type: "element",
      tagName: "div",
      properties: { className: ["transclude-ref"], "data-href": href },
      children: [
        {
          type: "element",
          tagName: "ul",
          properties: { className: ["metadata"] },
          children: metadata,
        },
        {
          type: "element",
          tagName: "svg",
          properties: {
            className: ["blockquote-link"],
            width: 16,
            height: 16,
            viewBox: "0 0 24 24",
            fill: "none",
            stroke: "currentColor",
            "stroke-width": "2",
            "stroke-linecap": "round",
            "stroke-linejoin": "round",
          },
          children: [
            {
              type: "element",
              tagName: "path",
              properties: {
                d: "M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71",
              },
              children: [],
            },
            {
              type: "element",
              tagName: "path",
              properties: {
                d: "M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71",
              },
              children: [],
            },
          ],
        },
      ],
    }
  }

  // NOTE: process transcludes in componentData
  visit(root, "element", (node) => {
    if (node.tagName === "blockquote") {
      const classNames = (node.properties?.className ?? []) as string[]
      const url = node.properties.dataUrl as string
      const alias = node.properties?.dataEmbedAlias
      if (classNames.includes("transclude")) {
        const inner = node.children[0] as Element
        const transcludeTarget = inner.properties["data-slug"] as FullSlug
        const page = allFiles.find((f) => f.slug === transcludeTarget)
        if (!page) {
          return
        }
        let transcludePageOpts: TranscludeOptions
        if (page.frontmatter?.transclude) {
          transcludePageOpts = { ...opts, ...page.frontmatter?.transclude }
        } else {
          transcludePageOpts = opts
        }

        const { title } = transcludePageOpts

        let blockRef = node.properties.dataBlock as string | undefined
        if (blockRef?.startsWith("#^")) {
          // block transclude
          blockRef = blockRef.slice("#^".length)
          let blockNode = page.blocks?.[blockRef]
          if (blockNode) {
            if (blockNode.tagName === "li") {
              blockNode = {
                type: "element",
                tagName: "ul",
                properties: {},
                children: [blockNode],
              }
            }

            node.children = [
              anchor(inner.properties?.href as string, url, alias),
              normalizeHastElement(blockNode, slug, transcludeTarget),
              {
                type: "element",
                tagName: "a",
                properties: { href: inner.properties?.href, class: ["internal", "transclude-src"] },
                children: [
                  { type: "text", value: i18n(cfg.locale).components.transcludes.linkToOriginal },
                ],
              },
            ]
          }
        } else if (blockRef?.startsWith("#") && page.htmlAst) {
          // header transclude
          blockRef = blockRef.slice(1)
          let startIdx = undefined
          let startDepth = undefined
          let endIdx = undefined
          for (const [i, el] of page.htmlAst.children.entries()) {
            // skip non-headers
            if (!(el.type === "element" && headingRank(el))) continue
            const depth = headingRank(el) as number

            // looking for our blockref
            if (startIdx === undefined || startDepth === undefined) {
              // skip until we find the blockref that matches
              if (el.properties?.id === blockRef) {
                startIdx = i
                startDepth = depth
              }
            } else if (depth <= startDepth) {
              // looking for new header that is same level or higher
              endIdx = i
              break
            }
          }

          if (startIdx === undefined) {
            return
          }

          node.children = [
            anchor(inner.properties?.href as string, url, alias),
            ...(page.htmlAst.children.slice(startIdx, endIdx) as ElementContent[]).map((child) =>
              normalizeHastElement(child as Element, slug, transcludeTarget),
            ),
            {
              type: "element",
              tagName: "a",
              properties: { href: inner.properties?.href, class: ["internal", "transclude-src"] },
              children: [
                { type: "text", value: i18n(cfg.locale).components.transcludes.linkToOriginal },
              ],
            },
          ]
        } else if (page.htmlAst) {
          // page transclude
          node.children = [
            anchor(inner.properties?.href as string, url, alias),
            title
              ? {
                  type: "element",
                  tagName: "h1",
                  properties: {},
                  children: [
                    {
                      type: "text",
                      value:
                        page.frontmatter?.title ??
                        i18n(cfg.locale).components.transcludes.transcludeOf({
                          targetSlug: page.slug!,
                        }),
                    },
                  ],
                }
              : ({} as ElementContent),
            ...(page.htmlAst.children as ElementContent[]).map((child) =>
              normalizeHastElement(child as Element, slug, transcludeTarget),
            ),
            {
              type: "element",
              tagName: "a",
              properties: { href: inner.properties?.href, class: ["internal", "transclude-src"] },
              children: [
                { type: "text", value: i18n(cfg.locale).components.transcludes.linkToOriginal },
              ],
            },
          ]
        }
      }
    }
  })

  // NOTE: handling collapsible nodes
  if (dynalist && !slug.includes("posts")) {
    root.children = processHeaders(root.children as ElementContent[])
  }

  // NOTE: We then merge all references and footnotes to final items
  mergeIsomorphic(root)
  return root
}

export const HyperAlias = {
  livres: "/books",
  "boîte aux lettres": "/posts/",
  projets: "/thoughts/work",
  advices: "/quotes",
  parfum: "/thoughts/Scents",
  "atelier with friends": "/thoughts/atelier-with-friends",
}

export const SocialAlias = {
  github: "https://github.com/aarnphm",
  twitter: "https://x.com/aarnphm_",
  substack: "https://livingalonealone.com",
  curius: "/curius",
  contact: "mailto:contact@aarnphm.xyz",
  "site source": "https://github.com/aarnphm/sites",
  "llms.txt": "/llms.txt",
  "llms-full.txt": "/llms-full.txt",
}

type AliasLinkProp = {
  name?: string
  url?: string
  isInternal?: boolean
  newTab?: boolean | ((name: string) => boolean)
  enablePopover?: boolean
}

const AliasLink = (props: AliasLinkProp) => {
  const opts = { isInternal: false, newTab: false, enablePopover: true, ...props }
  const className = ["landing-links"]
  if (opts.isInternal && opts.enablePopover) className.push("internal")
  return (
    <a
      href={opts.url}
      target={opts.newTab ? "_blank" : "_self"}
      rel="noopener noreferrer"
      className={className.join(" ")}
    >
      {opts.name}
    </a>
  )
}

const NotesComponent = ((opts?: { slug: SimpleSlug; numLimits?: number; header?: string }) => {
  const Spacer = SpacerConstructor()

  const Notes: QuartzComponent = (componentData: QuartzComponentProps) => {
    const { allFiles, fileData, cfg } = componentData
    const pages = allFiles
      .filter((f: QuartzPluginData) => {
        if (f.slug!.startsWith(opts!.slug)) {
          return (
            !["university", "tags", "index", ...cfg.ignorePatterns].some((it) =>
              (f.slug as FullSlug).includes(it),
            ) && !f.frontmatter?.noindex
          )
        }
        return false
      })
      .sort((a: QuartzPluginData, b: QuartzPluginData): number => {
        const afm = a.frontmatter!
        const bfm = b.frontmatter!
        if (afm.priority && bfm.priority) {
          return afm.priority - bfm.priority
        } else if (afm.priority && !bfm.priority) {
          return -1
        } else if (!afm.priority && bfm.priority) {
          return 1
        }
        return byDateAndAlphabetical(cfg)(a, b)
      })

    const remaining = Math.max(0, pages.length - opts!.numLimits!)
    const classes = ["min-links", "internal"].join(" ")
    return (
      <div id="note-item">
        <h2>{opts!.header}.</h2>
        <div class="notes-container">
          <div class="recent-links">
            <ul class="landing-notes">
              {pages.slice(0, opts!.numLimits).map((page) => {
                const title = page.frontmatter?.title ?? i18n(cfg.locale).propertyDefaults.title

                return (
                  <li>
                    <a href={resolveRelative(fileData.slug!, page.slug!)} class={classes}>
                      <div class="landing-meta">
                        <span class="landing-mspan">
                          {formatDate(getDate(cfg, page)!, cfg.locale)}
                        </span>
                        <span class="landing-mtitle">{title}</span>
                      </div>
                    </a>
                  </li>
                )
              })}
            </ul>
            {remaining > 0 && (
              <p>
                <em>
                  <a href={resolveRelative(fileData.slug!, opts!.slug)} class={classes}>
                    {i18n(cfg.locale).components.recentNotes.seeRemainingMore({
                      remaining,
                    })}
                  </a>
                </em>
              </p>
            )}
          </div>
          <Spacer {...componentData} />
        </div>
      </div>
    )
  }
  return Notes
}) satisfies QuartzComponentConstructor

const ClickableContainer = (props: {
  title: string
  links: Record<string, string>
  cfg: AliasLinkProp
}) => {
  const { title, links, cfg } = props
  let newTab: boolean | undefined

  return (
    <>
      <h2>{title}:</h2>
      <div class="clickable-container">
        {Object.entries(links).map(([name, url]) => {
          if (typeof cfg.newTab === "function") {
            newTab = cfg.newTab(name)
          } else {
            newTab = cfg.newTab
          }
          return <AliasLink key={name} {...cfg} name={name} url={url} newTab={newTab} />
        })}
      </div>
    </>
  )
}

const HyperlinksComponent = ((props?: { children: JSX.Element[] }) => {
  const { children } = props ?? { children: [] }

  const Hyperlink: QuartzComponent = () => <div class="hyperlinks">{children}</div>
  return Hyperlink
}) satisfies QuartzComponentConstructor

const ElementComponent = (() => {
  const Content = ContentConstructor()
  const RecentNotes = NotesComponent({
    header: "récentes",
    slug: "thoughts/" as SimpleSlug,
    numLimits: 12,
  })
  const RecentPosts = NotesComponent({
    header: "écriture",
    slug: "posts/" as SimpleSlug,
    numLimits: 6,
  })
  const Hyperlink = HyperlinksComponent({
    children: [
      ClickableContainer({
        title: "jardin",
        links: HyperAlias,
        cfg: { isInternal: true, newTab: false },
      }),
      ClickableContainer({
        title: "média",
        links: SocialAlias,
        cfg: { isInternal: false, newTab: (name) => name !== "curius" },
      }),
    ],
  })

  const Element: QuartzComponent = (componentData: QuartzComponentProps) => {
    return (
      <div class="content-container">
        <Content {...componentData} />
        <div class="notes-outer">
          <RecentNotes {...componentData} />
          <RecentPosts {...componentData} />
        </div>
        <Hyperlink {...componentData} />
      </div>
    )
  }

  return Element
}) satisfies QuartzComponentConstructor

export function renderPage(
  cfg: GlobalConfiguration,
  slug: FullSlug,
  componentData: QuartzComponentProps,
  components: RenderComponents,
  pageResources: StaticResources,
): string {
  // make a deep copy of the tree so we don't remove the transclusion references
  // for the file cached in contentMap in build.ts
  const root = clone(componentData.tree) as Root
  // NOTE: set componentData.tree to the edited html that has transclusions rendered
  componentData.tree = transcludeFinal(root, componentData)

  if (slug === "index") {
    components = {
      ...components,
      header: [
        () => (
          <h1 class="article-title" style="margin-top: 0" lang="fr">
            Bonjour, je suis Aaron.
          </h1>
        ),
        MetaConstructor(),
      ],
      left: [SpacerConstructor()],
      right: [SpacerConstructor()],
      afterBody: [],
      beforeBody: [],
      pageBody: ({ displayClass }: QuartzComponentProps) => {
        const Element = ElementComponent()

        return (
          <div class={classNames(displayClass, "landing")}>
            <Element {...componentData} />
          </div>
        )
      },
    }
  }

  const {
    head: Head,
    header,
    beforeBody,
    pageBody: Content,
    afterBody,
    left,
    right,
    footer: Footer,
  } = components
  const Header = HeaderConstructor()
  const Body = BodyConstructor()

  const LeftComponent =
    left.length > 0 ? (
      <div class="left sidebar">
        {left.map((BodyComponent) => (
          <BodyComponent {...componentData} />
        ))}
      </div>
    ) : (
      <></>
    )

  const RightComponent = (
    <div class="right sidebar">
      {right.map((BodyComponent) => (
        <BodyComponent {...componentData} />
      ))}
    </div>
  )

  // TODO: https://thesolarmonk.com/posts/a-spacebar-for-the-web style
  const lang = componentData.fileData.frontmatter?.lang ?? cfg.locale?.split("-")[0] ?? "en"
  const doc = (
    <html lang={lang}>
      <Head {...componentData} />
      <body
        data-slug={slug}
        data-language={lang}
        data-menu={componentData.fileData.frontmatter?.menu ?? false}
        data-poem={componentData.fileData.frontmatter?.poem ?? false}
      >
        <main id="quartz-root" class="page">
          <Body {...componentData}>
            {LeftComponent}
            <div class="center">
              <div class="page-header">
                <Header {...componentData}>
                  {header.map((HeaderComponent) => (
                    <HeaderComponent {...componentData} />
                  ))}
                </Header>
                <div class="popover-hint">
                  {beforeBody.map((BodyComponent) => (
                    <BodyComponent {...componentData} />
                  ))}
                </div>
              </div>
              <Content {...componentData} />
              <div class="page-footer">
                {afterBody.map((BodyComponent) => (
                  <BodyComponent {...componentData} />
                ))}
              </div>
            </div>
            {RightComponent}
            <Footer {...componentData} />
          </Body>
        </main>
      </body>
      {pageResources.js
        .filter((resource) => resource.loadTime === "afterDOMReady")
        .map((res) => JSResourceToScriptElement(res))}
    </html>
  )

  return (
    "<!DOCTYPE html>\n" +
    `<!--
/*************************************************************************
* Bop got your nose !!!
*
* Hehe
*
* Anw if you see a component you like ping @aarnphm on Discord I can try
* to send it your way. Have a wonderful day!
**************************************************************************/
-->` +
    render(doc)
  )
}
