import { render } from "preact-render-to-string"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import HeaderConstructor from "./Header"
import ContentConstructor from "./pages/Content"
import FooterConstructor from "./Footer"
import Search from "./Search"
import Graph from "./Graph"
import Palette from "./Palette"
import Image from "./Image"
import { byDateAndAlphabetical } from "./PageList"
import { getDate, Date as DateComponent } from "./Date"
import { classNames } from "../util/lang"
import { JSResourceToScriptElement, StaticResources } from "../util/resources"
import {
  FullSlug,
  SimpleSlug,
  RelativeURL,
  joinSegments,
  normalizeHastElement,
  resolveRelative,
} from "../util/path"
import { clone } from "../util/clone"
import { githubSvg, substackSvg, bskySvg, twitterSvg, svgOptions, QuartzIcon } from "./svg"
import { EXIT, visit } from "unist-util-visit"
import { Root, Element, ElementContent, Node, Text } from "hast"
import { i18n } from "../i18n"
import { JSX } from "preact"
import { headingRank } from "hast-util-heading-rank"
import type { TranscludeOptions } from "../plugins/transformers/frontmatter"
import { QuartzPluginData } from "../plugins/vfile"
import { h, s } from "hastscript"
// @ts-ignore
import collapseHeaderScript from "./scripts/collapse-header.inline.ts"
import collapseHeaderStyle from "./styles/collapseHeader.inline.scss"
//@ts-ignore
import curiusScript from "./scripts/curius.inline"
//@ts-ignore
import curiusFriendScript from "./scripts/curius-friends.inline"
//@ts-ignore
import curiusNavigationScript from "./scripts/curius-navigation.inline"
import { htmlToJsx } from "../util/jsx"
import Content from "./pages/Content"
import { BuildCtx } from "../util/ctx"
import { checkBib, checkBibSection } from "../plugins/transformers/citations"
import { checkFootnoteRef, checkFootnoteSection } from "../plugins/transformers/gfm"

interface RenderComponents {
  head: QuartzComponent
  header: QuartzComponent[]
  beforeBody: QuartzComponent[]
  pageBody: QuartzComponent
  afterBody: QuartzComponent[]
  sidebar: QuartzComponent[]
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

  const icons = s("svg", { ...svgOptions, class: "collapsed-dots" }, [
    s("use", { href: "#triple-dots" }),
  ])
  node.children.splice(lastIdx, 0, icons)

  let className = ["collapsible-header"]
  if (endHr) {
    className.push("end-hr")
  }

  node.children = [
    h(
      `span.toggle-button#${buttonId}-toggle`,
      {
        role: "button",
        ariaExpanded: true,
        ariaLabel: "Toggle content visibility",
        ariaControls: `${buttonId}-content`,
        type: "button",
      },
      [
        h(".toggle-icons", [
          s(
            "svg",
            {
              ...svgOptions,
              fill: "var(--dark)",
              stroke: "var(--dark)",
              class: "circle-icon",
            },
            [s("use", { href: "#circle-icon" })],
          ),
          s(
            "svg",
            {
              ...svgOptions,
              fill: "var(--iris)",
              stroke: "var(--iris)",
              class: "expand-icon",
            },
            [s("use", { href: "#arrow-down" })],
          ),
          s(
            "svg",
            {
              ...svgOptions,
              fill: "var(--foam)",
              stroke: "var(--foam)",
              class: "collapse-icon",
            },
            [s("use", { href: "#arrow-up" })],
          ),
        ]),
      ],
    ),
    ...node.children,
  ]

  const rank = headingRank(node) as number

  return h(`section.${className.join(".")}#${id}`, { "data-level": rank }, [
    node,
    h(
      ".collapsible-header-content-outer",
      {
        id: `${buttonId}-content`,
        arialabelledby: buttonId,
      },
      [
        h(
          ".collapsible-header-content",
          {
            ["data-references"]: `${buttonId}-toggle`,
            ["data-level"]: `${rank}`,
            ["data-heading-id"]: node.properties.id, // HACK: This assumes that rehype-slug already runs this target
          },
          content,
        ),
      ],
    ),
  ])
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
  visit(
    root,
    //@ts-ignore
    (node: Element) => checkBib(node as Element),
    (node: Element) => {
      node.properties.href = `${(node as Element).properties.href}${appendSuffix !== undefined ? "-" + appendSuffix : ""}`
    },
  )

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
      parent!.children.splice(index!, 1)
    }
  })

  // finally, update the final position
  visit(root, { tagName: "section" }, (node: Element, index, parent) => {
    if (node.properties.dataReferences == "") {
      // @ts-ignore
      node.children[1].children = finalRefs
      parent!.children.splice(index as number, 1, node)
    }
  })
}

const getFootnotesList = (node: Element) =>
  (node.children as Element[]).filter((val) => val.tagName === "ol")[0]

const getBibList = (node: Element) =>
  (node.children as Element[]).filter((val) => val.tagName === "ul")[0]

type FootnoteInfo = {
  originalHref: string
  index: number
  footnoteId: string
  referenceIds: string[]
}

function mergeFootnotes(root: Root, appendSuffix?: string | undefined): void {
  const notesByHref = new Map<string, FootnoteInfo>()
  const noteOrder: FootnoteInfo[] = []
  const finalRefs: Element[] = []
  const toRemove: Element[] = []
  const suffixFragment = appendSuffix ? `-${appendSuffix}` : ""

  visit(
    root,
    // @ts-ignore
    (node: Element) => {
      if (checkFootnoteRef(node)) {
        const originalHref = node.properties.href as string
        let info = notesByHref.get(originalHref)
        if (!info) {
          const index = notesByHref.size + 1
          info = {
            originalHref,
            index,
            footnoteId: `fn-${index}${suffixFragment}`,
            referenceIds: [],
          }
          notesByHref.set(originalHref, info)
          noteOrder.push(info)
        }

        const refId = `fnref-${info.index}-${info.referenceIds.length + 1}${suffixFragment}`
        info.referenceIds.push(refId)

        node.properties.href = `#${info.footnoteId}`
        node.properties.id = refId

        const current = info
        if (current) {
          visit(node, "text", (textNode: Text) => {
            textNode.value = `${current.index}`
          })
        }
      }
    },
  )

  visit(root, function (node) {
    if (checkFootnoteSection(node as Element)) {
      toRemove.push(node as Element)
      finalRefs.push(...(getFootnotesList(node as Element).children as Element[]))
    }
  })

  // we don't want to remove the last nodes
  toRemove.pop()
  if (noteOrder.length === 0) return

  // Remove all reference divs except the last one
  visit(root, { tagName: "section" }, (node: Element, index, parent) => {
    if (toRemove.includes(node)) {
      parent!.children.splice(index as number, 1)
    }
  })

  const sortedRefs: Element[] = []
  const seenOriginal = new Set<string>()

  for (const note of noteOrder) {
    const originalId = note.originalHref.replace("#", "")
    if (seenOriginal.has(originalId)) {
      continue
    }
    const refIdx = finalRefs.findIndex((ref) => ref.properties?.id === originalId)
    if (refIdx === -1) {
      continue
    }
    const ref = finalRefs[refIdx]
    seenOriginal.add(originalId)

    ref.properties = ref.properties ?? {}
    ref.properties.id = note.footnoteId

    const anchorsToRemove: { parent: Element; index: number }[] = []
    visit(ref, "element", (child: Element, index, parent) => {
      if (child.tagName === "a" && child.properties?.dataFootnoteBackref === "") {
        anchorsToRemove.push({ parent: parent as Element, index: index as number })
      }
    })

    anchorsToRemove.sort((a, b) => b.index - a.index)
    for (const { parent, index } of anchorsToRemove) {
      parent.children.splice(index, 1)
      const maybeText = parent.children[index - 1] as Text | undefined
      if (maybeText && maybeText.type === "text" && maybeText.value.trim() === "") {
        parent.children.splice(index - 1, 1)
      }
    }

    let container: Element = ref
    for (let i = ref.children.length - 1; i >= 0; i--) {
      const child = ref.children[i]
      if (child.type === "element") {
        container = child as Element
        break
      }
    }

    note.referenceIds.forEach((refId, ordinal) => {
      if (container.children.length > 0) {
        container.children.push({ type: "text", value: " " } as Text)
      }
      container.children.push(
        h(
          "a",
          {
            href: `#${refId}`,
            dataFootnoteBackref: "",
            ariaLabel: "Back to content",
          },
          `↩${ordinal === 0 ? "" : ordinal + 1}`,
        ) as Element,
      )
    })

    sortedRefs.push(ref)
  }

  // finally, update the final position
  visit(root, { tagName: "section" }, (node: Element) => {
    if (node.properties.dataFootnotes == "") {
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

export const pageResources = (
  baseDir: FullSlug | RelativeURL,
  staticResources: StaticResources,
  ctx: BuildCtx,
) =>
  ({
    css: [
      { content: joinSegments(baseDir, "index.css") },
      { content: collapseHeaderStyle, inline: true },
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
        script: `const fetchData = fetch("${joinSegments(baseDir, "static/contentIndex.json")}").then(data => data.json())`,
      },
      {
        loadTime: "beforeDOMReady",
        contentType: "inline",
        spaPreserve: true,
        script: `const semanticCfg = ${JSON.stringify(ctx.cfg?.configuration?.semanticSearch ?? {})}`,
      },
      {
        script: collapseHeaderScript,
        loadTime: "afterDOMReady",
        contentType: "inline",
      },
      ...staticResources.js,
      {
        src: joinSegments(baseDir, "postscript.js"),
        loadTime: "afterDOMReady",
        moduleType: "module",
        contentType: "external",
        crossOrigin: "anonymous",
      },
    ],
    additionalHead: staticResources.additionalHead,
  }) satisfies StaticResources

const defaultTranscludeOpts: TranscludeOptions = {
  dynalist: true,
  title: true,
  skipTranscludes: false,
}

interface TranscludeStats {
  words: number
  minutes: number
  files: Set<string>
}

export function transcludeFinal(
  root: Root,
  { cfg, allFiles, fileData }: QuartzComponentProps,
  userOpts?: Partial<TranscludeOptions>,
): Root {
  // NOTE: return early these cases, we probably don't want to transclude them anw
  if (fileData.frontmatter?.poem || fileData.frontmatter?.menu) return root

  // Track total reading stats including transclusions
  const stats: TranscludeStats = {
    words: fileData.readingTime?.words ?? 0,
    minutes: fileData.readingTime?.minutes ?? 0,
    files: new Set([fileData.filePath!]),
  }

  // hierarchy of transclusion: frontmatter > userOpts > defaultOpts
  const slug = fileData.slug as FullSlug
  let opts: TranscludeOptions
  if (userOpts) {
    opts = { ...defaultTranscludeOpts, ...userOpts }
  } else {
    opts = defaultTranscludeOpts
  }

  if (fileData.frontmatter?.transclude) {
    opts = { ...opts, ...fileData.frontmatter?.transclude }
  }

  const { dynalist, skipTranscludes } = opts
  const isLanding = slug === "index"

  const anchor = (
    href: string,
    url: string,
    description: string | null,
    title: boolean,
  ): Element => {
    if (!title) return {} as Element

    const [parent, ...children] = url.split("/")
    const truncated = children.length > 2 ? `${parent}/.../${children[children.length - 1]}` : url
    const metadata: Element[] = [
      h("li", { style: "font-style: italic; color: var(--gray);" }, [
        { type: "text", value: `url: ${truncated}` },
      ]),
    ]

    if (description) {
      metadata.push(
        h("li", [
          h("span", { style: "text-decoration: underline;" }, [
            { type: "text", value: `description` },
          ]),
          { type: "text", value: `: ${description}` },
        ]),
      )
    }

    return h(".transclude-ref", { "data-href": href }, [
      h("ul.metadata", metadata),
      h(
        "button.transclude-title-link",
        {
          type: "button",
          ariaLabel: "Go to original link",
        },
        s(
          "svg",
          {
            ...svgOptions,
            fill: "none",
            stroke: "currentColor",
            strokewidth: "2",
            class: "blockquote-link",
          },
          [s("use", { href: "#github-anchor" })],
        ),
      ),
    ])
  }

  // NOTE: process transcludes in componentData
  visit(root, { tagName: "blockquote" }, (node) => {
    const classNames = (node.properties?.className ?? []) as string[]
    const url = node.properties.dataUrl as string
    const alias = (
      node.properties?.dataEmbedAlias !== "undefined"
        ? node.properties?.dataEmbedAlias
        : node.properties?.dataBlock
    ) as string

    if (classNames.includes("transclude")) {
      if (skipTranscludes) {
        return
      }
      const [inner] = node.children as Element[]
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

      if (page?.readingTime && !stats.files.has(page.filePath!)) {
        stats.words += page.readingTime.words
        stats.minutes += page.readingTime.minutes
        stats.files.add(page.filePath!)
      }

      const { title } = transcludePageOpts

      let blockRef = node.properties.dataBlock as string | undefined
      if (blockRef?.startsWith("#^")) {
        // block transclude
        blockRef = blockRef.slice("#^".length)
        let blockNode = page.blocks?.[blockRef]
        if (blockNode) {
          if (blockNode.tagName === "li") blockNode = h("ul", blockNode)

          const children = [normalizeHastElement(blockNode, slug, transcludeTarget)]
          if (fileData.frontmatter?.pageLayout !== "reflection") {
            children.push(
              h("a", { href: inner.properties?.href, class: "internal transclude-src" }, [
                { type: "text", value: i18n(cfg.locale).components.transcludes.linkToOriginal },
              ]),
            )
          }
          node.children = children
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

        if (startIdx === undefined) return

        const children = [
          anchor(inner.properties?.href as string, url, alias, title),
          ...(page.htmlAst.children.slice(startIdx, endIdx) as ElementContent[]).map((child) =>
            normalizeHastElement(child as Element, slug, transcludeTarget),
          ),
        ]

        if (fileData.frontmatter?.pageLayout !== "reflection") {
          children.push(
            h("a", { href: inner.properties?.href, class: "internal transclude-src" }, [
              { type: "text", value: i18n(cfg.locale).components.transcludes.linkToOriginal },
            ]),
          )
        }
        node.children = children

        // support transcluding footnote and bib data
        let footnoteSection: Element | undefined = undefined
        let bibSection: Element | undefined = undefined
        visit(root, (node) => {
          if (checkFootnoteSection(node as Element)) {
            footnoteSection = node as Element
            return EXIT
          } else if (checkBibSection(node as Element)) {
            bibSection = node as Element
            return EXIT
          }
        })

        const transcludeFootnoteBlock: Element[] = []
        const transcludeBibBlock: Element[] = []

        visit(node, function (el) {
          const node = el as Element
          const { properties } = node
          if (checkFootnoteRef(node)) {
            visit(page.htmlAst!, { tagName: "section" }, (node) => {
              if (node.properties.dataFootnotes == "") {
                const noteId = (properties.href! as string).replace("#", "")
                transcludeFootnoteBlock.push(
                  getFootnotesList(node).children.find(
                    (ref) => (ref as Element).properties?.id === noteId,
                  ) as Element,
                )
              }
            })
          } else if (node.tagName === "cite" && node.children) {
            const linkId = (
              (node.children as Element[]).find((v) => v.tagName === "a")?.properties.href as string
            ).replace("#", "")
            visit(page.htmlAst!, { tagName: "section" }, (node) => {
              if (node.properties.dataReferences == "") {
                transcludeBibBlock.push(
                  getBibList(node).children.find(
                    (ref) => (ref as Element).properties?.id === linkId,
                  ) as Element,
                )
              }
            })
          }
        })

        if (transcludeFootnoteBlock.length !== 0) {
          if (!footnoteSection) {
            footnoteSection = h(
              "section.footnotes.main-col",
              { dataFootnotes: "", dataTransclude: "" },
              h(
                "h2.sr-only#footnote-label",
                { dir: "auto" },
                h("span.highlight-span", [{ type: "text", value: "Remarque" }]),
                h(
                  "a.internal#footnote-label",
                  { "data-role": "anchor", "data-no-popover": "true" },
                  s(
                    "svg",
                    { ...svgOptions, fill: "none", stroke: "currentColor", strokeWidth: "2" },
                    s("use", { href: "#github-anchor" }),
                  ),
                ),
              ),
              { type: "text", value: "\n" },
              h("ol", { dir: "auto" }, [...transcludeFootnoteBlock]),
              { type: "text", value: "\n" },
            )
            root.children.push(footnoteSection)
          } else {
            visit(footnoteSection, { tagName: "ol" }, (node: Element) => {
              node.children.push(...transcludeFootnoteBlock)
            })
          }
        }
        if (transcludeBibBlock.length !== 0) {
          if (!bibSection) {
            bibSection = h(
              "section.bibliography.main-col",
              { dataReferences: "", dataTransclude: "" },
              h(
                "h2#reference-label",
                { dir: "auto" },
                h("span.highlight-span", [{ type: "text", value: "Bibliographie" }]),
              ),
              { type: "text", value: "\n" },
              h("ul", { dir: "auto" }, [...transcludeBibBlock]),
              { type: "text", value: "\n" },
            )
            root.children.push(bibSection)
          } else {
            visit(bibSection, { tagName: "ul" }, (node: Element) => {
              node.children.push(...transcludeBibBlock)
            })
          }
        }
      } else if (page.htmlAst) {
        // page transclude
        const children = [
          anchor(inner.properties?.href as string, url, alias, title),
          title
            ? h("h1", [
                {
                  type: "text",
                  value:
                    page.frontmatter?.title ??
                    i18n(cfg.locale).components.transcludes.transcludeOf({
                      targetSlug: page.slug!,
                    }),
                },
              ])
            : ({} as ElementContent),
          ...(page.htmlAst.children as ElementContent[]).map((child) =>
            normalizeHastElement(child as Element, slug, transcludeTarget),
          ),
        ]

        if (fileData.frontmatter?.pageLayout !== "reflection") {
          children.push(
            h("a", { href: inner.properties?.href, class: "internal transclude-src" }, [
              { type: "text", value: i18n(cfg.locale).components.transcludes.linkToOriginal },
            ]),
          )
        }

        node.children = children
      }
    }
  })

  // NOTE: handling collapsible nodes
  if (dynalist && !slug.includes("posts")) {
    root.children = processHeaders(root.children as ElementContent[])
  }

  // NOTE: We then merge all references and footnotes to final items
  mergeIsomorphic(root)

  // NOTE: Update the file's reading time with transcluded content
  if (fileData.readingTime) {
    fileData.readingTime = { ...fileData.readingTime, words: stats.words, minutes: stats.minutes }
  }

  if (isLanding) {
    visit(root, { tagName: "a" }, (node: Element) => {
      node.properties["data-no-popover"] = true
    })
  }

  return root
}

export const TopLinks = {
  workshop: "/lectures",
  livres: "/books",
  merci: "/influence",
  advice: "/quotes",
  parfum: "/thoughts/Scents",
  "are.na": "/arena",
  craft: "/thoughts/craft",
  home: "/thoughts/furnitures",
  movies: "/movies",
  tunes: "/mixed",
}

type AliasLinkProp = {
  name?: string
  url?: string
  isInternal?: boolean
  newTab?: boolean | ((name: string) => boolean)
  enablePopover?: boolean
  classes?: string[]
  children?: JSX.Element | JSX.Element[]
}

const AliasLink = (props: AliasLinkProp) => {
  const opts = { isInternal: false, newTab: false, enablePopover: true, ...props }
  const className = ["landing-links"]
  if (opts.isInternal) className.push("internal")
  if (opts.classes) className.push(...opts.classes)
  return (
    <a
      href={opts.url}
      target={opts.newTab ? "_blank" : "_self"}
      rel="noopener noreferrer"
      className={className.join(" ")}
      data-no-popover={!opts.enablePopover}
    >
      {opts.name}
      {opts.children}
    </a>
  )
}

const NotesComponent = ((opts?: { slug: SimpleSlug; numLimits?: number; header?: string }) => {
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
      <section id={`note-item-${opts!.header}`} data-note>
        <h2>{opts!.header}.</h2>
        <div class="notes-container">
          <div class="recent-links">
            <ul class="landing-notes">
              {pages.slice(0, opts!.numLimits).map((page) => {
                const title = page.frontmatter?.title ?? i18n(cfg.locale).propertyDefaults.title
                return (
                  <li>
                    <a
                      data-no-popover
                      href={resolveRelative(fileData.slug!, page.slug!)}
                      class={classes}
                    >
                      <div class="landing-meta">
                        <span class="landing-mspan">
                          <DateComponent date={getDate(cfg, page)!} locale={cfg.locale} />
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
                  <a
                    data-no-popover
                    href={resolveRelative(fileData.slug!, opts!.slug)}
                    class={classes}
                  >
                    {i18n(cfg.locale).components.recentNotes.seeRemainingMore({
                      remaining,
                    })}
                  </a>
                </em>
              </p>
            )}
          </div>
        </div>
      </section>
    )
  }
  return Notes
}) satisfies QuartzComponentConstructor

const HyperlinksComponent = ((props?: { children: JSX.Element[] }) => {
  const { children } = props ?? { children: [] }

  const Hyperlink: QuartzComponent = () => <section class="hyperlinks">{children}</section>
  return Hyperlink
}) satisfies QuartzComponentConstructor

const ElementComponent = (() => {
  const Content = ContentConstructor()
  const RecentNotes = NotesComponent({
    header: "récentes",
    slug: "thoughts/" as SimpleSlug,
    numLimits: 9,
  })
  const RecentPosts = NotesComponent({
    header: "écriture",
    slug: "posts/" as SimpleSlug,
    numLimits: 6,
  })

  const Element: QuartzComponent = (componentData: QuartzComponentProps) => {
    const svgToJsx = (hast: Element): JSX.Element =>
      htmlToJsx(componentData.fileData.filePath!, hast)

    const rssIcon = svgToJsx(
      s(
        "svg",
        {
          version: "1.1",
          xmlns: "http://www.w3.org/2000/svg",
          viewbox: "0 0 8 8",
          width: 8,
          height: 8,
          role: "img",
          stroke: "none",
          "data-icon": "rss",
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
    const githubIcon = svgToJsx(githubSvg)
    const twitterIcon = svgToJsx(twitterSvg)
    const bskyIcon = svgToJsx(bskySvg)
    const substackIcon = svgToJsx(substackSvg)

    const Hyperlink = HyperlinksComponent({
      children: [
        <section>
          <h2>jardin:</h2>
          <address class="clickable-container">
            {Object.entries(TopLinks).map(([name, url]) => (
              <AliasLink isInternal enablePopover={false} key={name} name={name} url={url} />
            ))}
          </address>
        </section>,
        <section>
          <h2>média:</h2>
          <address class="clickable-container">
            <AliasLink newTab classes={["external"]} name="github" url="https://github.com/aarnphm">
              {githubIcon}
            </AliasLink>
            <AliasLink newTab classes={["external"]} name="twitter" url="https://x.com/aarnphm_">
              {twitterIcon}
            </AliasLink>
            <AliasLink
              newTab
              classes={["external"]}
              name="substack"
              url="https://livingalonealone.com"
            >
              {substackIcon}
            </AliasLink>
            <AliasLink
              newTab
              classes={["external"]}
              name="bluesky"
              url="https://bsky.app/profile/aarnphm.xyz"
            >
              {bskyIcon}
            </AliasLink>
            <AliasLink newTab name="rss" url="/feed.xml">
              {rssIcon}
            </AliasLink>
          </address>
        </section>,
        <section class="boring-legal">
          <address class="clickable-container">
            <AliasLink newTab classes={["external"]} name="notes" url="https://notes.aarnphm.xyz" />
            <AliasLink isInternal enablePopover={false} name="colophon" url="/colophon" />
            <AliasLink isInternal enablePopover={false} name="privacy" url="/privacy-policy" />
            <AliasLink
              isInternal
              enablePopover={false}
              name="term of service"
              url="/terms-of-service"
            />
          </address>
        </section>,
      ],
    })

    return (
      <div class="content-container">
        <Content {...componentData} />
        <section class="notes-outer">
          <RecentNotes {...componentData} />
          <RecentPosts {...componentData} />
        </section>
        <Hyperlink {...componentData} />
      </div>
    )
  }

  return Element
}) satisfies QuartzComponentConstructor

// Menu components

function Functions({ displayClass }: QuartzComponentProps) {
  return (
    <section class={classNames(displayClass, "menu", "side-col")} data-function={true}>
      <a href="../atelier-with-friends" class="internal alias" data-no-popover={true}>
        atelier with friends.
      </a>
    </section>
  )
}

// Curius components
export const CuriusContent: QuartzComponent = (props: QuartzComponentProps) => {
  const { cfg, displayClass } = props
  const searchPlaceholder = i18n(cfg.locale).components.search.searchBarPlaceholder

  return (
    <>
      <div class={classNames(displayClass, "curius", "curius-col")} id="curius">
        <div class="curius-page-container">
          <div class={classNames(displayClass, "curius-header")}>
            <div class="curius-search">
              <input
                id="curius-bar"
                type="text"
                aria-label={searchPlaceholder}
                placeholder={searchPlaceholder}
              />
              <div id="curius-search-container" />
            </div>
            <div class="curius-title">
              <em>
                Voir de plus{" "}
                <a href="https://curius.app/aaron-pham" target="_blank">
                  curius.app/aaron-pham
                </a>
              </em>
            </div>
          </div>
          <div id="curius-fetching-text" />
          <div id="curius-fragments" />
          <div class="highlight-modal" id="highlight-modal">
            <ul id="highlight-modal-list" />
          </div>
        </div>
      </div>
      <Content {...props} />
    </>
  )
}
CuriusContent.afterDOMLoaded = curiusScript

export const CuriusFriends: QuartzComponent = (props: QuartzComponentProps) => {
  const { displayClass } = props
  return (
    <div class={classNames(displayClass, "curius-friends")}>
      <h4 style={["font-size: initial", "margin-top: unset", "margin-bottom: 0.5rem"].join(";")}>
        mes amis.
      </h4>
      <ul class="overflow section-ul" id="friends-list" style="margin-top: unset"></ul>
      <div id="see-more-friends">
        Void{" "}
        <span id="more" style="text-decoration: none !important">
          de plus
        </span>
        <svg
          fill="currentColor"
          preserveAspectRatio="xMidYMid meet"
          height="1rem"
          width="1rem"
          viewBox="0 -10 40 40"
        >
          <g>
            <path d="m31 12.5l1.5 1.6-12.5 13.4-12.5-13.4 1.5-1.6 11 11.7z" />
          </g>
        </svg>
      </div>
    </div>
  )
}
CuriusFriends.afterDOMLoaded = curiusFriendScript

const CuriusTrail: QuartzComponent = (props: QuartzComponentProps) => {
  const { cfg, displayClass } = props
  return (
    <div class={classNames(displayClass, "curius-trail")} data-limits={3} data-locale={cfg.locale}>
      <h4 style={["font-size: initial", "margin-top: unset", "margin-bottom: 0.5rem"].join(";")}>
        sentiers.
      </h4>
      <ul class="section-ul" id="trail-list" />
    </div>
  )
}

export const CuriusNavigation: QuartzComponent = (props: QuartzComponentProps) => {
  const { displayClass } = props
  return (
    <div class={classNames(displayClass, "curius-pagination", "curius-col")} id="curius-pagination">
      <span id="curius-prev" style="visibility: hidden">
        (prev)
      </span>
      <span id="curius-next">next</span>
    </div>
  )
}
CuriusNavigation.afterDOMLoaded = curiusNavigationScript

export function renderPage(
  ctx: BuildCtx,
  slug: FullSlug,
  componentData: QuartzComponentProps,
  components: RenderComponents,
  pageResources: StaticResources,
  disableSidepanel?: boolean,
  isFolderTag?: boolean,
): string {
  // make a deep copy of the tree so we don't remove the transclusion references
  // for the file cached in contentMap in build.ts
  const root = clone(componentData.tree) as Root
  // NOTE: set componentData.tree to the edited html that has transclusions rendered

  const tree = transcludeFinal(root, componentData)

  // NOTE: Finally, we dump out the data-references and data-footnotes down to page footer, if exists
  let retrieval: Set<Element> = new Set<Element>()
  visit(tree, { tagName: "section" }, (node, index, parent) => {
    if (node.properties?.dataReferences === "" || node.properties?.dataFootnotes === "") {
      const className = Array.isArray(node.properties.className)
        ? node.properties.className
        : (node.properties.className = [])
      className.push("main-col")
      const els = new Set(parent?.children.splice(index!, 1)) as Set<Element>
      retrieval = retrieval.union(els)
    }
  })
  componentData.tree = tree
  disableSidepanel = disableSidepanel ?? true
  isFolderTag = isFolderTag ?? false

  if (slug === "index") {
    components = {
      ...components,
      header: [Image(), Graph(), Search(), Palette()],
      sidebar: [],
      afterBody: [],
      beforeBody: [],
      pageBody: (props: QuartzComponentProps) => {
        const { displayClass } = props
        const Element = ElementComponent()

        return (
          <>
            <h1 class="article-title" style="margin-top: 2rem" lang="fr">
              Bonjour, je suis Aaron.
            </h1>
            <div class={classNames(displayClass, "landing")}>
              <Element {...props} />
            </div>
          </>
        )
      },
    }
  } else if (slug === "curius") {
    components = {
      ...components,
      header: [],
      beforeBody: [],
      sidebar: [CuriusFriends, CuriusTrail],
      pageBody: CuriusContent,
      afterBody: [CuriusNavigation],
      footer: FooterConstructor({ layout: "curius" }),
    }
  }

  if (componentData.fileData.frontmatter?.poem) {
    components = {
      ...components,
      footer: FooterConstructor({ layout: "poetry" }),
    }
  }

  let isMenu = false
  if (componentData.fileData.frontmatter?.menu) {
    isMenu = true
    components = {
      ...components,
      header: [],
      beforeBody: [],
      sidebar: [],
      afterBody: [Functions],
      footer: FooterConstructor({ layout: "menu" }),
    }
  }

  if (componentData.fileData.frontmatter?.pageLayout === "letter-poem") {
    components = {
      ...components,
      header: [],
      sidebar: [],
      afterBody: [],
      beforeBody: [],
    }
  }

  const {
    head: Head,
    header,
    beforeBody,
    pageBody: Content,
    afterBody,
    sidebar,
    footer: Footer,
  } = components
  const Header = HeaderConstructor()

  // TODO: https://thesolarmonk.com/posts/a-spacebar-for-the-web style
  const lang =
    (componentData.fileData.frontmatter?.lang ?? componentData.cfg.locale)?.split("-")[0] ?? "en"
  const pageLayout = componentData.fileData.frontmatter?.pageLayout ?? "default"
  if (pageLayout === "technical") disableSidepanel = true
  const isSlides = componentData.fileData.frontmatter?.slides ?? false
  const isArena = slug === "arena" || slug.startsWith("arena/")
  const isCurius = slug === "curius"
  const isArenaSubpage = slug.startsWith("arena/") && slug !== "arena"
  if (isArena || isArenaSubpage) disableSidepanel = true

  return (
    `<!DOCTYPE html>
<!--
/*************************************************************************
* Bop got your nose !!!
*
* Hehe
*
* Anw if you see a component you like ping @aarnphm on Discord I can try
* to send it your way. Have a wonderful day!
**************************************************************************/
-->
` +
    render(
      <html lang={lang}>
        <Head {...componentData} />
        <body
          data-slug={slug}
          data-language={lang}
          data-menu={isMenu}
          data-slides={isSlides}
          data-layout={pageLayout}
          data-disable-sidepanel={disableSidepanel}
          data-is-folder-tag={isFolderTag}
          data-arena-subpage={isArenaSubpage}
        >
          <main
            id="quartz-root"
            class={classNames(undefined, "page", slug === "index" ? "grid" : "")}
            style={
              slug !== "index"
                ? { display: "flex", flexDirection: "column", minHeight: "100vh" }
                : undefined
            }
          >
            <Header {...componentData}>
              {header.map((HeaderComponent) => (
                <HeaderComponent {...componentData} />
              ))}
            </Header>
            <section id="stacked-notes-container" class="all-col">
              <div id="stacked-notes-main">
                <div class="stacked-notes-column" />
              </div>
            </section>
            {beforeBody.length > 0 && (
              <section
                class={classNames(
                  undefined,
                  "page-header",
                  "popover-hint",
                  isArena ? "all-col" : "all-col grid",
                )}
              >
                {beforeBody.map((BodyComponent) => (
                  <BodyComponent {...componentData} />
                ))}
              </section>
            )}
            <section
              class={classNames(
                undefined,
                "page-content",
                slug === "index" ? "side-col" : isArena ? "all-col" : "grid all-col",
              )}
              style={{ flex: "1 1 auto" }}
            >
              {sidebar.length > 0 && (
                <aside class="aside-container left-col">
                  {sidebar.map((BodyComponent) => (
                    <BodyComponent {...componentData} />
                  ))}
                </aside>
              )}
              <Content {...componentData} />
              {!isSlides && !isArena && !isCurius && (
                <>
                  <div id="wc-modal" class="wc-modal">
                    <div class="wc-inner" />
                  </div>
                  <aside class="sidepanel-container" />
                </>
              )}
            </section>
            {!isFolderTag && (
              <section class="page-footer popover-hint grid all-col">
                {retrieval.size > 0 &&
                  htmlToJsx(componentData.fileData.filePath!, {
                    type: "root",
                    children: [...retrieval],
                  } as Node)}
                {afterBody.length > 0 &&
                  afterBody.map((BodyComponent) => <BodyComponent {...componentData} />)}
                {slug !== "index" && <Footer {...componentData} />}
              </section>
            )}
            <QuartzIcon filePath={componentData.fileData.filePath!} />
          </main>
        </body>
        {pageResources.js
          .filter((resource) => resource.loadTime === "afterDOMReady")
          .map((res) => JSResourceToScriptElement(res))}
        {/* Cloudflare Web Analytics */}
        {!ctx.argv.serve && (
          <script
            defer
            src={"https://static.cloudflareinsights.com/beacon.min.js"}
            data-cf-beacon='{"token": "3b6a9ecda4294f8bb5770c2bfb44078c"}'
            crossOrigin={"anonymous"}
            spa-preserve={true}
          />
        )}
        {/* End Cloudflare Web Analytics */}
      </html>,
    )
  )
}
