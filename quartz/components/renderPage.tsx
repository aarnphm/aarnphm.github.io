import { render } from "preact-render-to-string"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import HeaderConstructor from "./Header"
import ContentConstructor from "./pages/Content"
import FooterConstructor from "./Footer"
import SearchConstructor from "./Search"
import GraphConstructor from "./Graph"
import { byDateAndAlphabetical } from "./PageList"
import { getDate, Date as DateComponent } from "./Date"
import { classNames } from "../util/lang"
import { JSResource, JSResourceToScriptElement, StaticResources } from "../util/resources"
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
import { i18n } from "../i18n"
import { JSX } from "preact"
import { headingRank } from "hast-util-heading-rank"
import type { TranscludeOptions } from "../plugins/transformers/frontmatter"
import { QuartzPluginData } from "../plugins/vfile"
// @ts-ignore
import mermaidScript from "./scripts/mermaid.inline"
// @ts-ignore
import mermaidImportScript from "./scripts/mermaid-import.inline"
import mermaidStyle from "./styles/mermaid.inline.scss"
import { h, s } from "hastscript"
// @ts-ignore
import collapseHeaderScript from "./scripts/collapse-header.inline.ts"
import collapseHeaderStyle from "./styles/collapseHeader.inline.scss"
//@ts-ignore
import curiusScript from "./scripts/curius.inline"
//@ts-ignore
import curiusFriendScript from "./scripts/curius-friends.inline"
import { htmlToJsx } from "../util/jsx"
import Content from "./pages/Content"
import { BuildCtx } from "../util/ctx"
import { checkBib } from "../plugins/transformers/citations"

interface RenderComponents {
  head: QuartzComponent
  header: QuartzComponent[]
  beforeBody: QuartzComponent[]
  pageBody: QuartzComponent
  afterBody: QuartzComponent[]
  sidebar: QuartzComponent[]
  footer: QuartzComponent
}

export const svgOptions = {
  xmlns: "http://www.w3.org/2000/svg",
  width: 16,
  height: 16,
  viewbox: "0 0 24 24",
  fill: "currentColor",
  stroke: "none",
  strokewidth: 0,
  strokelinecap: "round",
  strokelinejoin: "round",
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

const checkFootnoteRef = ({ type, tagName, properties }: Element) =>
  type === "element" && tagName === "a" && Boolean(properties) && properties.dataFootnoteRef === ""

const checkFootnotes = ({ type, tagName, properties }: Element) =>
  type === "element" && tagName === "section" && properties.dataFootnotes == ""

const getFootnotesList = (node: Element) =>
  (node.children as Element[]).filter((val) => val.tagName === "ol")[0]

function mergeFootnotes(root: Root, appendSuffix?: string | undefined): void {
  const orderNotes: Note[] = []
  const finalRefs: Element[] = []
  const toRemove: Element[] = []

  let idx = 0
  visit(
    root,
    // @ts-ignore
    (node: Element) => {
      if (checkFootnoteRef(node)) {
        orderNotes.push({ href: node.properties.href as string, id: node.properties.id as string })
        node.properties.href = `${node.properties.href}${appendSuffix !== undefined ? "-" + appendSuffix : ""}`
        node.properties.id =
          node.properties.id + `${appendSuffix !== undefined ? "-" + appendSuffix : ""}`
        visit(node, "text", (node) => {
          node.value = `${idx + 1}`
          idx++
        })
      }
    },
    false,
  )

  visit(
    root,
    function (node) {
      if (checkFootnotes(node as Element)) {
        toRemove.push(node as Element)
        finalRefs.push(...(getFootnotesList(node as Element).children as Element[]))
      }
    },
    false,
  )

  // we don't want to remove the last nodes
  toRemove.pop()
  if (orderNotes.length === 0) return

  // Remove all reference divs except the last one
  visit(root, { tagName: "section" }, (node: Element, index, parent) => {
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
    visit(ref, { tagName: "a" }, (c) => {
      if (c.properties.dataFootnoteBackref == "") {
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
  fileData: QuartzPluginData,
  staticResources: StaticResources,
): StaticResources {
  const contentIndexPath = joinSegments(baseDir, "static/contentIndex.json")
  const contentIndexScript = `const fetchData = fetch("${contentIndexPath}").then(data => data.json())`

  const resources: StaticResources = {
    css: [
      { content: joinSegments(baseDir, "index.css") },
      {
        content: collapseHeaderStyle,
        inline: true,
      },
      ...staticResources.css,
    ],
    js: [
      fileData.hasMermaidDiagram
        ? {
            script: mermaidImportScript,
            loadTime: "beforeDOMReady",
            moduleType: "module",
            contentType: "inline",
          }
        : ({} as JSResource),
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
        loadTime: "afterDOMReady",
        contentType: "inline",
      },
      ...staticResources.js,
    ],
    metadata: { hasMermaidDiagram: fileData.hasMermaidDiagram },
  }

  if (fileData.hasMermaidDiagram) {
    resources.js.push({
      script: mermaidScript,
      loadTime: "afterDOMReady",
      moduleType: "module",
      contentType: "inline",
    })
    resources.css.push({ content: mermaidStyle, inline: true })
  }

  // NOTE: we have to put this last to make sure spa.inline.ts is the last item.
  resources.js.push({
    src: joinSegments(baseDir, "postscript.js"),
    loadTime: "afterDOMReady",
    moduleType: "module",
    contentType: "external",
  })

  return resources
}

const defaultTranscludeOptions: TranscludeOptions = { dynalist: true, title: true }

interface TranscludeStats {
  words: number
  minutes: number
  files: Set<string>
}

export function transcludeFinal(
  ctx: BuildCtx,
  root: Root,
  { cfg, allFiles, fileData, externalResources }: QuartzComponentProps,
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
    opts = { ...defaultTranscludeOptions, ...userOpts }
  } else {
    opts = defaultTranscludeOptions
  }

  if (fileData.frontmatter?.transclude) {
    opts = { ...opts, ...fileData.frontmatter?.transclude }
  }

  const { dynalist } = opts

  const anchor = (href: string, url: string, description: string, title: boolean): Element => {
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

          const children = [
            anchor(inner.properties?.href as string, url, alias, title),
            normalizeHastElement(blockNode, slug, transcludeTarget),
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

        // support transcluding footnote data
        let hasFootnotes = false
        let footnoteSection: Element | undefined
        visit(
          root,
          (node) => {
            if (checkFootnotes(node as Element)) {
              hasFootnotes = true
              footnoteSection = node as Element
            }
          },
          true,
        )

        let transcludeFootnoteBlock: Element[] = []
        visit(
          node,
          function (node: Element) {
            const { properties } = node
            if (checkFootnoteRef(node as Element)) {
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
            }
          },
          false,
        )

        if (transcludeFootnoteBlock.length !== 0) {
          if (!footnoteSection) {
            footnoteSection = h(
              "section.footnotes",
              { dataFootnotes: "" },
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
            visit(footnoteSection, { tagName: "ol" }, (node) => {
              node.children.push(...transcludeFootnoteBlock)
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

      if (page.hasMermaidDiagram && !externalResources.metadata.hasMermaidDiagram) {
        externalResources.js.push({
          script: mermaidScript,
          loadTime: "afterDOMReady",
          moduleType: "module",
          contentType: "inline",
        })
        externalResources.css.push({ content: mermaidStyle, inline: true })
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

  return root
}

export const TopLinks = {
  livres: "/books",
  merci: "/influence",
  colophon: "/colophon",
  advice: "/quotes",
  parfum: "/thoughts/Scents",
  "are.na": "/are.na",
  tweets: "/tweets",
  curius: "/curius",
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
                    <a href={resolveRelative(fileData.slug!, page.slug!)} class={classes}>
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
                  <a href={resolveRelative(fileData.slug!, opts!.slug)} class={classes}>
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

export const githubSvg = s(
  "svg",
  {
    viewBox: "64 64 896 896",
    focusable: "false",
    version: "1.1",
    xmlns: "http://www.w3.org/2000/svg",
    "data-icon": "github",
    width: "1em",
    height: "1em",
    fill: "var(--gray)",
    role: "img",
    ariaLabel: "true",
  },
  s("path", {
    d: "M511.6 76.3C264.3 76.2 64 276.4 64 523.5 64 718.9 189.3 885 363.8 946c23.5 5.9 19.9-10.8 19.9-22.2v-77.5c-135.7 15.9-141.2-73.9-150.3-88.9C215 726 171.5 718 184.5 703c30.9-15.9 62.4 4 98.9 57.9 26.4 39.1 77.9 32.5 104 26 5.7-23.5 17.9-44.5 34.7-60.8-140.6-25.2-199.2-111-199.2-213 0-49.5 16.3-95 48.3-131.7-20.4-60.5 1.9-112.3 4.9-120 58.1-5.2 118.5 41.6 123.2 45.3 33-8.9 70.7-13.6 112.9-13.6 42.4 0 80.2 4.9 113.5 13.9 11.3-8.6 67.3-48.8 121.3-43.9 2.9 7.7 24.7 58.3 5.5 118 32.4 36.8 48.9 82.7 48.9 132.3 0 102.2-59 188.1-200 212.9a127.5 127.5 0 0138.1 91v112.5c.8 9 0 17.9 15 17.9 177.1-59.7 304.6-227 304.6-424.1 0-247.2-200.4-447.3-447.5-447.3z",
  }),
)

export const twitterSvg = s(
  "svg",
  {
    viewBox: "64 64 896 896",
    focusable: "false",
    version: "1.1",
    xmlns: "http://www.w3.org/2000/svg",
    "data-icon": "twitter",
    width: "1em",
    height: "1em",
    fill: "var(--gray)",
    role: "img",
    ariaLabel: "true",
  },
  s("path", {
    d: "M928 254.3c-30.6 13.2-63.9 22.7-98.2 26.4a170.1 170.1 0 0075-94 336.64 336.64 0 01-108.2 41.2A170.1 170.1 0 00672 174c-94.5 0-170.5 76.6-170.5 170.6 0 13.2 1.6 26.4 4.2 39.1-141.5-7.4-267.7-75-351.6-178.5a169.32 169.32 0 00-23.2 86.1c0 59.2 30.1 111.4 76 142.1a172 172 0 01-77.1-21.7v2.1c0 82.9 58.6 151.6 136.7 167.4a180.6 180.6 0 01-44.9 5.8c-11.1 0-21.6-1.1-32.2-2.6C211 652 273.9 701.1 348.8 702.7c-58.6 45.9-132 72.9-211.7 72.9-14.3 0-27.5-.5-41.2-2.1C171.5 822 261.2 850 357.8 850 671.4 850 843 590.2 843 364.7c0-7.4 0-14.8-.5-22.2 33.2-24.3 62.3-54.4 85.5-88.2z",
  }),
)
export const substackSvg = s(
  "svg",
  {
    width: 21,
    height: 24,
    viewBox: "0 0 21 24",
    fill: "#FF6719",
    role: "img",
    "data-icon": "substack",
    strokeWidth: "1.8",
    stroke: "none",
    xmlns: "http://www.w3.org/2000/svg",
    version: "1.1",
  },
  s(
    "g",
    s("path", { d: "M20.9991 5.40625H0V8.24275H20.9991V5.40625Z" }),
    s("path", { d: "M0 10.8125V24.0004L10.4991 18.1107L21 24.0004V10.8125H0Z" }),
    s("path", { d: "M20.9991 0H0V2.83603H20.9991V0Z" }),
  ),
)

export const hfSvg = s(
  "svg",
  {
    xmlns: "http://www.w3.org/2000/svg",
    fill: "none",
    width: "1em",
    height: "1em",
    role: "img",
    "data-icon": "huggingface",
    viewBox: "0 0 95 88",
  },
  s("path", {
    fill: "#FFD21E",
    d: "M47.21 76.5a34.75 34.75 0 1 0 0-69.5 34.75 34.75 0 0 0 0 69.5Z",
  }),
  s("path", {
    fill: "#FF9D0B",
    d: "M81.96 41.75a34.75 34.75 0 1 0-69.5 0 34.75 34.75 0 0 0 69.5 0Zm-73.5 0a38.75 38.75 0 1 1 77.5 0 38.75 38.75 0 0 1-77.5 0Z",
  }),
  s("path", {
    fill: "#3A3B45",
    d: "M58.5 32.3c1.28.44 1.78 3.06 3.07 2.38a5 5 0 1 0-6.76-2.07c.61 1.15 2.55-.72 3.7-.32ZM34.95 32.3c-1.28.44-1.79 3.06-3.07 2.38a5 5 0 1 1 6.76-2.07c-.61 1.15-2.56-.72-3.7-.32Z",
  }),
  s("path", {
    fill: "#FF323D",
    d: "M46.96 56.29c9.83 0 13-8.76 13-13.26 0-2.34-1.57-1.6-4.09-.36-2.33 1.15-5.46 2.74-8.9 2.74-7.19 0-13-6.88-13-2.38s3.16 13.26 13 13.26Z",
  }),
  s("path", {
    fill: "#3A3B45",
    fillRule: "evenodd",
    clipRule: "evenodd",
    d: "M39.43 54a8.7 8.7 0 0 1 5.3-4.49c.4-.12.81.57 1.24 1.28.4.68.82 1.37 1.24 1.37.45 0 .9-.68 1.33-1.35.45-.7.89-1.38 1.32-1.25a8.61 8.61 0 0 1 5 4.17c3.73-2.94 5.1-7.74 5.1-10.7 0-2.34-1.57-1.6-4.09-.36l-.14.07c-2.31 1.15-5.39 2.67-8.77 2.67s-6.45-1.52-8.77-2.67c-2.6-1.29-4.23-2.1-4.23.29 0 3.05 1.46 8.06 5.47 10.97Z",
  }),
  s("path", {
    fill: "#FF9D0B",
    d: "M70.71 37a3.25 3.25 0 1 0 0-6.5 3.25 3.25 0 0 0 0 6.5ZM24.21 37a3.25 3.25 0 1 0 0-6.5 3.25 3.25 0 0 0 0 6.5ZM17.52 48c-1.62 0-3.06.66-4.07 1.87a5.97 5.97 0 0 0-1.33 3.76 7.1 7.1 0 0 0-1.94-.3c-1.55 0-2.95.59-3.94 1.66a5.8 5.8 0 0 0-.8 7 5.3 5.3 0 0 0-1.79 2.82c-.24.9-.48 2.8.8 4.74a5.22 5.22 0 0 0-.37 5.02c1.02 2.32 3.57 4.14 8.52 6.1 3.07 1.22 5.89 2 5.91 2.01a44.33 44.33 0 0 0 10.93 1.6c5.86 0 10.05-1.8 12.46-5.34 3.88-5.69 3.33-10.9-1.7-15.92-2.77-2.78-4.62-6.87-5-7.77-.78-2.66-2.84-5.62-6.25-5.62a5.7 5.7 0 0 0-4.6 2.46c-1-1.26-1.98-2.25-2.86-2.82A7.4 7.4 0 0 0 17.52 48Zm0 4c.51 0 1.14.22 1.82.65 2.14 1.36 6.25 8.43 7.76 11.18.5.92 1.37 1.31 2.14 1.31 1.55 0 2.75-1.53.15-3.48-3.92-2.93-2.55-7.72-.68-8.01.08-.02.17-.02.24-.02 1.7 0 2.45 2.93 2.45 2.93s2.2 5.52 5.98 9.3c3.77 3.77 3.97 6.8 1.22 10.83-1.88 2.75-5.47 3.58-9.16 3.58-3.81 0-7.73-.9-9.92-1.46-.11-.03-13.45-3.8-11.76-7 .28-.54.75-.76 1.34-.76 2.38 0 6.7 3.54 8.57 3.54.41 0 .7-.17.83-.6.79-2.85-12.06-4.05-10.98-8.17.2-.73.71-1.02 1.44-1.02 3.14 0 10.2 5.53 11.68 5.53.11 0 .2-.03.24-.1.74-1.2.33-2.04-4.9-5.2-5.21-3.16-8.88-5.06-6.8-7.33.24-.26.58-.38 1-.38 3.17 0 10.66 6.82 10.66 6.82s2.02 2.1 3.25 2.1c.28 0 .52-.1.68-.38.86-1.46-8.06-8.22-8.56-11.01-.34-1.9.24-2.85 1.31-2.85Z",
  }),
  s("path", {
    fill: "#FFD21E",
    d: "M38.6 76.69c2.75-4.04 2.55-7.07-1.22-10.84-3.78-3.77-5.98-9.3-5.98-9.3s-.82-3.2-2.69-2.9c-1.87.3-3.24 5.08.68 8.01 3.91 2.93-.78 4.92-2.29 2.17-1.5-2.75-5.62-9.82-7.76-11.18-2.13-1.35-3.63-.6-3.13 2.2.5 2.79 9.43 9.55 8.56 11-.87 1.47-3.93-1.71-3.93-1.71s-9.57-8.71-11.66-6.44c-2.08 2.27 1.59 4.17 6.8 7.33 5.23 3.16 5.64 4 4.9 5.2-.75 1.2-12.28-8.53-13.36-4.4-1.08 4.11 11.77 5.3 10.98 8.15-.8 2.85-9.06-5.38-10.74-2.18-1.7 3.21 11.65 6.98 11.76 7.01 4.3 1.12 15.25 3.49 19.08-2.12Z",
  }),
  s("path", {
    fill: "#FF9D0B",
    d: "M77.4 48c1.62 0 3.07.66 4.07 1.87a5.97 5.97 0 0 1 1.33 3.76 7.1 7.1 0 0 1 1.95-.3c1.55 0 2.95.59 3.94 1.66a5.8 5.8 0 0 1 .8 7 5.3 5.3 0 0 1 1.78 2.82c.24.9.48 2.8-.8 4.74a5.22 5.22 0 0 1 .37 5.02c-1.02 2.32-3.57 4.14-8.51 6.1-3.08 1.22-5.9 2-5.92 2.01a44.33 44.33 0 0 1-10.93 1.6c-5.86 0-10.05-1.8-12.46-5.34-3.88-5.69-3.33-10.9 1.7-15.92 2.78-2.78 4.63-6.87 5.01-7.77.78-2.66 2.83-5.62 6.24-5.62a5.7 5.7 0 0 1 4.6 2.46c1-1.26 1.98-2.25 2.87-2.82A7.4 7.4 0 0 1 77.4 48Zm0 4c-.51 0-1.13.22-1.82.65-2.13 1.36-6.25 8.43-7.76 11.18a2.43 2.43 0 0 1-2.14 1.31c-1.54 0-2.75-1.53-.14-3.48 3.91-2.93 2.54-7.72.67-8.01a1.54 1.54 0 0 0-.24-.02c-1.7 0-2.45 2.93-2.45 2.93s-2.2 5.52-5.97 9.3c-3.78 3.77-3.98 6.8-1.22 10.83 1.87 2.75 5.47 3.58 9.15 3.58 3.82 0 7.73-.9 9.93-1.46.1-.03 13.45-3.8 11.76-7-.29-.54-.75-.76-1.34-.76-2.38 0-6.71 3.54-8.57 3.54-.42 0-.71-.17-.83-.6-.8-2.85 12.05-4.05 10.97-8.17-.19-.73-.7-1.02-1.44-1.02-3.14 0-10.2 5.53-11.68 5.53-.1 0-.19-.03-.23-.1-.74-1.2-.34-2.04 4.88-5.2 5.23-3.16 8.9-5.06 6.8-7.33-.23-.26-.57-.38-.98-.38-3.18 0-10.67 6.82-10.67 6.82s-2.02 2.1-3.24 2.1a.74.74 0 0 1-.68-.38c-.87-1.46 8.05-8.22 8.55-11.01.34-1.9-.24-2.85-1.31-2.85Z",
  }),
  s("path", {
    fill: "#FFD21E",
    d: "M56.33 76.69c-2.75-4.04-2.56-7.07 1.22-10.84 3.77-3.77 5.97-9.3 5.97-9.3s.82-3.2 2.7-2.9c1.86.3 3.23 5.08-.68 8.01-3.92 2.93.78 4.92 2.28 2.17 1.51-2.75 5.63-9.82 7.76-11.18 2.13-1.35 3.64-.6 3.13 2.2-.5 2.79-9.42 9.55-8.55 11 .86 1.47 3.92-1.71 3.92-1.71s9.58-8.71 11.66-6.44c2.08 2.27-1.58 4.17-6.8 7.33-5.23 3.16-5.63 4-4.9 5.2.75 1.2 12.28-8.53 13.36-4.4 1.08 4.11-11.76 5.3-10.97 8.15.8 2.85 9.05-5.38 10.74-2.18 1.69 3.21-11.65 6.98-11.76 7.01-4.31 1.12-15.26 3.49-19.08-2.12Z",
  }),
)

export const openaiSvg = s(
  "svg",
  {
    xmlns: "http://www.w3.org/2000/svg",
    viewBox: "0 0 150 150",
    fill: "#000000",
    stroke: "none",
    "data-icon": "openai",
    width: "1em",
    height: "1em",
    role: "img",
    "aria-label": "true",
  },
  s("path", {
    d: "M132.17,62.55c3.02-9.09,1.97-19.04-2.87-27.3-7.28-12.67-21.91-19.19-36.19-16.12-12.45-13.85-33.77-14.98-47.62-2.53-4.4,3.95-7.67,8.99-9.51,14.61-9.38,1.92-17.48,7.8-22.23,16.12-7.36,12.65-5.69,28.61,4.13,39.46-3.03,9.09-1.99,19.04,2.84,27.3,7.29,12.67,21.93,19.19,36.22,16.12,6.36,7.16,15.49,11.23,25.07,11.18,14.64,.01,27.62-9.44,32.09-23.38,9.38-1.93,17.48-7.8,22.23-16.12,7.27-12.63,5.59-28.5-4.16-39.33Zm-50.16,70.1c-5.85,0-11.51-2.04-15.99-5.79l.79-.45,26.57-15.34c1.35-.79,2.17-2.23,2.18-3.79v-37.46l11.23,6.5c.11,.06,.19,.16,.21,.29v31.04c-.03,13.79-11.2,24.96-24.99,24.99Zm-53.71-22.94c-2.93-5.06-3.98-10.99-2.97-16.76l.79,.47,26.59,15.34c1.34,.79,3,.79,4.34,0l32.49-18.73v12.97c0,.14-.07,.26-.18,.34l-26.91,15.52c-11.96,6.89-27.24,2.79-34.14-9.15Zm-7-57.87c2.95-5.09,7.61-8.98,13.15-10.97v31.57c-.02,1.55,.81,2.99,2.16,3.76l32.33,18.65-11.23,6.5c-.12,.07-.27,.07-.39,0l-26.86-15.49c-11.93-6.92-16.03-22.18-9.15-34.14v.13Zm92.28,21.44l-32.43-18.83,11.21-6.47c.12-.07,.27-.07,.39,0l26.86,15.52c11.95,6.9,16.05,22.18,9.15,34.13-2.9,5.03-7.47,8.9-12.92,10.93v-31.57c-.05-1.55-.91-2.96-2.26-3.71Zm11.18-16.81l-.79-.47-26.54-15.47c-1.35-.79-3.02-.79-4.37,0l-32.46,18.73v-12.97c-.01-.13,.05-.27,.16-.34l26.86-15.49c11.97-6.9,27.27-2.78,34.16,9.19,2.91,5.06,3.97,10.97,2.98,16.72v.11Zm-70.29,22.99l-11.23-6.47c-.11-.07-.19-.18-.21-.32v-30.96c.02-13.82,11.23-25,25.05-24.98,5.83,0,11.48,2.05,15.96,5.78l-.79,.45-26.57,15.34c-1.35,.79-2.17,2.23-2.18,3.79l-.03,37.38Zm6.1-13.15l14.47-8.34,14.49,8.34v16.68l-14.44,8.34-14.49-8.34-.03-16.68Z",
  }),
)

export const bentomlSvg = s(
  "svg",
  {
    xmlns: "http://www.w3.org/2000/svg",
    viewBox: "0 0 59.76 60",
    "data-icon": "bentoml",
    width: "1em",
    height: "1em",
    fill: "var(--gray)",
    role: "img",
  },
  s("g", [
    s("g", [
      s("path", {
        d: "m59.56,3.16c-.11-.35-.26-.69-.45-.99-.23-.37-.5-.71-.82-1-.05-.05-.11-.1-.16-.14-.39-.32-.83-.58-1.31-.76s-1-.27-1.55-.27H14.94c-.09,0-.18,0-.27,0-.03,0-.05,0-.08,0-.07,0-.13.01-.2.02-.03,0-.06,0-.08.01-.08.01-.15.02-.23.04-.01,0-.02,0-.03,0-.09.02-.17.04-.26.06-.02,0-.05.01-.07.02-.06.02-.12.04-.19.06-.03,0-.05.02-.08.03-.07.02-.13.05-.2.08-.02,0-.03.01-.05.02-.08.03-.16.07-.24.11-.02,0-.04.02-.06.03-.06.03-.12.06-.18.09-.02.01-.05.03-.07.04-.06.03-.12.07-.17.11-.02.01-.03.02-.05.03-.07.05-.14.1-.21.15-.01.01-.03.02-.04.03-.06.04-.11.09-.16.13-.02.02-.04.03-.06.05-.05.04-.1.09-.15.13-.02.01-.03.03-.05.04-.03.03-.05.05-.08.08,0,0,0,0,0,0L1.12,10.91c-.72.72-1.12,1.69-1.12,2.7v42.57c0,2.11,1.71,3.82,3.82,3.82h42.57c1.01,0,1.99-.4,2.7-1.12l9.52-11.05c.17-.19.33-.4.47-.62,0,0,.01-.02.02-.02.03-.04.05-.09.08-.13,0-.02.02-.03.03-.05.02-.04.04-.08.06-.12.01-.02.02-.04.03-.07.02-.04.04-.08.05-.12.01-.03.02-.05.03-.08.02-.04.03-.07.05-.11.01-.03.02-.06.03-.09.01-.04.03-.07.04-.11.01-.03.02-.06.03-.09.01-.04.02-.07.03-.11,0-.03.02-.07.03-.1,0-.03.02-.07.03-.1,0-.04.02-.07.02-.11,0-.03.02-.07.02-.1,0-.04.01-.08.02-.12,0-.03.01-.06.02-.1,0-.04.01-.08.02-.13,0-.03,0-.06.01-.09,0-.05,0-.1.01-.14,0-.03,0-.05,0-.08,0-.07,0-.15,0-.22V4.5h0c0-.47-.07-.91-.2-1.34Zm-4.3-1.81c.54,0,1.05.14,1.5.38.22.12.43.27.62.44.04.03.07.07.11.1.07.07.14.15.2.22.06.08.12.16.18.24.34.5.54,1.11.54,1.76v40.33c0,.09,0,.18-.01.27,0,.01,0,.02,0,.03,0,.08-.02.17-.03.25,0,.01,0,.02,0,.04-.02.08-.03.16-.06.24,0,0,0,.02,0,.03-.02.08-.05.16-.08.24,0,0,0,0,0,0-.1.26-.23.5-.38.71h0c-.05.08-.11.15-.17.22-.04.04-.07.08-.11.12,0,0,0,0,0,0-.04.04-.07.08-.11.11,0,0,0,0,0,0-.04.04-.08.07-.12.11,0,0,0,0,0,0-.21.18-.44.33-.7.45,0,0-.01,0-.02,0-.04.02-.09.04-.13.06-.01,0-.02,0-.04.01-.04.02-.08.03-.12.04-.02,0-.03.01-.05.02-.04.01-.07.02-.11.03-.02,0-.04.01-.05.02-.03,0-.07.02-.1.03-.02,0-.04,0-.06.01-.03,0-.07.01-.1.02-.02,0-.04,0-.06.01-.04,0-.07.01-.11.02-.02,0-.04,0-.06,0-.04,0-.08,0-.12.01-.02,0-.03,0-.05,0-.06,0-.11,0-.17,0H14.94c-.87,0-1.66-.35-2.23-.92-.14-.14-.27-.3-.38-.47-.28-.42-.47-.91-.52-1.44-.01-.11-.02-.21-.02-.32V4.5c0-.06,0-.12,0-.18,0-.02,0-.04,0-.06,0-.04,0-.08.01-.12,0-.02,0-.04,0-.06,0-.04.01-.07.02-.11,0-.02,0-.04.01-.06,0-.04.01-.07.02-.11,0-.02,0-.04.01-.06.01-.04.02-.08.03-.12,0-.01,0-.03.01-.04.03-.11.07-.22.12-.32,0,0,0,0,0,0,.02-.05.04-.1.07-.15,0,0,0-.02.01-.02.02-.04.04-.09.07-.13,0,0,0-.02.01-.03.02-.04.05-.08.08-.13,0,0,0-.01.01-.02.03-.05.06-.09.09-.14,0,0,0,0,0,0,.17-.24.38-.45.61-.63.2-.16.42-.29.65-.39,0,0,0,0,.01,0,.07-.03.15-.06.23-.09,0,0,.01,0,.02,0,.15-.05.31-.09.47-.12.01,0,.03,0,.04,0,.08-.01.15-.02.23-.03.01,0,.03,0,.04,0,.08,0,.16-.01.25-.01h40.33Z",
      }),
      s("g", [
        s("circle", { cx: "40.54", cy: "24.66", r: "2.74" }),
        s("circle", { cx: "29.67", cy: "24.66", r: "2.74" }),
      ]),
    ]),
  ]),
)

export const doiSvg = s(
  "svg",
  {
    xmlns: "http://www.w3.org/2000/svg",
    viewBox: "0 0 130 130",
    width: "1em",
    height: "1em",
    role: "img",
    style: "margin-left: 0.2em;",
  },
  s("circle", { fill: "#fcb425", cx: 65, cy: 65, r: 64 }),
  s("path", {
    d: "m 49.819127,84.559148 -11.854304,0 0,-4.825665 c -1.203594,1.510894 -4.035515,3.051053 -5.264716,3.742483 -2.151101,1.203585 -5.072066,1.987225 -7.812161,1.987225 -4.430246,0 -8.373925,-1.399539 -11.831057,-4.446924 -4.1229464,-3.636389 -6.0602455,-9.19576 -6.0602455,-15.188113 0,-6.094791 2.1126913,-10.960381 6.3380645,-14.59676 3.354695,-2.893745 7.457089,-5.209795 11.810505,-5.209795 2.535231,0 5.661807,0.227363 7.889738,1.302913 1.280414,0.614601 3.572628,2.060721 4.929872,3.469179 l 0,-25.420177 11.854304,0 z m -12.1199,-18.692584 c 0,-2.253538 -0.618258,-4.951555 -2.205973,-6.513663 -1.587724,-1.587724 -4.474153,-2.996182 -6.727691,-2.996182 -2.509615,0 -4.834476,1.825511 -6.447807,3.720535 -1.306031,1.536501 -1.959041,3.905269 -1.959041,5.877114 0,1.971835 0.740815,4.165004 2.046836,5.701505 1.587714,1.895025 3.297985,3.193739 5.833216,3.193739 2.279145,0 4.989965,-0.956662 6.552083,-2.51877 1.587714,-1.562108 2.908377,-4.185134 2.908377,-6.464278 z",
    fill: "#231f20",
  }),
  s("path", {
    d: "m 105.42764,25.617918 c -1.97184,0 -3.64919,0.69142 -5.03204,2.074271 -1.357247,1.357245 -2.035864,3.021779 -2.035864,4.993633 0,1.971835 0.678617,3.649193 2.035864,5.032034 1.38285,1.382861 3.0602,2.074281 5.03204,2.074281 1.99744,0 3.67479,-0.678627 5.03203,-2.035861 1.38285,-1.382861 2.07428,-3.073012 2.07428,-5.070454 0,-1.971854 -0.69143,-3.636388 -2.07428,-4.993633 -1.38285,-1.382851 -3.0602,-2.074271 -5.03203,-2.074271 z M 74.219383,45.507921 c -7.323992,0 -12.970625,2.283009 -16.939921,6.848949 -3.277876,3.782438 -4.916803,8.118252 -4.916803,13.008406 0,5.430481 1.626124,10.009834 4.878383,13.738236 3.943689,4.538918 9.475093,6.808622 16.59421,6.808622 7.093512,0 12.612122,-2.269704 16.555801,-6.808622 3.252259,-3.728402 4.878393,-8.1993 4.878393,-13.413648 0,-5.160323 -1.638938,-9.604602 -4.916803,-13.332994 -4.020509,-4.56594 -9.398263,-6.848949 -16.13326,-6.848949 z m 24.908603,1.386686 0,37.634676 12.599304,0 0,-37.634676 -12.599304,0 z M 73.835252,56.975981 c 2.304752,0 4.263793,0.852337 5.877124,2.554426 1.638928,1.675076 2.458402,3.727881 2.458402,6.159457 0,2.458578 -0.806671,4.538022 -2.419992,6.240111 -1.613331,1.675086 -3.585175,2.514099 -5.915534,2.514099 -2.612051,0 -4.737546,-1.027366 -6.376474,-3.080682 -1.331637,-1.648053 -1.997451,-3.539154 -1.997451,-5.673528 0,-2.107362 0.665814,-3.985138 1.997451,-5.633201 1.638928,-2.053316 3.764423,-3.080682 6.376474,-3.080682 z",
    fill: "#fff",
  }),
)

export const anthropicSvg = s(
  "svg",
  {
    xmlns: "http://www.w3.org/2000/svg",
    viewBox: "0 0 36 25",
    "data-icon": "anthropic",
    width: "36",
    height: "25",
    fill: "none",
    role: "img",
  },
  s("path", {
    d: "M24.8304 0H19.5612L29.1696 24.1071H34.4388L24.8304 0Z",
    fill: "#1F1F1E",
  }),
  s("path", {
    d: "M9.60842 0L0 24.1071H5.37245L7.33753 19.0446H17.3895L19.3546 24.1071H24.727L15.1186 0H9.60842ZM9.07531 14.5676L12.3635 6.09566L15.6517 14.5676H9.07531Z",
    fill: "#1F1F1E",
  }),
)

export const bskySvg = s(
  "svg",
  {
    viewBox: "0 0 512 512",
    focusable: "false",
    version: "1.1",
    xmlns: "http://www.w3.org/2000/svg",
    "data-icon": "bsky",
    width: "1em",
    height: "1em",
    role: "img",
    ariaLabel: "true",
  },
  s("path", {
    d: "M111.8 62.2C170.2 105.9 233 194.7 256 242.4c23-47.6 85.8-136.4 144.2-180.2c42.1-31.6 110.3-56 110.3 21.8c0 15.5-8.9 130.5-14.1 149.2C478.2 298 412 314.6 353.1 304.5c102.9 17.5 129.1 75.5 72.5 133.5c-107.4 110.2-154.3-27.6-166.3-62.9l0 0c-1.7-4.9-2.6-7.8-3.3-7.8s-1.6 3-3.3 7.8l0 0c-12 35.3-59 173.1-166.3 62.9c-56.5-58-30.4-116 72.5-133.5C100 314.6 33.8 298 15.7 233.1C10.4 214.4 1.5 99.4 1.5 83.9c0-77.8 68.2-53.4 110.3-21.8z",
    fill: "#1185fe",
  }),
)

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
    numLimits: 9,
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
          <div class="clickable-container">
            {Object.entries(TopLinks).map(([name, url]) => (
              <AliasLink isInternal enablePopover={false} key={name} name={name} url={url} />
            ))}
            <AliasLink newTab classes={["external"]} name="notes" url="https://notes.aarnphm.xyz" />
          </div>
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
              <svg
                id="curius-refetch"
                aria-labelledby="refetch"
                data-tooltip="refresh"
                data-id="refetch"
                height="12"
                type="button"
                viewBox="0 0 24 24"
                width="12"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <use href="#refetch-icon" />
              </svg>
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
      <ul class="section-ul" id="trail-list" />
    </div>
  )
}

export function renderPage(
  ctx: BuildCtx,
  slug: FullSlug,
  componentData: QuartzComponentProps,
  components: RenderComponents,
  pageResources: StaticResources,
  headerStyle: "main-col" | "full-col" = "main-col",
): string {
  // make a deep copy of the tree so we don't remove the transclusion references
  // for the file cached in contentMap in build.ts
  const root = clone(componentData.tree) as Root
  // NOTE: set componentData.tree to the edited html that has transclusions rendered
  componentData.tree = transcludeFinal(ctx, root, componentData)

  if (slug === "index") {
    components = {
      ...components,
      header: [],
      sidebar: [],
      afterBody: [],
      beforeBody: [],
      pageBody: (props: QuartzComponentProps) => {
        const { displayClass } = props
        const Element = ElementComponent()
        const Search = SearchConstructor({ includeButton: false })
        const Graph = GraphConstructor({
          repelForce: 2.3385416666667,
          centerForce: 0.588020833333333,
        })

        return (
          <>
            <h1 class="article-title" style="margin-top: 2rem" lang="fr">
              Bonjour, je suis Aaron.
            </h1>
            <div class={classNames(displayClass, "landing")}>
              <Element {...componentData} />
              <Search {...componentData} />
              <Graph {...componentData} />
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
      afterBody: [],
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

  const disablePageFooter = componentData.fileData.frontmatter?.poem || slug === "curius"

  // Filter out components that should be skipped during serve
  const serve = (components: RenderComponents) => {
    if (ctx.argv.serve) {
      for (const [key, comps] of Object.entries(components)) {
        if (Array.isArray(comps)) {
          components[key as keyof RenderComponents] = Array.from(
            comps.filter((comp) => !comp.skipDuringServe),
          ) as QuartzComponent[] & QuartzComponent
        }
      }
    }
    return components
  }

  const {
    head: Head,
    header,
    beforeBody,
    pageBody: Content,
    afterBody,
    sidebar,
    footer: Footer,
  } = serve(components)
  const Header = HeaderConstructor()

  // TODO: https://thesolarmonk.com/posts/a-spacebar-for-the-web style
  const lang =
    (componentData.fileData.frontmatter?.lang ?? componentData.cfg.locale)?.split("-")[0] ?? "en"
  const pageLayout = componentData.fileData.frontmatter?.pageLayout ?? "default"
  const doc = (
    <html lang={lang}>
      <Head {...componentData} />
      <body data-slug={slug} data-language={lang} data-menu={isMenu} data-layout={pageLayout}>
        <main
          data-scroll-container
          id="quartz-root"
          class="page grid"
          style={{ gridTemplateRows: "repeat(5, auto)" }}
        >
          <Header {...componentData} headerStyle={headerStyle}>
            {header.map((HeaderComponent) => (
              <HeaderComponent {...componentData} />
            ))}
          </Header>
          <section id="stacked-notes-container" class="all-col">
            <div id="stacked-notes-main">
              <div class="stacked-notes-column" />
            </div>
          </section>
          {beforeBody.length > 0 ? (
            <section class="page-header popover-hint grid all-col">
              {beforeBody.map((BodyComponent) => (
                <BodyComponent {...componentData} />
              ))}
            </section>
          ) : (
            <></>
          )}
          <section
            class={classNames(
              undefined,
              "page-content",
              slug === "index" ? "side-col" : "grid all-col",
            )}
          >
            {sidebar.length > 0 ? (
              <aside class="aside-container left-col">
                {sidebar.map((BodyComponent) => (
                  <BodyComponent {...componentData} />
                ))}
              </aside>
            ) : (
              <></>
            )}
            <Content {...componentData} />
          </section>
          {disablePageFooter ? (
            <></>
          ) : afterBody.length > 0 ? (
            <section class={classNames(undefined, "page-footer", "all-col", "grid")}>
              {afterBody.map((BodyComponent) => (
                <BodyComponent {...componentData} />
              ))}
            </section>
          ) : (
            <></>
          )}
          {slug !== "index" && <Footer {...componentData} />}
          {htmlToJsx(
            componentData.fileData.filePath!,
            s(
              "svg.quartz-icons",
              {
                xmlns: "http://www.w3.org/2000/svg",
                viewbox: "0 0 24 24",
                style: "height: 0;",
                "data-singleton": true,
              },
              [
                s("symbol", { id: "arrow-up", viewbox: "0 0 24 24" }, [
                  s("path", {
                    d: "M12 3l7 7-1.4 1.4L13 6.8V21h-2V6.8L6.4 11.4 5 10l7-7z",
                    fill: "currentColor",
                  }),
                ]),
                s("symbol", { id: "arrow-down", viewbox: "0 0 24 24" }, [
                  s("path", {
                    d: "M12 21l-7-7 1.4-1.4L11 17.2V3h2v14.2l4.6-4.6L19 14l-7 7z",
                    fill: "currentColor",
                  }),
                ]),
                s("symbol", { id: "plus-icon", viewbox: "0 0 24 24" }, [
                  s("line", { x1: 12, y1: 5, x2: 12, y2: 19 }),
                  s("line", { x1: 5, y1: 12, x2: 19, y2: 12 }),
                ]),
                s("symbol", { id: "minus-icon", viewbox: "0 0 24 24" }, [
                  s("line", { x1: 5, y1: 12, x2: 19, y2: 12 }),
                ]),
                s("symbol", { id: "circle-icon", viewbox: "0 0 24 24" }, [
                  s("circle", { cx: 12, cy: 12, r: 3 }),
                ]),
                s("symbol", { id: "zoom-in", viewbox: "0 0 24 24" }, [
                  s("path", {
                    d: "M15.5 14h-.79l-.28-.27A6.47 6.47 0 0 0 16 9.5 6.5 6.5 0 1 0 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14zm.5-7H9v2H7v1h2v2h1v-2h2V9h-2z",
                    fill: "currentColor",
                  }),
                ]),
                s("symbol", { id: "zoom-out", viewbox: "0 0 24 24" }, [
                  s("path", {
                    d: "M15.5 14h-.79l-.28-.27A6.47 6.47 0 0 0 16 9.5 6.5 6.5 0 1 0 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14zM7 9h5v1H7z",
                    fill: "currentColor",
                  }),
                ]),
                s("symbol", { id: "expand-sw-ne", viewbox: "0 0 24 24" }, [
                  s("path", {
                    d: "M4 20v-5h2v2.17L17.17 6H15V4h5v5h-2V6.83L6.83 18H9v2z",
                    fill: "currentColor",
                  }),
                ]),
                s("symbol", { id: "expand-e-w", viewbox: "0 0 24 24" }, [
                  s("path", {
                    d: "M3.72 3.72a.75.75 0 011.06 1.06L2.56 7h10.88l-2.22-2.22a.75.75 0 011.06-1.06l3.5 3.5a.75.75 0 010 1.06l-3.5 3.5a.75.75 0 11-1.06-1.06l2.22-2.22H2.56l2.22 2.22a.75.75 0 11-1.06 1.06l-3.5-3.5a.75.75 0 010-1.06l3.5-3.5z",
                    fillrule: "evenodd",
                  }),
                ]),
                s("symbol", { id: "triple-dots", viewbox: "0 0 24 24" }, [
                  s("circle", { cx: 6, cy: 12, r: 2 }),
                  s("circle", { cx: 12, cy: 12, r: 2 }),
                  s("circle", { cx: 18, cy: 12, r: 2 }),
                ]),
                s("symbol", { id: "github-copy", viewbox: "0 0 24 24" }, [
                  s("path", {
                    fillrule: "evenodd",
                    d: "M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 010 1.5h-1.5a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-1.5a.75.75 0 011.5 0v1.5A1.75 1.75 0 019.25 16h-7.5A1.75 1.75 0 010 14.25v-7.5z",
                  }),
                  s("path", {
                    fillrule: "evenodd",
                    d: "M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0114.25 11h-7.5A1.75 1.75 0 015 9.25v-7.5zm1.75-.25a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-7.5a.25.25 0 00-.25-.25h-7.5z",
                  }),
                ]),
                s("symbol", { id: "github-check", viewbox: "0 0 24 24" }, [
                  s("path", {
                    fillrule: "evenodd",
                    fill: "rgb(63, 185, 80)",
                    d: "M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z",
                  }),
                ]),
                s("symbol", { id: "github-anchor", viewbox: "0 0 24 24" }, [
                  s("path", { d: "M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71" }),
                  s("path", { d: "M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" }),
                ]),
                s("symbol", { id: "arrow-ne", viewbox: "0 0 24 24" }, [
                  s("path", { d: "M4.5 11.5l7-7" }),
                  s("path", { d: "M6.5 4.5h5v5" }),
                ]),
                s("symbol", { id: "code-icon", viewbox: "0 0 24 24" }, [
                  s("path", { d: "m18 16 4-4-4-4" }),
                  s("path", { d: "m6 8-4 4 4 4" }),
                  s("path", { d: "m14.5 4-5 16" }),
                ]),
                s("symbol", { id: "refetch-icon", viewbox: "0 0 24 24" }, [
                  s("path", {
                    d: "M17.65 6.35c-1.63-1.63-3.94-2.57-6.48-2.31-3.67.37-6.69 3.35-7.1 7.02C3.52 15.91 7.27 20 12 20c3.19 0 5.93-1.87 7.21-4.56.32-.67-.16-1.44-.9-1.44-.37 0-.72.2-.88.53-1.13 2.43-3.84 3.97-6.8 3.31-2.22-.49-4.01-2.3-4.48-4.52C5.31 9.44 8.26 6 12 6c1.66 0 3.14.69 4.22 1.78l-1.51 1.51c-.63.63-.19 1.71.7 1.71H19c.55 0 1-.45 1-1V6.41c0-.89-1.08-1.34-1.71-.71z",
                  }),
                ]),
              ],
            ),
          )}
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
          crossorigin={"anonymous"}
        />
      )}
      {/* End Cloudflare Web Analytics */}
    </html>
  )

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
` + render(doc)
  )
}
