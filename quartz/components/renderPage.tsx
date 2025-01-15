import { render } from "preact-render-to-string"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import HeaderConstructor from "./Header"
import ContentConstructor from "./pages/Content"
import FooterConstructor from "./Footer"
import SearchConstructor from "./Search"
import { byDateAndAlphabetical } from "./PageList"
import { getDate, Date } from "./Date"
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
import mermaidScript from "./scripts/mermaid.inline"
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
        arialabel: "Toggle content visibility",
        role: "button",
        ariaexpanded: true,
        ariacontrols: `${buttonId}-content`,
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
    ])
  }

  // NOTE: process transcludes in componentData
  visit(root, "element", (node) => {
    if (node.tagName === "blockquote") {
      const classNames = (node.properties?.className ?? []) as string[]
      const url = node.properties.dataUrl as string
      const alias = (
        node.properties?.dataEmbedAlias !== "undefined"
          ? node.properties?.dataEmbedAlias
          : node.properties?.dataBlock
      ) as string

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
            h("a", { href: inner.properties?.href, class: "internal transclude-src" }, [
              { type: "text", value: i18n(cfg.locale).components.transcludes.linkToOriginal },
            ]),
          ]

          if (fileData.frontmatter?.pageLayout !== "reflection") {
            children.push(
              h("a", { href: inner.properties?.href, class: "internal transclude-src" }, [
                { type: "text", value: i18n(cfg.locale).components.transcludes.linkToOriginal },
              ]),
            )
          }
          node.children = children
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

export const links = {
  colophon: "/colophon",
  livres: "/books",
  advices: "/quotes",
  parfum: "/thoughts/Scents",
  curius: "/curius",
  tweets: "/tweets",
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
                          <Date date={getDate(cfg, page)!} locale={cfg.locale} />
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
    const githubIcon = svgToJsx(
      s(
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
      ),
    )
    const twitterIcon = svgToJsx(
      s(
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
      ),
    )
    const bskyIcon = svgToJsx(
      s(
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
      ),
    )
    const substackIcon = svgToJsx(
      s(
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
      ),
    )

    const Hyperlink = HyperlinksComponent({
      children: [
        <section>
          <h2>jardin:</h2>
          <div class="clickable-container">
            {Object.entries(links).map(([name, url]) => (
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
  cfg: GlobalConfiguration,
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
  componentData.tree = transcludeFinal(root, componentData)

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

        return (
          <>
            <h1 class="article-title" style="margin-top: 2rem" lang="fr">
              Bonjour, je suis Aaron.
            </h1>
            <div class={classNames(displayClass, "landing")}>
              <Element {...componentData} />
              <Search {...componentData} />
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
      sidebar: [CuriusTrail, CuriusFriends],
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
  const lang = (componentData.fileData.frontmatter?.lang ?? cfg.locale)?.split("-")[0] ?? "en"
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
