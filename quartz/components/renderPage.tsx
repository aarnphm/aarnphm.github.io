import { render } from "preact-render-to-string"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import HeaderConstructor from "./Header"
import ContentConstructor from "./pages/Content"
import GraphConstructor from "./Graph"
import SearchConstructor from "./Search"
import SpacerConstructor from "./Spacer"
import FooterConstructor from "./Footer"
import Navigation from "./Navigation"
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
import { h, s } from "hastscript"
// @ts-ignore
import collapseHeaderScript from "./scripts/collapse-header.inline.ts"
import collapseHeaderStyle from "./styles/collapseHeader.inline.scss"
//@ts-ignore
import curiusScript from "./scripts/curius.inline"
//@ts-ignore
import curiusFriendScript from "./scripts/curius-friends.inline"
import DesktopOnly from "./DesktopOnly"
import Spacer from "./Spacer"
import { htmlToJsx } from "../util/jsx"

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

  const icons = s(
    "svg",
    { ...svgOptions, class: "collapsed-dots", style: "padding-left: 0.2rem;" },
    [s("use", { href: "#triple-dots" })],
  )
  node.children.splice(lastIdx, 0, icons)

  let className = ["collapsible-header"]
  if (endHr) {
    className.push("end-hr")
  }

  const rank = headingRank(node) as number

  return h(`section.${className.join(".")}#${id}`, { "data-level": rank }, [
    h(".header-controls", [
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
      node,
    ]),
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
        loadTime: "afterDOMReady",
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
        [
          s("path", { d: "M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71" }),
          s("path", { d: "M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" }),
        ],
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

            node.children = [
              anchor(inner.properties?.href as string, url, alias, title),
              normalizeHastElement(blockNode, slug, transcludeTarget),

              h("a", { href: inner.properties?.href, class: "internal transclude-src" }, [
                { type: "text", value: i18n(cfg.locale).components.transcludes.linkToOriginal },
              ]),
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
            anchor(inner.properties?.href as string, url, alias, title),
            ...(page.htmlAst.children.slice(startIdx, endIdx) as ElementContent[]).map((child) =>
              normalizeHastElement(child as Element, slug, transcludeTarget),
            ),
            h("a", { href: inner.properties?.href, class: "internal transclude-src" }, [
              { type: "text", value: i18n(cfg.locale).components.transcludes.linkToOriginal },
            ]),
          ]
        } else if (page.htmlAst) {
          // page transclude
          node.children = [
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
            h("a", { href: inner.properties?.href, class: "internal transclude-src" }, [
              { type: "text", value: i18n(cfg.locale).components.transcludes.linkToOriginal },
            ]),
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

export const links = {
  livres: "/books",
  "boîte aux lettres": "/posts/",
  projets: "/thoughts/work",
  curius: "/curius",
  advices: "/quotes",
  parfum: "/thoughts/Scents",
  "atelier with friends": "/thoughts/atelier-with-friends",
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
                          <Date date={getDate(cfg, fileData)!} locale={cfg.locale} />
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
  const Hyperlink = HyperlinksComponent({
    children: [
      <section>
        <h2>jardin:</h2>
        <div class="clickable-container">
          {Object.entries(links).map(([name, url]) => (
            <AliasLink key={name} isInternal name={name} url={url} />
          ))}
        </div>
      </section>,
      <section>
        <h2>média:</h2>
        <address class="clickable-container">
          <AliasLink newTab name="github" url="https://github.com/aarnphm" />
          <AliasLink newTab name="twitter" url="https://x.com/aarnphm_" />
          <AliasLink newTab name="substack" url="https://livingalonealone.com" />
          <AliasLink newTab name="bluesky" url="https://bsky.app/profile/aarnphm.xyz" />
          <AliasLink name="contact" url="mailto:contact@aarnphm.xyz" />
          <AliasLink newTab name="llms.txt" url="/llms.txt" />
          <AliasLink newTab name="llms-full.txt" url="/llms-full.txt" />
        </address>
      </section>,
    ],
  })

  const Element: QuartzComponent = (componentData: QuartzComponentProps) => {
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
    <section class={classNames(displayClass, "menu")} data-function={true}>
      <a href="../atelier-with-friends" class="internal alias" data-no-popover={true}>
        atelier with friends.
      </a>
    </section>
  )
}

// Curius components

const CuriusContent: QuartzComponent = (props: QuartzComponentProps) => {
  const { cfg, displayClass } = props
  const searchPlaceholder = i18n(cfg.locale).components.search.searchBarPlaceholder
  const Footer = Navigation({ prev: "/mechinterp", next: "/books" })

  return (
    <>
      <div class={classNames(displayClass, "curius-header")}>
        <div class="curius-search">
          <input
            id="curius-bar"
            type="text"
            aria-label={searchPlaceholder}
            placeholder={searchPlaceholder}
          />
          <div id="curius-search-container"></div>
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
            <path d="M17.65 6.35c-1.63-1.63-3.94-2.57-6.48-2.31-3.67.37-6.69 3.35-7.1 7.02C3.52 15.91 7.27 20 12 20c3.19 0 5.93-1.87 7.21-4.56.32-.67-.16-1.44-.9-1.44-.37 0-.72.2-.88.53-1.13 2.43-3.84 3.97-6.8 3.31-2.22-.49-4.01-2.3-4.48-4.52C5.31 9.44 8.26 6 12 6c1.66 0 3.14.69 4.22 1.78l-1.51 1.51c-.63.63-.19 1.71.7 1.71H19c.55 0 1-.45 1-1V6.41c0-.89-1.08-1.34-1.71-.71z"></path>
          </svg>
        </div>
      </div>
      <div class={classNames(displayClass, "curius", "popover-hint")} id="curius">
        <div class="curius-page-container">
          <div id="curius-fetching-text" />
          <div id="curius-fragments" />
          <div class="highlight-modal" id="highlight-modal">
            <ul id="highlight-modal-list" />
          </div>
        </div>
      </div>
      <Footer {...props} />
    </>
  )
}
CuriusContent.afterDOMLoaded = curiusScript

const CuriusFriends: QuartzComponent = (props: QuartzComponentProps) => {
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
        const Graph = GraphConstructor()
        const Search = SearchConstructor({ includeButton: false })

        return (
          <>
            <h1 class="article-title" style="margin-top: 1rem" lang="fr">
              Bonjour, je suis Aaron.
            </h1>
            <div class={classNames(displayClass, "landing")}>
              <Element {...componentData} />
            </div>
            <Graph {...props} />
            <Search {...props} />
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
      footer: Spacer(),
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
  const lang = componentData.fileData.frontmatter?.lang ?? cfg.locale?.split("-")[0] ?? "en"
  const doc = (
    <html lang={lang}>
      <Head {...componentData} />
      <body data-slug={slug} data-language={lang} data-menu={isMenu}>
        <main id="quartz-root" class="page grid" style={{ gridTemplateRows: "repeat(5, auto)" }}>
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
          <section class="page-header popover-hint grid all-col">
            {beforeBody.map((BodyComponent) => (
              <BodyComponent {...componentData} />
            ))}
          </section>
          <section class="page-content grid all-col">
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
            <section class={classNames(undefined, "page-footer", "all-col", !isMenu ? "grid" : "")}>
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
