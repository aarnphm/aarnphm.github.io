import { render } from "preact-render-to-string"
import { QuartzComponent, QuartzComponentProps } from "./types"
import HeaderConstructor from "./Header"
import BodyConstructor from "./Body"
import LandingConstructor from "./Landing"
import { JSResourceToScriptElement, StaticResources } from "../util/resources"
import { clone, FullSlug, RelativeURL, joinSegments, normalizeHastElement } from "../util/path"
import { visit } from "unist-util-visit"
import { Root, Element, ElementContent, Node } from "hast"
import { GlobalConfiguration } from "../cfg"
import { i18n } from "../i18n"
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

const headerRegex = new RegExp(/h[1-6]/)

function headerElement(node: Element, content: Element[], idx: number): Element {
  const buttonId = `collapsible-header-${node.properties?.id ?? idx}`

  const id = node.properties?.id ?? idx
  // indicate whether the header is collapsed or not
  const lastIdx = node.children.length > 0 ? node.children.length - 1 : 0
  node.children.splice(lastIdx, 0, {
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
  })

  return {
    type: "element",
    tagName: "div",
    properties: {
      className: ["collapsible-header"],
      "data-level": node.tagName[1],
      id,
    },
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
        properties: {
          className: ["collapsible-header-content-outer"],
        },
        children: [
          {
            type: "element",
            tagName: "div",
            properties: {
              className: ["collapsible-header-content"],
              ["data-references"]: `${buttonId}-toggle`,
            },
            children: content,
          },
        ],
      },
    ],
  }
}

function processHeaders(node: Element, idx: number | undefined, parent: Element) {
  idx = idx ?? parent.children.indexOf(node)
  const currentLevel = parseInt(node.tagName[1])
  const contentNodes: Element[] = []
  let i = idx + 1

  // Collect all content until next header of same or higher level
  while (i < parent.children.length) {
    const nextNode = parent.children[i] as Element
    if (
      (nextNode?.type === "element" && nextNode.properties.dataReferences == "") ||
      (nextNode?.type === "element" && nextNode.properties.dataFootnotes == "") ||
      (nextNode?.type === "element" && ["hr"].includes(nextNode.tagName))
    ) {
      break
    }

    if (nextNode?.type === "element" && nextNode.tagName?.match(headerRegex)) {
      const nextLevel = parseInt(nextNode.tagName[1])
      if (nextLevel <= currentLevel) {
        break
      }
      // Process nested header recursively
      processHeaders(nextNode, i, parent)

      // After processing, the next node at index i will be the wrapper
      contentNodes.push(parent.children[i] as Element)
      parent.children.splice(i, 1)
    } else {
      contentNodes.push(nextNode)
      parent.children.splice(i, 1)
    }
  }

  parent.children.splice(idx, 1, headerElement(node, contentNodes, idx))
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

  visit(root, "element", (node: Element) => {
    if (node.type === "element" && node.tagName === "a" && node.properties.dataFootnoteRef === "") {
      orderNotes.push({ href: node.properties.href as string, id: node.properties.id as string })
      node.properties.href = `${node.properties.href}${appendSuffix !== undefined ? "-" + appendSuffix : ""}`
      node.properties.id =
        node.properties.id + `${appendSuffix !== undefined ? "-" + appendSuffix : ""}`
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
      visit(node, "element", (entry, index, parent) => {
        if (entry.tagName === "ol") {
          entry.children = sortedRefs
          parent!.children.splice(index as number, 1, entry)
        }
      })
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

type TranscludeOptions = {
  dynalist: boolean
}

const defaultTranscludeOptions = { dynalist: true }

export function transcludeFinal(
  root: Root,
  { cfg, allFiles, fileData }: QuartzComponentProps,
  userOpts?: TranscludeOptions,
): Root {
  const slug = fileData.slug as FullSlug
  let opts: TranscludeOptions
  if (userOpts) {
    opts = { ...defaultTranscludeOptions, ...userOpts }
  } else {
    opts = defaultTranscludeOptions
  }
  const { dynalist } = opts

  // NOTE: process transcludes in componentData
  visit(root, "element", (node, _index, _parent) => {
    if (node.tagName === "blockquote") {
      const classNames = (node.properties?.className ?? []) as string[]
      if (classNames.includes("transclude")) {
        const inner = node.children[0] as Element
        const transcludeTarget = inner.properties["data-slug"] as FullSlug
        const page = allFiles.find((f) => f.slug === transcludeTarget)
        if (!page) {
          return
        }

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
            if (!(el.type === "element" && el.tagName.match(headerRegex))) continue
            const depth = Number(el.tagName.substring(1))

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
            (page.frontmatter?.transclude?.title ?? true)
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
  if (dynalist && !slug.includes("posts")) {
    // NOTE: handling collapsible nodes
    visit(root, "element", (node: Element, idx, parent) => {
      const denyIds = new Set(["footnote-label", "reference-label"])
      if (
        slug !== "index" &&
        node.tagName.match(headerRegex) &&
        !denyIds.has(node.properties.id as string) &&
        !(fileData.frontmatter?.menu ?? false) &&
        (fileData.frontmatter?.collapsible ?? true)
      ) {
        // then do the process headers and its children here
        processHeaders(node, idx, parent as Element)
      }
    })
  }
  // NOTE: We then merge all references and footnotes to final items
  mergeIsomorphic(root)
  return root
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
  const Landing = LandingConstructor()

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

  const lang = componentData.fileData.frontmatter?.lang ?? cfg.locale?.split("-")[0] ?? "en"
  const doc = (
    <html lang={lang}>
      <Head {...componentData} />
      <body data-slug={slug} data-menu={componentData.fileData.frontmatter?.menu ?? false}>
        {slug === "index" ? (
          <Landing {...componentData} />
        ) : (
          <div id="quartz-root" class="page">
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
          </div>
        )}
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
