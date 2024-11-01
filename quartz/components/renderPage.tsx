import { render } from "preact-render-to-string"
import { QuartzComponent, QuartzComponentProps } from "./types"
import HeaderConstructor from "./Header"
import BodyConstructor from "./Body"
import LandingConstructor from "./Landing"
import { JSResourceToScriptElement, StaticResources } from "../util/resources"
import { clone, FullSlug, RelativeURL, joinSegments, normalizeHastElement } from "../util/path"
import { visit } from "unist-util-visit"
import { Root, Element, ElementContent } from "hast"
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

function headerElement(node: Element, content: Element[], idx: number): Element {
  const buttonId = `collapsible-header-${node.properties?.id ?? idx}`
  return {
    type: "element",
    tagName: "div",
    properties: {
      className: ["collapsible-header"],
      "data-level": node.tagName[1],
    },
    children: [
      {
        type: "element",
        tagName: "button",
        properties: {
          id: buttonId,
          ariaLabel: "Toggle content visibility",
          ariaExpanded: true,
          className: ["header-button"],
        },
        children: [
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
              ["data-references"]: buttonId,
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
      (["div"].includes(nextNode.tagName) && nextNode.properties.id == "refs") ||
      (nextNode?.type === "element" &&
        nextNode.tagName?.match(/^h[1-6]$/) &&
        nextNode.properties["data-footnotes"])
    ) {
      break
    }

    if (nextNode?.type === "element" && nextNode.tagName?.match(/^h[1-6]$/)) {
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

const headerRegex = new RegExp(/h[1-6]/)
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
      ...staticResources.js,
      {
        src: joinSegments(baseDir, "postscript.js"),
        loadTime: "afterDOMReady",
        moduleType: "module",
        contentType: "external",
      },
      {
        script: collapseHeaderScript,
        loadTime: "afterDOMReady",
        contentType: "inline",
      },
    ],
  }
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

  // process transcludes in componentData
  visit(root, "element", (node, _index, _parent) => {
    if (node.tagName === "blockquote") {
      const classNames = (node.properties?.className ?? []) as string[]
      if (classNames.includes("transclude")) {
        const inner = node.children[0] as Element
        const transcludeTarget = inner.properties["data-slug"] as FullSlug
        const page = componentData.allFiles.find((f) => f.slug === transcludeTarget)
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
            (page.frontmatter?.titleTransclude ?? true)
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

  visit(root, "element", (node: Element, idx, parent) => {
    if (
      node.tagName.match(headerRegex) &&
      parent &&
      node.properties.id !== "footnote-label" &&
      slug !== "index" &&
      !(componentData.fileData.frontmatter?.menu ?? false)
    ) {
      // then do the process headers and its children here
      processHeaders(node, idx, parent as Element)
    }
  })

  // set componentData.tree to the edited html that has transclusions rendered
  componentData.tree = root

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
      <body
        data-slug={slug}
        data-enable-preview={componentData.fileData.frontmatter?.preview ?? true}
        data-menu={componentData.fileData.frontmatter?.menu ?? false}
      >
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
                <hr />
                <div class="page-footer">
                  {afterBody.map((BodyComponent) => (
                    <BodyComponent {...componentData} />
                  ))}
                </div>
              </div>
              {RightComponent}
            </Body>
            <Footer {...componentData} />
          </div>
        )}
      </body>
      {pageResources.js
        .filter((resource) => resource.loadTime === "afterDOMReady")
        .map((res) => JSResourceToScriptElement(res))}
    </html>
  )

  return "<!DOCTYPE html>\n" + render(doc)
}
