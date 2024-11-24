import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import modernStyle from "./styles/toc.scss"
import { classNames } from "../util/lang"
// @ts-ignore
import script from "./scripts/toc.inline"
import { i18n } from "../i18n"
import { fromHtml } from "hast-util-from-html"
import { htmlToJsx } from "../util/jsx"
import { SKIP, visit } from "unist-util-visit"
import { clone, FullSlug } from "../util/path"
import { Root, Element, ElementContent, Text } from "hast"
import { headingRank } from "hast-util-heading-rank"
import { TocEntry } from "../plugins/transformers/toc"
import Slugger from "github-slugger"
import { QuartzPluginData } from "../plugins/vfile"
import { toString } from "hast-util-to-string"

const ghSlugger = new Slugger()

// NOTE: We will mutate fileData.toc here.
function extractTransclude(
  root: Root,
  allFiles: QuartzPluginData[],
  fileData: QuartzPluginData,
  toc: TocEntry[],
): TocEntry[] {
  const entries: TocEntry[] = []
  let insertIdx: number

  visit(root, "element", (node, index, parent) => {
    const classNames = (node.properties?.className ?? []) as string[]
    if (node.tagName === "blockquote" && classNames.includes("transclude")) {
      // TODO: extract the nearest headers from given root and extract the position from toc
      if (parent!.type === "root") {
        // Find the nearest preceding TOC entry to determine where to insert
        let nearestTocIndex = -1
        for (let i = index! - 1; i >= 0; i--) {
          const prevSibling = parent!.children[i] as Element
          if (prevSibling.properties["data-level"]) {
            const headingId = prevSibling.properties["data-heading-id"]
            nearestTocIndex = toc.findIndex((entry) => entry.id === headingId)
            if (nearestTocIndex !== -1) break
          }
        }
        // If no preceding header found, append to end, otherwise insert after nearest header
        insertIdx = nearestTocIndex === -1 ? toc.length : nearestTocIndex
      } else {
        insertIdx = fileData.toc.find(
          (h) => h.id === (parent as Element).properties["data-heading-id"],
        )
      }

      const transcludeTarget = node.properties.dataUrl as FullSlug
      const page = allFiles.find((f) => f.slug === transcludeTarget)

      if (!page?.htmlAst || !page?.toc) return SKIP

      let blockRef = node.properties.dataBlock as string | undefined
      if (blockRef?.startsWith("#^")) {
        // Handle block transcludes
        blockRef = blockRef.slice("#^".length)
        const blockNode = page.blocks?.[blockRef]
        if (blockNode) {
          visit(blockNode, "element", (node) => {
            if (headingRank(node)) {
              const text = (node.children[0] as Text).value
              const toc = page.toc!.find((it) => it.slug === text) as TocEntry
              entries.push(toc)
            }
          })
        }
      } else if (blockRef?.startsWith("#") && page.htmlAst) {
        // Handle header transcludes
        blockRef = blockRef.slice(1)
        let startIdx = undefined
        let startDepth = undefined
        let endIdx = undefined

        for (const [i, el] of page.htmlAst.children.entries()) {
          const depth = headingRank(el)
          if (!(el.type === "element" && depth)) continue

          if (!startIdx && !startDepth) {
            if (el.properties?.id === blockRef) {
              startIdx = i
              startDepth = depth
            }
          } else if (depth <= startDepth!) {
            endIdx = i
            break
          }
        }

        if (startIdx !== undefined) {
          const contentSlice = (page.htmlAst.children.slice(startIdx, endIdx) as Element[])
            .filter((s) => headingRank(s))
            .map((h) => {
              const startToc = page.toc!.find((it) => it.slug === h.properties.id) as TocEntry
              const startTocIdx = page.toc!.indexOf(startToc)
              let endTocIdx: number = page.toc?.length!
              for (let i = startTocIdx; i <= page.toc?.length! - 1; i++) {
                const maybeNextLevel = page.toc![i] as TocEntry
                if (maybeNextLevel.depth === startToc?.depth) {
                  endTocIdx = i
                  break
                }
              }
              return page.toc!.slice(startTocIdx, endTocIdx)
            })
            .flat()
          entries.push(...contentSlice)
        }
      } else if (page.htmlAst) {
        // Handle full page transcludes
        entries.push(...page.toc)
      }
    }
  })

  console.log(entries, insertIdx)

  return entries
}

interface Options {
  layout: "minimal" | "default"
}

const defaultOptions: Options = {
  layout: "minimal",
}

export default ((userOpts?: Partial<Options>) => {
  const opts = { ...defaultOptions, ...userOpts }

  const TableOfContents: QuartzComponent = ({
    fileData,
    displayClass,
    cfg,
    tree,
    allFiles,
  }: QuartzComponentProps) => {
    if (!fileData.toc) {
      return null
    }
    ghSlugger.reset()

    const convertFromText = (text: string) => {
      const tocAst = fromHtml(text, { fragment: true })
      return htmlToJsx(fileData.filePath!, tocAst)
    }

    // const entries = extractTransclude(clone(tree) as Root, allFiles, fileData.toc)

    return (
      <div class={classNames(displayClass, "toc")} id="toc" data-layout={opts.layout}>
        {opts.layout === "minimal" ? (
          <nav id="toc-vertical">
            {fileData.toc.map((entry, idx) => (
              <button
                key={entry.slug}
                class="toc-item"
                data-depth={entry.depth}
                data-href={`#${entry.slug}`}
                data-for={entry.slug}
                style={{ "--animation-order": idx + 1 }}
              >
                <div class="fill" />
                <div class="indicator">{convertFromText(entry.text)}</div>
              </button>
            ))}
          </nav>
        ) : (
          <>
            <button type="button" id="toc" aria-controls="toc-content">
              <h3>{i18n(cfg.locale).components.tableOfContents.title}</h3>
            </button>
            <div id="toc-content">
              <ul class="overflow">
                {fileData.toc.map((entry) => (
                  <li key={entry.slug} class={`depth-${entry.depth}`}>
                    <a href={`#${entry.slug}`} data-for={entry.slug}>
                      {convertFromText(entry.text)}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          </>
        )}
      </div>
    )
  }

  TableOfContents.css = modernStyle
  TableOfContents.afterDOMLoaded = script
  return TableOfContents
}) satisfies QuartzComponentConstructor
