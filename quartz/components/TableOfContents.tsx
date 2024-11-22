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

function extractTransclude(root: Root, allFiles: QuartzPluginData[]): TocEntry[] {
  const entries: TocEntry[] = []

  visit(root, "element", (node, _index, _parent) => {
    const classNames = (node.properties?.className ?? []) as string[]
    if (node.tagName === "blockquote") {
      if (classNames.includes("transclude")) {
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
                const refs = h.children[0] as Element
                console.log(refs, page.toc)
                return page.toc!.find((it) => it.slug === refs.properties.id)
              })
            console.log(contentSlice, page, node)
            entries.push(...contentSlice)
          }
        } else if (page.htmlAst) {
          // Handle full page transcludes
          entries.push(...page.toc)
        }
      }
    }
  })

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

    // const entries = extractTransclude(clone(tree) as Root, allFiles)
    // console.log(entries)

    return (
      <div class={classNames(displayClass, "toc")} data-layout={opts.layout}>
        {opts.layout === "minimal" ? (
          <nav id="toc-vertical">
            {fileData.toc.map((entry) => (
              <a
                key={entry.slug}
                class={`depth-${entry.depth}`}
                href={`#${entry.slug}`}
                data-for={entry.slug}
                data-hover={entry.text}
              />
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
