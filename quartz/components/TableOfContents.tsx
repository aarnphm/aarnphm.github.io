import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import modernStyle from "./styles/toc.scss"
import { classNames } from "../util/lang"
// @ts-ignore
import script from "./scripts/toc.inline"
import { i18n } from "../i18n"
import { fromHtml } from "hast-util-from-html"
import { htmlToJsx } from "../util/jsx"
import { visit } from "unist-util-visit"
import { clone, FullSlug, simplifySlug } from "../util/path"
import { Root, Element } from "hast"
import { TocEntry } from "../plugins/transformers/toc"
import Slugger from "github-slugger"
import { QuartzPluginData } from "../plugins/vfile"
import { toText } from "hast-util-to-text"
import { headingRank } from "hast-util-heading-rank"

const ghSlugger = new Slugger()

// NOTE: We will mutate fileData.toc here.
function mutateTransclude(
  root: Root,
  allFiles: QuartzPluginData[],
  fileData: QuartzPluginData,
): TocEntry[] {
  const entries: TocEntry[] = []

  // Helper to find nearest header in parent chain
  function findNearestHeader(node: Element): string | undefined {
    let current = node
    while (current) {
      if (current.properties?.["data-level"]) {
        return current.properties?.["data-heading-id"] as string
      }
      current = current.parent as Element
    }
    return undefined
  }

  // Helper to find nearest TOC index
  function findTocIndex(headingId: string | undefined): number {
    if (!headingId) return -1
    return fileData.toc!.findIndex((entry) => entry.slug === headingId)
  }

  const tree = clone(root)
  visit(tree, "element", (node, index, parent) => {
    if (!parent || index === undefined) return
    if (headingRank(node) && node.properties.dataReader !== "") return

    const classNames = (node.properties?.className ?? []) as string[]
    if (node.tagName === "blockquote" && classNames.includes("transclude")) {
      // Case 1: Inside collapsible-header-content
      const isInCollapsible = ((parent as Element).properties?.className as string[])?.includes(
        "collapsible-header-content",
      )
      let insertIdx: number

      if (isInCollapsible) {
        // Use the parent collapsible header ID
        const headerId = findNearestHeader(parent as Element)
        insertIdx = findTocIndex(headerId)
      } else {
        // Case 2: Find nearest preceding header
        let nearestTocIndex = -1
        for (let i = index - 1; i >= 0; i--) {
          const prevSibling = parent.children[i] as Element
          if (prevSibling.properties?.["data-level"]) {
            const headingId = prevSibling.properties["data-heading-id"]
            nearestTocIndex = findTocIndex(headingId as string)
            if (nearestTocIndex !== -1) break
          }
        }
        insertIdx = nearestTocIndex === -1 ? fileData.toc!.length : nearestTocIndex + 1
      }

      // TODO: Process nested transclude inside transclude

      const transcludeTarget = node.properties?.dataUrl as FullSlug
      const page = allFiles.find((f) => f.slug === transcludeTarget)
      if (!page?.toc) return

      // Track depth for nested transcludes
      let currentDepth = 0
      const parentHeaderEl = findNearestHeader(parent as Element)
      if (parentHeaderEl) {
        currentDepth = parseInt((parent as Element).properties?.["data-level"] as string) || 0
      }

      // Handle block references
      const blockRef = node.properties?.dataBlock as string | undefined
      if (blockRef?.startsWith("#^")) {
      } else if (blockRef?.startsWith("#")) {
        // Header transclude - extract subsection
        const targetHeader = blockRef.slice(1)
        const startIdx = page.toc!.findIndex((e) => e.slug === targetHeader)
        if (startIdx !== -1) {
          const baseDepth = page.toc[startIdx].depth
          let endIdx = startIdx + 1
          while (endIdx < page.toc.length && page.toc[endIdx].depth > baseDepth) {
            endIdx++
          }
          entries.push(...page.toc.slice(startIdx, endIdx))
        }
      } else {
        // Full page transclude
        entries.push(
          ...page.toc.filter((entry) => {
            return !["backlinks-label", "footnote-label", "reference-label"].some(
              (slug) => entry.slug === slug,
            )
          }),
        )
      }

      // Deduplicate before splicing
      const uniqueEntries = entries.filter((entry) => {
        return !fileData.toc!.some(
          (existing) =>
            existing.depth === entry.depth &&
            existing.slug === entry.slug &&
            existing.text === entry.text,
        )
      })

      // Splice unique entries into main TOC
      if (uniqueEntries.length > 0) {
        fileData.toc!.splice(insertIdx + 1, 0, ...uniqueEntries)
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
    const slug = simplifySlug(fileData.slug!)
    const backlinkFiles = allFiles.filter((file) => file.links?.includes(slug))

    const convertFromText = (text: string) => {
      const tocAst = fromHtml(text, { fragment: true })
      return htmlToJsx(fileData.filePath!, tocAst)
    }

    mutateTransclude(tree as Root, allFiles, fileData)

    // const entries = extractTransclude(clone(tree) as Root, allFiles, fileData.toc)
    const sectionToc: TocEntry[] = []
    visit(tree, "element", (node: Element) => {
      if (
        node.tagName === "section" &&
        (node.properties?.dataReferences === "" || node.properties?.dataFootnotes === "")
      ) {
        const heading = node.children[0] as Element
        sectionToc.push({
          depth: 0,
          text: toText(heading),
          slug: heading.properties?.id as string,
        })
      }
    })
    fileData.toc.push(...sectionToc)

    if (backlinkFiles.length > 0) {
      fileData.toc.push({
        depth: 0,
        text: i18n(cfg.locale).components.backlinks.title,
        slug: "backlinks-label",
      })
    }

    const MinimalToc = () => (
      <nav id="toc-vertical">
        {fileData.toc!.map((entry, idx) => (
          <button
            key={entry.slug}
            class={`depth-${entry.depth} toc-item`}
            data-depth={entry.depth}
            data-href={`#${entry.slug}`}
            data-for={entry.slug}
            tabindex={-1}
            type="button"
            style={{ "--animation-order": idx + 1 }}
            aria-label={`${entry.text}`}
            title={`${entry.text}`}
          >
            <div class="fill" />
            <div class="indicator">{convertFromText(entry.text)}</div>
          </button>
        ))}
      </nav>
    )

    const DefaultToc = () => (
      <nav>
        <button type="button" id="toc" aria-controls="toc-content">
          <h3>{i18n(cfg.locale).components.tableOfContents.title}</h3>
        </button>
        <div id="toc-content">
          <ul class="overflow">
            {fileData.toc!.map((entry) => (
              <li key={entry.slug} class={`depth-${entry.depth}`}>
                <a href={`#${entry.slug}`} data-for={entry.slug}>
                  {convertFromText(entry.text)}
                </a>
              </li>
            ))}
          </ul>
        </div>
      </nav>
    )

    return (
      <div class={classNames(displayClass, "toc")} id="toc" data-layout={opts.layout}>
        {opts.layout === "minimal" ? <MinimalToc /> : <DefaultToc />}
      </div>
    )
  }

  TableOfContents.css = modernStyle
  TableOfContents.afterDOMLoaded = script
  return TableOfContents
}) satisfies QuartzComponentConstructor
