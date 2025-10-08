import { visit } from "unist-util-visit"
import { Root as MdRoot, Link, Html, PhrasingContent, BlockContent, Paragraph, Text } from "mdast"
import { toMarkdown } from "mdast-util-to-markdown"
import { gfmToMarkdown } from "mdast-util-gfm"
import { mathToMarkdown } from "mdast-util-math"
import { QuartzTransformerPlugin } from "../types"
import { QuartzPluginData } from "../vfile"
import { FullSlug, isAbsoluteURL } from "../../util/path"
import { clone } from "../../util/clone"
import { fromHtml } from "hast-util-from-html"
import { toMdast } from "hast-util-to-mdast"
import { toString } from "mdast-util-to-string"
import { wikilinkToMarkdown } from "../../util/ofm-wikilink"

const stripKnownExtensions = (value: string) =>
  value.replace(/\.(md|mdx|markdown)$/i, "").replace(/\.html?$/i, "")

const normalizeSlug = (currentSlug: FullSlug, target: string): string | null => {
  const trimmed = target.trim()
  if (trimmed.length === 0) {
    return currentSlug
  }

  const anchorIdx = trimmed.indexOf("#")
  const anchor = anchorIdx >= 0 ? trimmed.slice(anchorIdx) : ""
  let bare = anchorIdx >= 0 ? trimmed.slice(0, anchorIdx) : trimmed

  if (bare.startsWith("http://") || bare.startsWith("https://")) {
    return null
  }

  if (bare.startsWith("/")) {
    bare = bare.slice(1)
  } else if (bare.startsWith("./") || bare.startsWith("../")) {
    const segments = currentSlug.split("/").filter(Boolean).slice(0, -1)
    for (const segment of bare.split("/")) {
      if (segment === "" || segment === ".") continue
      if (segment === "..") {
        segments.pop()
        continue
      }
      segments.push(segment)
    }
    bare = segments.join("/")
  } else if (bare.length === 0) {
    bare = currentSlug
  }

  bare = stripKnownExtensions(bare)
  return `${bare}${anchor}`
}

const extractAttribute = (value: string, attr: string): string | null => {
  const double = new RegExp(`${attr}\\s*=\\s*"([^"]*)"`, "i").exec(value)
  if (double && double[1] !== undefined) {
    return double[1]
  }
  const single = new RegExp(`${attr}\\s*=\\s*'([^']*)'`, "i").exec(value)
  return single && single[1] !== undefined ? single[1] : null
}

const convertLinksToRefs = (tree: MdRoot, slug: FullSlug) => {
  visit(tree, "link", (node: Link, index, parent) => {
    if (!parent || index == null) return
    if (!node.url || isAbsoluteURL(node.url) || node.url.startsWith("#")) return

    const resolvedSlug = normalizeSlug(slug, node.url)
    if (!resolvedSlug) return

    const children = clone(node.children ?? []) as PhrasingContent[]
    const hadDisplay = children.length > 0
    if (!hadDisplay) {
      const fallbackBase = resolvedSlug.split("#")[0]
      const segments = fallbackBase.split("/")
      const fallback = segments[segments.length - 1] ?? fallbackBase
      children.push({ type: "text", value: fallback })
    }

    parent.children.splice(index, 1, {
      type: "html",
      value: `<ref slug="${resolvedSlug}">`,
    })
  })
}

const convertTranscludesToRefs = (tree: MdRoot, slug: FullSlug) => {
  visit(tree, "html", (node: Html, index, parent) => {
    if (!parent || index == null) return
    const { value } = node
    if (!value || !/transclude/.test(value)) return

    const dataSlug = extractAttribute(value, "data-slug")
    const dataUrl = extractAttribute(value, "data-url")
    const dataBlock = extractAttribute(value, "data-block") ?? ""

    const base = normalizeSlug(slug, dataSlug ?? dataUrl ?? "")
    if (!base) return

    parent.children.splice(index, 1, {
      type: "html",
      value: `<ref slug="${base}${dataBlock}">`,
    })
  })
}

const normalizeClassList = (className: unknown): string[] => {
  if (Array.isArray(className)) {
    return className.map((c) => c.toString())
  }
  if (typeof className === "string") {
    return className.split(/\s+/).filter(Boolean)
  }
  return []
}

const extractCalloutTitle = (htmlValue: string): PhrasingContent[] => {
  const match =
    /<div[^>]*class=["'][^"']*callout-title-inner[^"']*["'][^>]*>([\s\S]*?)<\/div>/i.exec(htmlValue)
  if (!match) return []

  const fragment = fromHtml(match[1], { fragment: true })
  const md = toMdast(fragment) as MdRoot

  const titleChildren: PhrasingContent[] = []
  for (const child of md.children) {
    if (child.type === "paragraph") {
      titleChildren.push(...(child.children as PhrasingContent[]))
    } else if (child.type === "text") {
      titleChildren.push(child as PhrasingContent)
    }
  }

  const contentText = toString({ type: "paragraph", children: titleChildren } as Paragraph).trim()
  if (contentText.length === 0) {
    return []
  }

  return titleChildren
}

const convertCalloutsToMarkdown = (tree: MdRoot) => {
  visit(tree, "blockquote", (node) => {
    const props = (node.data?.hProperties ?? {}) as Record<string, unknown>
    const classes = normalizeClassList(props.className)
    if (!classes.includes("callout")) return

    const calloutType = (props["data-callout"] ?? "note").toString()
    const metadataValue = props["data-callout-metadata"]?.toString().trim()
    const metadata = metadataValue ? `|${metadataValue}` : ""
    const collapseChar = classes.includes("is-collapsible")
      ? classes.includes("is-collapsed")
        ? "-"
        : "+"
      : ""

    let titleChildren: PhrasingContent[] = []
    const firstChild = node.children.at(0)
    if (firstChild && firstChild.type === "html") {
      titleChildren = extractCalloutTitle(firstChild.value)
    }

    const prefix: Text = {
      type: "text",
      value: `[!${calloutType}${metadata}]${collapseChar}`,
    }

    const header: Paragraph = { type: "paragraph", children: [prefix] }
    if (titleChildren.length > 0) {
      header.children.push({ type: "text", value: " " })
      header.children.push(...titleChildren)
    }

    const content: BlockContent[] = []
    for (const child of node.children.slice(1)) {
      if (
        child.type === "blockquote" &&
        child.data?.hProperties &&
        normalizeClassList(child.data.hProperties.className).includes("callout-content")
      ) {
        content.push(...((child.children as BlockContent[]) ?? []))
      } else if (child.type !== "html") {
        content.push(child as BlockContent)
      }
    }

    node.children = [header, ...content]
    if (node.data) {
      delete node.data.hProperties
    }
  })
}

export const LLM: QuartzTransformerPlugin = () => {
  return {
    name: "LLM",
    markdownPlugins({ argv }) {
      if (argv.watch && !argv.force) return []

      return [
        () => {
          return (tree: MdRoot, file) => {
            const fileData = file.data as QuartzPluginData
            if (!fileData.slug) return

            const cloned = clone(tree) as MdRoot
            const slug = fileData.slug as FullSlug

            cloned.children = cloned.children.filter((child) => child.type !== "yaml")

            convertCalloutsToMarkdown(cloned)
            convertLinksToRefs(cloned, slug)
            convertTranscludesToRefs(cloned, slug)

            const llmsText = toMarkdown(cloned, {
              bullet: "-",
              emphasis: "_",
              rule: "-",
              extensions: [mathToMarkdown(), gfmToMarkdown(), wikilinkToMarkdown()],
            })

            fileData.llmsText = llmsText
          }
        },
      ]
    },
  }
}
