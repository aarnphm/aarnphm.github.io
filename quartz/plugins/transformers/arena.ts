import { QuartzTransformerPlugin } from "../types"
import { Root as MdastRoot, List, ListItem, Paragraph, PhrasingContent, Link, Text } from "mdast"
import { ElementContent } from "hast"
import { QuartzPluginData } from "../vfile"
import { toString } from "mdast-util-to-string"
import { toHast } from "mdast-util-to-hast"
import Slugger from "github-slugger"
import { clone } from "../../util/clone"
import { visit, CONTINUE } from "unist-util-visit"
import { wikiTextTransform } from "./ofm"
import { createHash } from "crypto"
import { fromMarkdown } from "mdast-util-from-markdown"

export interface ArenaBlock {
  id: string
  content: string
  url?: string
  title?: string
  titleHtmlNode?: ElementContent
  snapshotKey?: string
  subItems?: ArenaBlock[]
  highlighted?: boolean
  htmlNode?: ElementContent
}

export interface ArenaChannel {
  id: string
  name: string
  slug: string
  blocks: ArenaBlock[]
}

export interface ArenaData {
  channels: ArenaChannel[]
}

declare module "vfile" {
  interface DataMap {
    arenaData?: ArenaData
    arenaChannel?: ArenaChannel
  }
}

const HIGHLIGHT_TRAILING = /\s*\[?\*\*\]?\s*$/

const parseLinkTitle = (text: string): { url: string; title?: string } | undefined => {
  const match = text.match(/^(https?:\/\/[^\s]+)\s*(?:(?:--|—)\s*(.+))?$/)
  if (!match) {
    return undefined
  }

  return {
    url: match[1],
    title: match[2]?.trim(),
  }
}

const stripHighlightMarker = (value: string): string => value.replace(HIGHLIGHT_TRAILING, "").trim()

const cleanParagraphForHtml = (paragraph: Paragraph): Paragraph => {
  const cleaned = clone(paragraph) as Paragraph

  while (cleaned.children.length > 0) {
    const last = cleaned.children[cleaned.children.length - 1] as PhrasingContent

    if (last.type === "text") {
      const nextValue = last.value.replace(HIGHLIGHT_TRAILING, "")
      if (nextValue.length === 0 && HIGHLIGHT_TRAILING.test(last.value)) {
        cleaned.children.pop()
        continue
      }

      if (nextValue !== last.value) {
        last.value = nextValue.trimEnd()
      }
    }

    break
  }

  return cleaned
}

const paragraphToHtml = (paragraph?: Paragraph): ElementContent | undefined => {
  if (!paragraph) return undefined

  const hastNode = toHast(cleanParagraphForHtml(paragraph), { allowDangerousHtml: true })

  if (hastNode.type === "element") {
    if (hastNode.tagName === "p") {
      return {
        type: "element",
        tagName: "span",
        properties: {},
        children: hastNode.children as ElementContent[],
      }
    }
    return hastNode as ElementContent
  }

  if (hastNode.type === "root") {
    return {
      type: "element",
      tagName: "span",
      properties: {},
      children: hastNode.children as ElementContent[],
    }
  }

  return undefined
}

const findDelimiter = (value: string): { index: number; length: number } | null => {
  const doubleDashIndex = value.indexOf("--")
  const emDashIndex = value.indexOf("—")

  if (doubleDashIndex === -1 && emDashIndex === -1) {
    return null
  }

  if (doubleDashIndex !== -1 && (emDashIndex === -1 || doubleDashIndex < emDashIndex)) {
    return { index: doubleDashIndex, length: 2 }
  }

  return { index: emDashIndex, length: 1 }
}

const buildTitleParagraph = (
  paragraph: Paragraph | undefined,
  {
    parsedTitle,
    preferLink,
    fallbackText,
  }: { parsedTitle?: string; preferLink?: Link; fallbackText: string },
): Paragraph => {
  const ensureParagraph = (node: Paragraph | undefined): Paragraph | undefined => {
    if (!node) return undefined
    return node.type === "paragraph" ? node : undefined
  }

  if (preferLink) {
    return {
      type: "paragraph",
      children: clone(preferLink.children ?? []) as PhrasingContent[],
    }
  }

  const originalParagraph = ensureParagraph(paragraph)

  if (originalParagraph) {
    if (parsedTitle) {
      const extracted: PhrasingContent[] = []
      let delimiterFound = false

      for (const child of originalParagraph.children) {
        if (!delimiterFound) {
          if (child.type === "text") {
            const info = findDelimiter(child.value)
            if (info) {
              delimiterFound = true
              const suffix = stripHighlightMarker(child.value.slice(info.index + info.length))
              if (suffix.trim().length > 0) {
                extracted.push({ type: "text", value: suffix.trimStart() } as Text)
              }
              continue
            }
          }
          continue
        }

        extracted.push(clone(child) as PhrasingContent)
      }

      if (delimiterFound) {
        if (extracted.length === 0) {
          extracted.push({ type: "text", value: stripHighlightMarker(parsedTitle) } as Text)
        }

        return {
          type: "paragraph",
          children: extracted,
        }
      }
    }

    return {
      type: "paragraph",
      children: clone(originalParagraph.children) as PhrasingContent[],
    }
  }

  const fallback = stripHighlightMarker(fallbackText)
  const markdown = wikiTextTransform(fallback)
  const mdast = fromMarkdown(markdown)
  const firstParagraph = mdast.children.find((child): child is Paragraph => child.type === "paragraph")
  if (firstParagraph) {
    return clone(firstParagraph) as Paragraph
  }

  return {
    type: "paragraph",
    children:
      fallback.length > 0 ? ([{ type: "text", value: fallback } as Text] as PhrasingContent[]) : [],
  }
}

const makeSnapshotKey = (url: string, title: string): string => {
  const hash = createHash("sha256")
  hash.update(url)
  hash.update("::")
  hash.update(title)
  return hash.digest("hex").slice(0, 32)
}

export const Arena: QuartzTransformerPlugin = () => {
  return {
    name: "Arena",
    markdownPlugins() {
      return [
        () => {
          return (tree: MdastRoot, file) => {
            const fileData = file.data as QuartzPluginData

            if (fileData.slug !== "are.na") return

            const channels: ArenaChannel[] = []
            let blockCounter = 0
            const slugger = new Slugger()

            const createBlock = (listItem: ListItem, depth = 0): ArenaBlock | null => {
              const paragraph = listItem.children.find(
                (child): child is Paragraph => child.type === "paragraph",
              )

              const textContent = paragraph ? toString(paragraph).trim() : toString(listItem).trim()
              const highlighted = HIGHLIGHT_TRAILING.test(textContent)
              const strippedContent = stripHighlightMarker(textContent)

              const firstLink = paragraph?.children.find(
                (child): child is Link => child.type === "link",
              )

              let url: string | undefined
              let title: string | undefined
              let content = strippedContent
              let parsed: ReturnType<typeof parseLinkTitle> | undefined
              let useLinkForTitle = false

              if (firstLink && depth === 0) {
                const linkText = stripHighlightMarker(toString(firstLink).trim())
                if (linkText.length > 0) {
                  content = linkText
                  title = linkText
                  useLinkForTitle = true
                }

                if (firstLink.url && /^https?:\/\//.test(firstLink.url)) {
                  url = firstLink.url
                }
              }

              if (depth === 0 && strippedContent.toLowerCase().startsWith("http")) {
                parsed = parseLinkTitle(strippedContent)
                if (parsed) {
                  url = parsed.url
                  if (parsed.title) {
                    title = parsed.title
                    content = parsed.title
                  } else {
                    content = parsed.url
                  }
                  useLinkForTitle = false
                }
              }

              if (depth > 0 && firstLink && !url && /^https?:\/\//.test(firstLink.url)) {
                url = firstLink.url
              }

              const titleParagraph = buildTitleParagraph(paragraph, {
                parsedTitle: depth === 0 ? parsed?.title : undefined,
                preferLink: depth === 0 && useLinkForTitle ? firstLink : undefined,
                fallbackText: content,
              })

              const cleanedTitleParagraph = cleanParagraphForHtml(titleParagraph)
              const titleText = toString(cleanedTitleParagraph).trim()
              const titleHtmlNode = paragraphToHtml(titleParagraph)
              const finalTitle = titleText.length > 0 ? titleText : title

              const fallbackContent = content || finalTitle || url || ""

              const blockId = `block-${blockCounter++}`
              const snapshotKey = url
                ? makeSnapshotKey(url, finalTitle ?? fallbackContent)
                : undefined

              const block: ArenaBlock = {
                id: blockId,
                content: fallbackContent,
                url,
                title: finalTitle,
                titleHtmlNode,
                snapshotKey,
                highlighted,
                htmlNode: paragraphToHtml(paragraph),
              }

              const nestedList = listItem.children.find(
                (child): child is List => child.type === "list",
              )
              if (nestedList) {
                const subItems = buildBlocks(nestedList, depth + 1)
                if (subItems.length > 0) {
                  block.subItems = subItems
                }
              }

              return block
            }

            const buildBlocks = (list: List, depth = 0): ArenaBlock[] =>
              list.children
                .map((child) =>
                  child.type === "listItem" ? createBlock(child as ListItem, depth) : null,
                )
                .filter((block): block is ArenaBlock => block !== null)

            let pendingChannel: ArenaChannel | undefined
            visit(tree, (node, _index, parent) => {
              if (parent !== tree) {
                return CONTINUE
              }

              if (node.type === "heading") {
                const depth = (node as any).depth as number | undefined
                if (depth === 2) {
                  const headingText = toString(node).trim()
                  const name =
                    headingText.length > 0 ? headingText : `Channel ${channels.length + 1}`
                  const slug = slugger.slug(name || `channel-${channels.length + 1}`)

                  const channel: ArenaChannel = {
                    id: `channel-${channels.length}`,
                    name,
                    slug,
                    blocks: [],
                  }

                  channels.push(channel)
                  pendingChannel = channel
                  return CONTINUE
                }

                if (pendingChannel) {
                  pendingChannel = undefined
                }

                return CONTINUE
              }

              if (node.type === "list" && pendingChannel) {
                pendingChannel.blocks = buildBlocks(node as List)
                pendingChannel = undefined
                return CONTINUE
              }

              return CONTINUE
            })

            fileData.arenaData = { channels }
          }
        },
      ]
    },
    htmlPlugins() {
      return []
    },
  }
}
