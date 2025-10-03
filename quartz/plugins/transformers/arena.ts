import { QuartzTransformerPlugin } from "../types"
import {
  Root as MdastRoot,
  List,
  ListItem,
  Paragraph,
  PhrasingContent,
  Link,
  Text,
  Content as MdastContent,
} from "mdast"
import { Element, ElementContent } from "hast"
import { QuartzPluginData } from "../vfile"
import { toString } from "mdast-util-to-string"
import { toHast } from "mdast-util-to-hast"
import Slugger from "github-slugger"
import { clone } from "../../util/clone"
import { visit, CONTINUE } from "unist-util-visit"
import { wikiTextTransform, wikilinkRegex, externalLinkRegex } from "./ofm"
import { createHash } from "crypto"
import { fromMarkdown } from "mdast-util-from-markdown"
import { fetchTwitterEmbed, twitterUrlRegex } from "./twitter"
import { splitAnchor } from "../../util/path"
import { findAndReplace as mdastFindReplace, ReplaceFunction } from "mdast-util-find-and-replace"

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
  embedHtml?: string
  metadata?: Record<string, string>
}

export interface ArenaChannel {
  id: string
  name: string
  slug: string
  blocks: ArenaBlock[]
  titleHtmlNode?: ElementContent
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

// Convert wikilinks inside a paragraph into mdast link nodes, mirroring OFM behavior for inline links
const convertWikilinksInParagraph = (paragraph: Paragraph): Paragraph => {
  const node = clone(paragraph) as Paragraph
  const replacements: [RegExp, string | ReplaceFunction][] = []

  replacements.push([
    wikilinkRegex,
    (value: string, ...capture: string[]) => {
      // ignore embeds inside title fallback
      if (value.startsWith("!")) return false

      let [rawFp, rawHeader, rawAlias] = capture
      const fp = rawFp?.trim() ?? ""
      const header = rawHeader?.trim() ?? ""
      const alias: string | undefined = rawAlias?.slice(1).trim()

      if (fp.match(externalLinkRegex)) {
        return {
          type: "link",
          url: fp,
          children: [{ type: "text", value: alias ?? fp }],
        } satisfies Link
      }

      const [_f, anchor] = splitAnchor(`${fp}${header}`)
      const url = `${fp}${anchor}`
      return {
        type: "link",
        url,
        children: [{ type: "text", value: alias ?? fp }],
      } satisfies Link
    },
  ])

  mdastFindReplace(node, replacements)
  return node
}

// Build a combined inline HAST fragment from the list item paragraph and any inline siblings (excluding nested lists)
const listItemContentToInlineHtml = (listItem: ListItem): ElementContent | undefined => {
  const inlineChildren: MdastContent[] = []

  for (const child of listItem.children) {
    if (child.type === "list") continue
    if (child.type === "paragraph") {
      inlineChildren.push(cleanParagraphForHtml(child))
    } else {
      inlineChildren.push(child as MdastContent)
    }
  }

  if (inlineChildren.length === 0) return undefined

  const children: ElementContent[] = []
  for (const child of inlineChildren) {
    const hnode = toHast(child, { allowDangerousHtml: true })
    if (hnode.type === "element") {
      if (hnode.tagName === "p") {
        children.push(...(hnode.children as ElementContent[]))
      } else {
        children.push(hnode as unknown as ElementContent)
      }
    } else if (hnode.type === "root") {
      children.push(...(hnode.children as ElementContent[]))
    }
  }

  return {
    type: "element",
    tagName: "span",
    properties: {},
    children,
  }
}

type NodeWithData = {
  data?: {
    hProperties?: Record<string, unknown>
  }
}

const setDataAttribute = <T extends NodeWithData>(node: T, key: string, value: string): void => {
  if (!node.data) {
    node.data = {}
  }
  if (!node.data.hProperties) {
    node.data.hProperties = {}
  }
  node.data.hProperties[key] = value
}

const cloneElementContent = <T extends ElementContent>(node: T): T => {
  // structuredClone preserves prototypes and avoids JSON serialization issues
  return typeof structuredClone === "function"
    ? structuredClone(node)
    : (JSON.parse(JSON.stringify(node)) as T)
}

const elementContainsAnchor = (node: ElementContent): boolean => {
  if (node.type !== "element") {
    return false
  }

  if (node.tagName === "a") {
    return true
  }

  if (!("children" in node) || !node.children) {
    return false
  }

  return (node.children as ElementContent[]).some((child) => elementContainsAnchor(child))
}

const makeSnapshotKey = (url: string, title: string): string => {
  const hash = createHash("sha256")
  hash.update(url)
  hash.update("::")
  hash.update(title)
  return hash.digest("hex").slice(0, 32)
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
  const firstParagraph = mdast.children.find(
    (child): child is Paragraph => child.type === "paragraph",
  )
  if (firstParagraph) {
    return convertWikilinksInParagraph(firstParagraph)
  }

  return {
    type: "paragraph",
    children:
      fallback.length > 0 ? ([{ type: "text", value: fallback } as Text] as PhrasingContent[]) : [],
  }
}

export const Arena: QuartzTransformerPlugin = () => {
  return {
    name: "Arena",
    markdownPlugins(ctx) {
      const localeConfig = ctx.cfg.configuration.locale ?? "en"
      const locale = localeConfig.split("-")[0] ?? "en"
      return [
        () => {
          return async (tree: MdastRoot, file) => {
            const fileData = file.data as QuartzPluginData

            if (fileData.slug !== "are.na") return

            const channels: ArenaChannel[] = []
            let blockCounter = 0
            const slugger = new Slugger()
            const embedPromises: Promise<void>[] = []

            const tryExtractMetadata = (list: List): Record<string, string> | undefined => {
              if (!list.children || list.children.length === 0) {
                return undefined
              }

              const firstChild = list.children[0]
              if (firstChild?.type !== "listItem") {
                return undefined
              }

              const metaParagraph = firstChild.children.find(
                (child): child is Paragraph => child.type === "paragraph",
              )

              const metaHeading = metaParagraph ? toString(metaParagraph).trim().toLowerCase() : ""
              if (!metaHeading.startsWith("_meta")) {
                return undefined
              }

              const metaList = firstChild.children.find(
                (child): child is List => child.type === "list",
              )
              if (!metaList || metaList.children.length === 0) {
                // remove the _meta marker so it doesn't render downstream
                list.children.shift()
                return undefined
              }

              const metadata: Record<string, string> = {}
              for (const item of metaList.children) {
                if (item.type !== "listItem") continue
                const itemParagraph = item.children.find(
                  (child): child is Paragraph => child.type === "paragraph",
                )
                const raw = itemParagraph ? toString(itemParagraph).trim() : toString(item).trim()
                if (raw.length === 0) continue
                const delimiterIndex = raw.indexOf(":")
                if (delimiterIndex === -1) continue
                const key = raw.slice(0, delimiterIndex).trim().toLowerCase().replace(/\s+/g, "_")
                const value = raw.slice(delimiterIndex + 1).trim()
                if (key.length === 0 || value.length === 0) continue
                metadata[key] = value
              }

              list.children.shift()

              return Object.keys(metadata).length > 0 ? metadata : undefined
            }

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
              setDataAttribute(listItem as NodeWithData, "data-arena-block-id", blockId)
              if (paragraph) {
                setDataAttribute(paragraph as NodeWithData, "data-arena-block-paragraph", blockId)
              }

              const block: ArenaBlock = {
                id: blockId,
                content: fallbackContent,
                url,
                title: finalTitle,
                titleHtmlNode,
                snapshotKey,
                highlighted,
                htmlNode: listItemContentToInlineHtml(listItem),
              }

              const nestedList = listItem.children.find(
                (child): child is List => child.type === "list",
              )
              if (nestedList) {
                const metadata = tryExtractMetadata(nestedList)
                if (metadata) {
                  block.metadata = metadata
                }
              }
              if (nestedList) {
                const subItems = buildBlocks(nestedList, depth + 1)
                if (subItems.length > 0) {
                  block.subItems = subItems
                }
              }

              if (block.url && twitterUrlRegex.test(block.url)) {
                embedPromises.push(
                  fetchTwitterEmbed(block.url, locale).then((html) => {
                    block.embedHtml = html
                  }),
                )
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

                  const channelId = `channel-${channels.length}`
                  const channel: ArenaChannel = {
                    id: channelId,
                    name,
                    slug,
                    blocks: [],
                  }

                  setDataAttribute(node as NodeWithData, "data-arena-channel-id", channelId)

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

            if (embedPromises.length > 0) {
              await Promise.all(embedPromises)
            }

            fileData.arenaData = { channels }
          }
        },
      ]
    },
    htmlPlugins() {
      return [
        () => {
          return (tree, file) => {
            const fileData = file.data as QuartzPluginData
            if (fileData.slug !== "are.na") return

            const arenaData = fileData.arenaData
            if (!arenaData) return

            const blocksById = new Map<string, ArenaBlock>()
            const channelById = new Map<string, ArenaChannel>()
            const registerBlocks = (blocks: ArenaBlock[]) => {
              for (const block of blocks) {
                blocksById.set(block.id, block)
                if (block.subItems) {
                  registerBlocks(block.subItems)
                }
              }
            }

            for (const channel of arenaData.channels) {
              channelById.set(channel.id, channel)
              registerBlocks(channel.blocks)
            }

            visit(tree, "element", (node: Element, _index, parent) => {
              if (!node.properties) return

              const channelId = node.properties["data-arena-channel-id"]
              if (typeof channelId === "string") {
                const channel = channelById.get(channelId)
                if (channel) {
                  const headingClone = cloneElementContent(node)
                  if (headingClone.type === "element") {
                    if (elementContainsAnchor(headingClone as ElementContent)) {
                      const spanNode: Element = {
                        type: "element",
                        tagName: "span",
                        properties: {},
                        children: (headingClone.children ?? []).map((child) =>
                          cloneElementContent(child as ElementContent),
                        ),
                      }
                      channel.titleHtmlNode = spanNode
                    }
                  }
                }

                delete node.properties["data-arena-channel-id"]
              }

              const blockOwner = node.properties["data-arena-block-id"]
              if (typeof blockOwner === "string") {
                delete node.properties["data-arena-block-id"]
              }

              const blockId = node.properties["data-arena-block-paragraph"]
              if (typeof blockId !== "string") {
                return
              }

              const block = blocksById.get(blockId)
              if (!block) {
                delete node.properties["data-arena-block-paragraph"]
                return
              }

              // Build an aggregated inline content node from the list item containing this paragraph
              let aggregated: ElementContent | undefined
              if (parent && parent.type === "element" && parent.tagName === "li") {
                const li = parent
                // Collect all children of the li except nested lists
                const collected: ElementContent[] = []
                for (const ch of (li.children ?? []) as ElementContent[]) {
                  const el = cloneElementContent(ch)
                  if (el.type === "element" && el.tagName === "ul") continue
                  if (el.type === "element") {
                    if (el.properties) {
                      delete el.properties["data-arena-block-paragraph"]
                      delete el.properties["data-arena-block-id"]
                    }
                  }
                  collected.push(el)
                }

                aggregated = {
                  type: "element",
                  tagName: "span",
                  properties: {},
                  children: collected,
                }
              }

              // Title is the cleaned paragraph itself (inline)
              const inlineTitleClone = cloneElementContent(node)
              if (inlineTitleClone.type === "element") {
                if (inlineTitleClone.properties) {
                  delete inlineTitleClone.properties["data-arena-block-paragraph"]
                  delete inlineTitleClone.properties["data-arena-block-id"]
                }
                if (inlineTitleClone.tagName === "p") inlineTitleClone.tagName = "span"
              }

              block.titleHtmlNode = inlineTitleClone as ElementContent
              if (aggregated) {
                block.htmlNode = aggregated as ElementContent
              } else {
                // fallback to paragraph clone if aggregation failed
                block.htmlNode = inlineTitleClone as ElementContent
              }

              delete node.properties["data-arena-block-paragraph"]
            })
          }
        },
      ]
    },
  }
}
