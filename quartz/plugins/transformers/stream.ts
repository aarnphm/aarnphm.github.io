import { QuartzTransformerPlugin } from "../types"
import type { Element, ElementContent, Root as HastRoot, RootContent } from "hast"
import { toString } from "hast-util-to-string"
import yaml from "js-yaml"

export type StreamMetadata = Record<string, unknown>

export interface StreamEntry {
  id: string
  title?: string
  metadata: StreamMetadata
  content: ElementContent[]
  date?: string
  timestamp?: number
  importance?: number
}

export interface StreamData {
  entries: StreamEntry[]
}

declare module "vfile" {
  interface DataMap {
    streamData?: StreamData
  }
}

const isElement = (node: RootContent | ElementContent): node is Element => node.type === "element"

const isH2 = (node: RootContent | ElementContent): node is Element =>
  isElement(node) && node.tagName === "h2"

const isUl = (node: RootContent | ElementContent): node is Element =>
  isElement(node) && node.tagName === "ul"

const isHr = (node: RootContent | ElementContent): node is Element =>
  isElement(node) && node.tagName === "hr"

const isLi = (node: RootContent | ElementContent): node is Element =>
  isElement(node) && node.tagName === "li"

const getFirstTextContent = (node: Element): string => {
  for (const child of node.children) {
    if (child.type === "text") {
      return child.value
    }
    if (isElement(child)) {
      const text = getFirstTextContent(child)
      if (text) return text
    }
  }
  return ""
}

const extractNestedList = (li: Element): Element | null => {
  for (const child of li.children) {
    if (isUl(child)) return child
  }
  return null
}

const appendListToYaml = (list: Element, indent: number, lines: string[]): void => {
  const indentStr = "  ".repeat(indent)

  for (const child of list.children) {
    if (!isLi(child)) continue

    const nested = extractNestedList(child)
    let rawText = ""
    for (const ch of child.children as ElementContent[]) {
      if (isElement(ch) && ch.tagName === "ul") continue
      rawText += toString(ch)
    }
    rawText = rawText.trim()

    if (nested) {
      if (rawText.length > 0) {
        lines.push(`${indentStr}${rawText}`)
      } else {
        lines.push(`${indentStr}-`)
      }
      appendListToYaml(nested, indent + 1, lines)
      continue
    }

    if (rawText.length === 0) continue
    const normalized = rawText.replace(/^\-+\s*/, "").trim()

    if (normalized.includes(":")) {
      lines.push(`${indentStr}${normalized}`)
    } else {
      lines.push(`${indentStr}- ${normalized}`)
    }
  }
}

const extractMetadata = (list: Element): StreamMetadata | null => {
  if (list.children.length === 0) return null

  const firstItem = list.children.find(isLi)
  if (!firstItem) return null

  const label = getFirstTextContent(firstItem).trim().toLowerCase()
  if (!/^\[meta\](?:\s*[:\-–—])?$/.test(label)) return null

  const metaList = extractNestedList(firstItem)
  if (!metaList || metaList.children.length === 0) return {}

  const metadata: StreamMetadata = {}

  for (const item of metaList.children) {
    if (!isLi(item)) continue
    const raw = getFirstTextContent(item).trim()
    const sublist = extractNestedList(item)

    let keySource: string | undefined
    let value: string | undefined

    if (raw.length > 0) {
      const delimiterIndex = raw.indexOf(":")
      if (delimiterIndex !== -1) {
        keySource = raw.slice(0, delimiterIndex).trim()
        value = raw.slice(delimiterIndex + 1).trim()
      } else if (sublist) {
        keySource = raw.trim().toLowerCase()
        value = ""
      }
    } else if (sublist) {
      value = ""
    }

    if (!keySource) continue

    const normalizedKey = keySource.toLowerCase().replace(/\s+/g, "_")

    if (sublist && (!value || value.length === 0)) {
      const yamlLines: string[] = []
      appendListToYaml(sublist, 0, yamlLines)
      const yamlSource = yamlLines.join("\n")

      if (yamlSource.trim().length > 0) {
        const parsed = yaml.load(yamlSource)
        if (parsed && typeof parsed === "object") {
          metadata[normalizedKey] = parsed
          continue
        }
      }
    }

    if (!value || value.length === 0) continue
    metadata[normalizedKey] = value
  }

  return Object.keys(metadata).length > 0 ? metadata : {}
}

const parseDateValue = (value: unknown): { date?: string; timestamp?: number } => {
  if (typeof value !== "string") return {}
  let trimmed = value.trim()
  if (trimmed.length === 0) return {}

  trimmed = trimmed.replace(/\s+(PST|PDT|EST|EDT|CST|CDT|MST|MDT)$/i, "")

  let timestamp = Date.parse(trimmed)

  if (Number.isNaN(timestamp) && /^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}/.test(trimmed)) {
    const isoFormat = trimmed.replace(/^(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}.*)$/, "$1T$2")
    timestamp = Date.parse(isoFormat)
  }

  if (Number.isNaN(timestamp)) return {}

  const date = new Date(timestamp).toISOString()
  return { date, timestamp }
}

interface ParsedEntry {
  title?: ElementContent
  metadata: StreamMetadata
  content: ElementContent[]
}

export const Stream: QuartzTransformerPlugin = () => {
  return {
    name: "Stream",
    htmlPlugins() {
      return [
        () => {
          return (tree: HastRoot, file) => {
            if (file.data.slug !== "stream") return

            // at htmlPlugins stage, content is direct children of root (no body wrapper yet)
            const bodyChildren = tree.children.filter(
              (child) => child.type !== "doctype",
            ) as RootContent[]

            if (bodyChildren.length === 0) return

            const entries: ParsedEntry[] = []
            let currentEntry: ParsedEntry | null = null
            const indicesToRemove = new Set<number>()

            for (let i = 0; i < bodyChildren.length; i++) {
              const node = bodyChildren[i]

              if (node.type === "text" && (!node.value || node.value.trim().length === 0)) {
                continue
              }

              if (isH2(node)) {
                if (currentEntry) {
                  entries.push(currentEntry)
                }
                currentEntry = {
                  title: node,
                  metadata: {},
                  content: [],
                }
                continue
              }

              if (isHr(node)) {
                if (currentEntry) {
                  entries.push(currentEntry)
                  currentEntry = null
                }
                indicesToRemove.add(i)
                continue
              }

              if (isUl(node)) {
                const metadata = extractMetadata(node)
                if (metadata !== null) {
                  if (!currentEntry) {
                    currentEntry = {
                      metadata: {},
                      content: [],
                    }
                  }
                  currentEntry.metadata = metadata
                  indicesToRemove.add(i)
                  continue
                }
              }

              if (!isElement(node)) continue

              if (!currentEntry) {
                currentEntry = {
                  metadata: {},
                  content: [],
                }
              }
              currentEntry.content.push(node)
            }

            if (currentEntry) {
              entries.push(currentEntry)
            }

            tree.children = bodyChildren.filter((_, index) => !indicesToRemove.has(index))

            const streamEntries: StreamEntry[] = entries.map((entry, idx) => {
              const entryId = `stream-entry-${idx}`
              for (const [contentIdx, contentNode] of entry.content.entries()) {
                if (!isElement(contentNode)) continue
                const data = (contentNode.data ??= {} as any) as Record<string, unknown>
                data.streamEntryId = entryId
                data.streamEntryContentIndex = contentIdx
              }

              const { date, timestamp } = parseDateValue(entry.metadata.date)

              let importance: number | undefined
              const importanceValue = entry.metadata.importance
              if (importanceValue !== undefined) {
                const parsed =
                  typeof importanceValue === "number"
                    ? importanceValue
                    : Number.parseFloat(String(importanceValue))
                if (!Number.isNaN(parsed) && Number.isFinite(parsed)) {
                  importance = parsed
                }
              }

              const cleanMetadata = { ...entry.metadata }
              delete cleanMetadata.date
              delete cleanMetadata.importance

              return {
                id: entryId,
                title: entry.title ? toString(entry.title) : undefined,
                metadata: cleanMetadata,
                content: entry.content,
                date,
                timestamp,
                importance,
              }
            })

            streamEntries.sort((a, b) => {
              if (a.timestamp && b.timestamp) return b.timestamp - a.timestamp
              if (a.timestamp) return -1
              if (b.timestamp) return 1
              return 0
            })

            file.data.streamData = { entries: streamEntries }
          }
        },
      ]
    },
  }
}
