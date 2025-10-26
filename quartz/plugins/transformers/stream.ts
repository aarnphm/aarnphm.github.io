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

    const rawText = getFirstTextContent(child).trim()
    const nested = extractNestedList(child)

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
  const metaMatch = label.match(/^\[meta\](?:\s*[:\-–—])?$/)
  if (!metaMatch) return null

  const metaList = extractNestedList(firstItem)
  if (!metaList || metaList.children.length === 0) return {}

  const yamlLines: string[] = []
  appendListToYaml(metaList, 0, yamlLines)
  const yamlSource = yamlLines.join("\n")
  if (yamlSource.trim().length === 0) return {}

  try {
    const parsed = yaml.load(yamlSource)
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return {}
    }
    return parsed as StreamMetadata
  } catch {
    return {}
  }
}

const parseDateValue = (value: unknown): { date?: string; timestamp?: number } => {
  if (typeof value !== "string") return {}
  let trimmed = value.trim()
  if (trimmed.length === 0) return {}

  // strip timezone abbreviations like PST, PDT, EST, etc
  trimmed = trimmed.replace(/\s+(PST|PDT|EST|EDT|CST|CDT|MST|MDT)$/i, "")

  // try parsing with Date.parse first
  let timestamp = Date.parse(trimmed)

  // if that fails, try adding T separator for ISO-like formats
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

              // skip whitespace text nodes and doctype
              if (node.type === "text" && (!node.value || node.value.trim().length === 0)) {
                continue
              }
              if (node.type === "doctype") {
                continue
              }

              // h2 starts new entry with title
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

              // hr ends current entry
              if (isHr(node)) {
                if (currentEntry) {
                  entries.push(currentEntry)
                  currentEntry = null
                }
                indicesToRemove.add(i)
                continue
              }

              // check if ul is metadata list
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

              // everything else is content (if it's an element)
              if (!isElement(node)) continue

              if (!currentEntry) {
                currentEntry = {
                  metadata: {},
                  content: [],
                }
              }
              currentEntry.content.push(node)
            }

            // push final entry if exists
            if (currentEntry) {
              entries.push(currentEntry)
            }

            // remove meta lists and separators from tree
            const children: RootContent[] = []
            for (let i = 0; i < bodyChildren.length; i++) {
              if (!indicesToRemove.has(i)) {
                children.push(bodyChildren[i])
              }
            }
            tree.children = children

            // build final StreamData with IDs and sorted by timestamp
            const streamEntries: StreamEntry[] = entries.map((entry, idx) => {
              const { date, timestamp } = parseDateValue(entry.metadata.date)

              return {
                id: `stream-entry-${idx}`,
                title: entry.title ? toString(entry.title) : undefined,
                metadata: entry.metadata,
                content: entry.content,
                date,
                timestamp,
              }
            })

            // sort by timestamp (newest first)
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
