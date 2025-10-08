import path from "path"
import isAbsoluteUrl from "is-absolute-url"

import {
  FilePath,
  FullSlug,
  getFileExtension,
  slugifyFilePath,
  splitAnchor,
  stripSlashes,
} from "./path"

export const WIKILINK_PATTERN = /!?\[\[([^\[\]\|#\\]+)?(#+[^\[\]\|#\\]+)?(\\?\|[^\[\]#]*)?\]\]/

/**
 * Create a fresh wikilink regex. Consumers should prefer using this helper instead of sharing
 * a singleton RegExp to avoid `lastIndex` state across different callers.
 */
export function createWikilinkRegex(flags: string = "g"): RegExp {
  const normalizedFlags = Array.from(new Set((flags + "g").split(""))).join("")
  return new RegExp(WIKILINK_PATTERN.source, normalizedFlags)
}

const singleWikilinkRegex = new RegExp(`^${WIKILINK_PATTERN.source}$`, "i")

export interface ParsedWikilink {
  raw: string
  target: string
  anchor?: string
  alias?: string
  embed: boolean
}

export function parseWikilink(raw: string): ParsedWikilink | null {
  const trimmed = raw.trim()
  const match = trimmed.match(singleWikilinkRegex)
  if (!match) {
    return null
  }

  const isEmbed = trimmed.startsWith("!")

  const [, rawFp, rawHeader, rawAlias] = match

  const [_target, anchor] = splitAnchor(`${rawFp ?? ""}${rawHeader ?? ""}`)
  const blockRef = rawHeader?.startsWith("#^") ? "^" : ""
  const displayAnchor = anchor ? `#${blockRef}${anchor.trim().replace(/^#+/, "")}` : undefined
  const alias = rawAlias ? rawAlias.replace(/^\\?\|/, "") : undefined

  return {
    raw: trimmed,
    target: rawFp ?? "",
    anchor: displayAnchor,
    alias,
    embed: isEmbed,
  }
}

export function extractWikilinks(text: string): ParsedWikilink[] {
  const links: ParsedWikilink[] = []
  const regex = createWikilinkRegex()
  let match: RegExpExecArray | null

  while ((match = regex.exec(text)) !== null) {
    const parsed = parseWikilink(match[0])
    if (parsed) {
      links.push(parsed)
    }
  }

  return links
}

export interface ResolvedWikilinkTarget {
  slug: FullSlug
  anchor?: string
}

function ensureFilePath(target: string): FilePath {
  if (target.length === 0) {
    return "index.md" as FilePath
  }

  if (target.endsWith("/")) {
    return `${target}index.md` as FilePath
  }

  const ext = getFileExtension(target as FilePath)
  if (ext) {
    return target as FilePath
  }

  return `${target}.md` as FilePath
}

export function resolveWikilinkTarget(
  link: ParsedWikilink,
  currentSlug: FullSlug,
): ResolvedWikilinkTarget | null {
  const baseSlug = stripSlashes(currentSlug)

  if (link.target && isAbsoluteUrl(link.target)) {
    return null
  }

  const normalizedTarget = link.target?.replace(/\\/g, "/") ?? ""

  let combined: string

  if (normalizedTarget.startsWith("/")) {
    combined = normalizedTarget.slice(1)
  } else if (normalizedTarget.length === 0) {
    combined = baseSlug
  } else {
    const dir = path.posix.dirname(baseSlug)
    const baseDir = dir === "." ? "" : dir
    combined = path.posix.join(baseDir, normalizedTarget)
  }

  let normalized = path.posix.normalize(combined)
  while (normalized.startsWith("../")) {
    normalized = normalized.slice(3)
  }
  if (normalized.startsWith("./")) {
    normalized = normalized.slice(2)
  }

  const filePath = ensureFilePath(normalized)
  const slug = slugifyFilePath(stripSlashes(filePath) as FilePath)

  return {
    slug,
    anchor: link.anchor,
  }
}

export function stripWikilinkFormatting(text: string): string {
  if (!text) {
    return text
  }

  return text.replace(createWikilinkRegex(), (value) => {
    const parsed = parseWikilink(value)
    if (!parsed) return value
    return parsed.alias ?? parsed.target
  })
}
