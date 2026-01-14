import katex from "katex"
import type { KatexOptions } from "katex"
import { extractWikilinks, resolveWikilinkTarget } from "./wikilinks"
import type { FullSlug } from "./path"

const defaultKatexOptions: Omit<KatexOptions, "output"> = {
  strict: true,
  throwOnError: true,
}

export function renderLatexInString(
  text: string,
  options: Omit<KatexOptions, "output"> = defaultKatexOptions,
): string {
  let result = text

  const blockMathRegex = /\$\$([\s\S]*?)\$\$/g
  result = result.replace(blockMathRegex, (match, math) => {
    try {
      return katex.renderToString(math.trim(), {
        ...options,
        displayMode: true,
      })
    } catch {
      return match
    }
  })

  const inlineMathRegex = /(?<!\$)\$([^$\n]+?)\$(?!\$)/g
  result = result.replace(inlineMathRegex, (match, math) => {
    try {
      return katex.renderToString(math.trim(), {
        ...options,
        displayMode: false,
      })
    } catch {
      return match
    }
  })

  return result
}

export function processWikilinksToHtml(text: string, currentSlug: FullSlug): string {
  const wikilinks = extractWikilinks(text)
  let result = text

  for (const link of wikilinks) {
    const resolved = resolveWikilinkTarget(link, currentSlug)
    if (resolved) {
      const displayText = link.alias || link.target
      const href = `/${resolved.slug}${resolved.anchor || ""}`
      const htmlLink = `<a href="${href}" class="internal">${displayText}</a>`
      result = result.replace(link.raw, htmlLink)
    }
  }

  return result
}
