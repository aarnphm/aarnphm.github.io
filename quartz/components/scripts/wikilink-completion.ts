import { CompletionContext, CompletionResult, Completion } from "@codemirror/autocomplete"
import { querySearchIndex } from "./search-index"
import { isStreamHost } from "./util"

function isInsideWikilink(context: CompletionContext): {
  inside: boolean
  start: number
  query: string
  hasClosingBracket: boolean
} {
  const { state, pos } = context
  const line = state.doc.lineAt(pos)
  const textBefore = line.text.slice(0, pos - line.from)
  const textAfter = line.text.slice(pos - line.from)

  const openBracketIndex = textBefore.lastIndexOf("[[")
  const closeBracketIndex = textBefore.lastIndexOf("]]")

  if (openBracketIndex === -1 || closeBracketIndex > openBracketIndex) {
    return { inside: false, start: 0, query: "", hasClosingBracket: false }
  }

  const query = textBefore.slice(openBracketIndex + 2)
  const hasClosingBracket = textAfter.startsWith("]]")

  return {
    inside: true,
    start: line.from + openBracketIndex + 2,
    query,
    hasClosingBracket,
  }
}

export async function wikilinkCompletionSource(
  context: CompletionContext,
): Promise<CompletionResult | null> {
  const wikilinkContext = isInsideWikilink(context)

  if (!wikilinkContext.inside) return null

  const query = wikilinkContext.query.trim()
  const results = await querySearchIndex(query || "", 100)

  const completions: Completion[] = results.flatMap((item) => {
    const baseSlug = isStreamHost() ? `https://aarnphm.xyz${item.slug}` : item.slug

    const closingSuffix = wikilinkContext.hasClosingBracket ? "" : "]]"

    const slugMatches = query === "" || item.slug.toLowerCase().includes(query.toLowerCase())
    const titleMatches =
      query === "" || (item.title && item.title.toLowerCase().includes(query.toLowerCase()))
    const nameMatches = query === "" || item.name.toLowerCase().includes(query.toLowerCase())

    if (!slugMatches && !titleMatches && !nameMatches) {
      return []
    }

    const mainCompletion: Completion = {
      label: item.title || item.name,
      detail: item.slug,
      type: "page",
      apply: `${baseSlug}${closingSuffix}`,
      boost: slugMatches && item.slug.toLowerCase().startsWith(query.toLowerCase()) ? 2 : 0,
    }

    const aliasCompletions: Completion[] = item.aliases
      .filter((alias) => query === "" || alias.toLowerCase().includes(query.toLowerCase()))
      .map((alias) => ({
        label: alias,
        detail: `${item.slug} (alias)`,
        type: "page",
        apply: `${baseSlug}|${alias}${closingSuffix}`,
        boost: 1,
      }))

    return [mainCompletion, ...aliasCompletions]
  })

  if (completions.length === 0) return null

  return {
    from: wikilinkContext.start,
    options: completions,
    validFor: /^[^\[\]]*$/,
  }
}
