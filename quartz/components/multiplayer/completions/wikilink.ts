import { CompletionContext, CompletionResult, Completion } from "@codemirror/autocomplete"
import { EditorView } from "@codemirror/view"
import { fuzzyMatch, fuzzyMatchMultiple } from "./fuzzy"
import { frecencyStore } from "./frecency"
import { getContentIndex } from "./data"
import type { FuzzyMatch } from "./types"
import type { ContentDetails } from "../../../plugins/emitters/contentIndex"
import { isStreamHost } from "../../scripts/util"

function extractNamespace(query: string): { prefix: string; remainder: string } {
  const slashIdx = query.lastIndexOf("/")
  if (slashIdx === -1) return { prefix: "", remainder: query }
  return {
    prefix: query.slice(0, slashIdx + 1),
    remainder: query.slice(slashIdx + 1),
  }
}

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

  return {
    inside: true,
    start: line.from + openBracketIndex + 2,
    query: textBefore.slice(openBracketIndex + 2),
    hasClosingBracket: textAfter.startsWith("]]"),
  }
}

interface ScoredItem {
  slug: string
  item: ContentDetails
  match: FuzzyMatch
  matchedField: "slug" | "title" | "alias"
  matchedAlias?: string
}

export async function wikilinkCompletionSource(
  context: CompletionContext,
): Promise<CompletionResult | null> {
  const wikiCtx = isInsideWikilink(context)
  if (!wikiCtx.inside) return null

  const data = await getContentIndex()

  const query = wikiCtx.query.trim()
  const { prefix, remainder } = extractNamespace(query)

  const scored: ScoredItem[] = []

  for (const [slug, item] of Object.entries(data)) {
    if (prefix && !slug.startsWith(prefix)) continue

    const slugToMatch = prefix ? slug.slice(prefix.length) : slug

    if (remainder === "") {
      const frecencyBoost = frecencyStore.getBoost(slug)
      scored.push({
        slug,
        item,
        match: { score: frecencyBoost, positions: [] },
        matchedField: "slug",
      })
    } else {
      const targets: string[] = [slugToMatch]
      if (item.title) targets.push(item.title)

      const best = fuzzyMatchMultiple(remainder, targets)

      let aliasMatch: { alias: string; match: FuzzyMatch } | null = null
      if (item.aliases) {
        for (const alias of item.aliases) {
          const m = fuzzyMatch(remainder, alias)
          if (m && (!aliasMatch || m.score > aliasMatch.match.score)) {
            aliasMatch = { alias, match: m }
          }
        }
      }

      let finalMatch: FuzzyMatch | null = null
      let matchedField: "slug" | "title" | "alias" = "slug"
      let matchedAlias: string | undefined

      if (best && (!aliasMatch || best.match.score >= aliasMatch.match.score)) {
        finalMatch = best.match
        matchedField = best.target === slugToMatch ? "slug" : "title"
      } else if (aliasMatch) {
        finalMatch = aliasMatch.match
        matchedField = "alias"
        matchedAlias = aliasMatch.alias
      }

      if (finalMatch) {
        const frecencyBoost = frecencyStore.getBoost(slug)
        scored.push({
          slug,
          item,
          match: { score: finalMatch.score + frecencyBoost, positions: finalMatch.positions },
          matchedField,
          matchedAlias,
        })
      }
    }
  }

  scored.sort((a, b) => b.match.score - a.match.score)

  const limited = scored.slice(0, 50)

  if (limited.length === 0) return null

  const baseUrl = isStreamHost() ? "https://aarnphm.xyz" : ""
  const closingSuffix = wikiCtx.hasClosingBracket ? "" : "]]"

  const completions: Completion[] = limited.map(
    ({ slug, item, match, matchedField, matchedAlias }) => {
      const label = item.title || slug.split("/").pop() || slug

      let detail = slug
      if (matchedField === "alias" && matchedAlias) {
        detail = `${slug} (alias: ${matchedAlias})`
      }

      const insertText =
        matchedField === "alias" && matchedAlias
          ? `${baseUrl}${slug}|${matchedAlias}${closingSuffix}`
          : `${baseUrl}${slug}${closingSuffix}`

      return {
        label,
        detail,
        type: "page",
        boost: match.score,
        apply: (view: EditorView, _completion: Completion, from: number, to: number) => {
          frecencyStore.recordAccess(slug)
          view.dispatch({
            changes: { from, to, insert: insertText },
            selection: { anchor: from + insertText.length },
          })
        },
      }
    },
  )

  return {
    from: wikiCtx.start,
    options: completions,
    filter: false,
  }
}
