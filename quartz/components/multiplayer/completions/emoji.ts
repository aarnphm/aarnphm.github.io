import { CompletionContext, CompletionResult } from '@codemirror/autocomplete'
import type { FuzzyMatch } from './types'
import { getEmojiEntries, type EmojiEntry } from '../../../util/emoji'
import { fuzzyMatch } from './fuzzy'

interface ScoredEmoji {
  entry: EmojiEntry
  match: FuzzyMatch
}

export async function emojiCompletionSource(
  context: CompletionContext,
): Promise<CompletionResult | null> {
  const word = context.matchBefore(/:[^\s:]*/)
  if (!word || word.from === word.to) return null

  const query = word.text.slice(1)
  if (query.length === 0 && !context.explicit) return null

  const entries = await getEmojiEntries()

  const scored: ScoredEmoji[] = []

  for (const entry of entries) {
    if (query.length === 0) {
      scored.push({ entry, match: { score: 0, positions: [] } })
      if (scored.length >= 15) break
    } else {
      const match = fuzzyMatch(query, entry.name)
      if (match) {
        scored.push({ entry, match })
      }
    }
  }

  if (query.length > 0) {
    scored.sort((a, b) => b.match.score - a.match.score)
  }

  const limited = scored.slice(0, 15)

  if (limited.length === 0) return null

  return {
    from: word.from,
    filter: false,
    options: limited.map(({ entry, match }) => ({
      label: `:${entry.name}:`,
      displayLabel: entry.emoji,
      detail: entry.name,
      type: 'emoji',
      apply: entry.emoji,
      boost: match.score,
    })),
  }
}
