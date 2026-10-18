import { type Completion, CompletionContext, CompletionResult } from '@codemirror/autocomplete'
import { getEmojiEntries, type EmojiEntry } from '../../../util/emoji'
import { fuzzyMatch } from './fuzzy'

const EMOJI_COMPLETION_LIMIT = 15

interface IndexedEmoji {
  name: string
  option: Completion
}

interface ScoredEmoji {
  option: Completion
  score: number
}

interface EmojiCompletionIndex {
  entries: EmojiEntry[]
  items: IndexedEmoji[]
  emptyOptions: Completion[]
}

let cachedIndex: EmojiCompletionIndex | null = null

function getEmojiCompletionIndex(entries: EmojiEntry[]): EmojiCompletionIndex {
  if (cachedIndex && cachedIndex.entries === entries) return cachedIndex
  const items: IndexedEmoji[] = entries.map(entry => ({
    name: entry.name,
    option: {
      label: `:${entry.name}:`,
      displayLabel: entry.emoji,
      detail: entry.name,
      type: 'emoji',
      apply: entry.emoji,
      boost: 0,
    },
  }))
  cachedIndex = {
    entries,
    items,
    emptyOptions: items.slice(0, EMOJI_COMPLETION_LIMIT).map(item => item.option),
  }
  return cachedIndex
}

export async function emojiCompletionSource(
  context: CompletionContext,
): Promise<CompletionResult | null> {
  const word = context.matchBefore(/:[^\s:]*/)
  if (!word || word.from === word.to) return null

  const query = word.text.slice(1)
  if (query.length === 0 && !context.explicit) return null

  const index = getEmojiCompletionIndex(await getEmojiEntries())

  if (query.length === 0) {
    if (index.emptyOptions.length === 0) return null
    return { from: word.from, filter: false, options: index.emptyOptions }
  }

  const top: ScoredEmoji[] = []
  let minScore = -Infinity

  for (let i = 0; i < index.items.length; i++) {
    const item = index.items[i]!
    const match = fuzzyMatch(query, item.name, minScore)
    if (!match) continue
    const score = match.score
    let pos = top.length
    while (pos > 0 && top[pos - 1]!.score < score) {
      pos -= 1
    }
    if (top.length < EMOJI_COMPLETION_LIMIT) {
      top.splice(pos, 0, { option: item.option, score })
    } else {
      for (let j = EMOJI_COMPLETION_LIMIT - 1; j > pos; j--) {
        top[j] = top[j - 1]!
      }
      top[pos] = { option: item.option, score }
    }
    if (top.length === EMOJI_COMPLETION_LIMIT) {
      minScore = top[EMOJI_COMPLETION_LIMIT - 1]!.score
    }
  }

  if (top.length === 0) return null

  return {
    from: word.from,
    filter: false,
    options: top.map(({ option, score }) => ({ ...option, boost: score })),
  }
}
