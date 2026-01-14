import { CompletionContext, CompletionResult } from "@codemirror/autocomplete"
import { getEmojiEntries } from "../../util/emoji"

export async function emojiCompletionSource(
  context: CompletionContext,
): Promise<CompletionResult | null> {
  const word = context.matchBefore(/:[\w\s-]*/)
  if (!word || word.from === word.to) return null

  const entries = await getEmojiEntries()
  const query = word.text.slice(1).toLowerCase()

  const matches = entries
    .filter((e) => e.name.toLowerCase().includes(query))
    .slice(0, 15)
    .map((e) => ({
      label: `:${e.name}:`,
      detail: e.emoji,
      type: "emoji",
      apply: e.emoji,
    }))

  if (matches.length === 0) return null

  return {
    from: word.from,
    options: matches,
  }
}
