import { CompletionContext, CompletionResult } from "@codemirror/autocomplete"
import { queryMentionUsers, isDBAvailable } from "./mention-cache"

export async function mentionCompletionSource(
  context: CompletionContext,
): Promise<CompletionResult | null> {
  if (!isDBAvailable()) return null

  const word = context.matchBefore(/@[\w-]*/)
  if (!word || (word.from === word.to && !context.explicit)) return null

  const query = word.text.slice(1)
  const users = await queryMentionUsers(query)
  if (users.length === 0) return null

  return {
    from: word.from,
    options: users.map((u) => ({
      label: `@${u.login}`,
      detail: u.displayName && u.displayName !== u.login ? u.displayName : undefined,
      type: "mention",
      apply: `@${u.login} `,
    })),
  }
}
