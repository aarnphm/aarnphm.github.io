import { CompletionSource } from "@codemirror/autocomplete"
import { emojiCompletionSource } from "./emoji"
import { mentionCompletionSource } from "./mention"
import { wikilinkCompletionSource } from "./wikilink"

export { fuzzyMatch, fuzzyMatchMultiple } from "./fuzzy"
export { frecencyStore } from "./frecency"
export type { FuzzyMatch, FrecencyEntry, CompletionCandidate, ScoredCandidate } from "./types"

export const completionSources: CompletionSource[] = [
  wikilinkCompletionSource,
  emojiCompletionSource,
  mentionCompletionSource,
]
