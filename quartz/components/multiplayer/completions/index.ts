import { CompletionSource } from "@codemirror/autocomplete"
import { wikilinkCompletionSource } from "./wikilink"
import { emojiCompletionSource } from "./emoji"
import { mentionCompletionSource } from "./mention"

export { fuzzyMatch, fuzzyMatchMultiple } from "./fuzzy"
export { frecencyStore } from "./frecency"
export type { FuzzyMatch, FrecencyEntry, CompletionCandidate, ScoredCandidate } from "./types"

export const completionSources: CompletionSource[] = [
  wikilinkCompletionSource,
  emojiCompletionSource,
  mentionCompletionSource,
]
