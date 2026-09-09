# Deck format

The parser in `quartz/util/flashcards.ts` defines the format. Card faces use Garden Markdown; rendering lives in `quartz/plugins/transformers/flashcards.ts` and `quartz/plugins/emitters/flashcardsPage.tsx`. Scheduling lives in `worker/flashcards.ts`. Read those owners only when the task concerns their behavior.

```text
Q: What does $A \subseteq B$ mean?
A: Every element of $A$ is also an element of $B$.
---
C: De Morgan's law: $A \setminus (B \cup C) = (A \setminus B) [\cap] (A \setminus C)$.
---
C: A [basis|structure] is a [linearly independent] spanning set. See [[thoughts/basis]].
```

This example parses to four cards: one Q/A, one single-deletion cloze, and two siblings from the final cloze.

- Faces may span lines and contain lists, inline math, or display math. A Q/A card needs a nonempty answer. A cloze needs at least one deletion.
- Use `---` between cards for readability. A new `Q:` or `C:` also starts a card without a separator. A standalone line of three or more dashes ends the card, including inside a multiline answer.
- `[answer]` hides the answer; `[answer|hint]` supplies a cue. Other deletions remain visible when one sibling is tested.
- Wikilinks and inline Markdown links are recognized without becoming deletions. Check the parsed result when using other literal bracket syntax, such as array notation, in a cloze.
- Optional leading YAML frontmatter is stripped by the parser. Usually omit it; a `title` can override the viewer heading. Avoid a leading separator that would be consumed as frontmatter.

`hashCard` hashes normalized text. Q/A IDs use both faces. Cloze group IDs use the sentence; sibling IDs also include the deletion index. Adding a citation to an existing card changes that text and may change its ID. Compare parser output before and after an edit instead of estimating identity changes from appearance.
