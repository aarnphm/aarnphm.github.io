---
name: flashcards
description: Create or revise Garden .fc study cards from source material while preserving existing card identities and schedules.
---

# flashcards

a deck is a plain-text file of cards that lives next to the note it tests: `content/thoughts/Sets.md` (source) pairs with `content/thoughts/Sets.fc` (deck). same basename, same directory, bare `.fc` (or `.flashcards`) extension — never `.fc.md`. quartz emits a viewer at `/<slug>/flashcards` and the source note grows a `review` link automatically (ContentMeta keys off the sibling file existing). one deck per note.

the format is a port of [eudoxia0/hashcards](https://github.com/eudoxia0/hashcards). the machinery lives in `quartz/util/flashcards.ts` (parser), `quartz/plugins/transformers/flashcards.ts` (faces → mdast), `quartz/plugins/emitters/flashcardsPage.tsx`, and `worker/flashcards.ts` (FSRS scheduling over D1, behind GitHub login). you write the `.fc`; that chain renders it.

## the format

```
Q: front of a basic card
A: back of the card
---
C: a sentence with one or more [bracketed] deletions
```

- **`Q:` / `A:`** — a directional recall card. both faces run multi-line: everything after `A:` (including blank lines, lists, display math) belongs to the back until the next `Q:`, `C:`, or `---`.
- **`C:`** — a cloze card. each `[answer]` is a blanked span. `[answer|hint]` shows `hint` on the front instead of `[…]`. **multiple `[...]` in one `C:` become sibling cards** (one blank each, the rest shown); siblings share a group and are buried together when you grade one in a drill.
- **`---`** (three or more dashes) separates cards. optional between a complete `Q:/A:` and the next `Q:`, but **required** to split two cloze cards or to end a card before a non-`Q:` line. use it everywhere for legibility.
- **math** — `$inline$` and `$$display$$` render via katex, in either face.
- **links** — `[[wikilinks]]` resolve against the garden and are safe inside `C:` (the cloze parser skips `[[ ]]` and markdown `](` tails, so a link is never mistaken for a deletion). markdown links and `![](images)` work too; relative paths rebase from the source note.
- **frontmatter** — optional leading `---\n…\n---` yaml is stripped before parsing. omit it by default (the note already has frontmatter); add a `title:` only if you want the deck header to read differently from the filename.

parse errors surface as a warning callout on the rendered page: a `Q:` with no `A:`, or a `C:` with no `[...]`. a `.fc` with zero cards renders an empty deck — don't save prose-only notes as `.fc`.

## authoring workflow

1. read the source note. identify its testable facts: definitions, notation/symbol rows, theorems and named laws, distinctions ("X vs Y"), formulas, and "if P then Q" statements. those are the cards; narrative connective tissue is not.
2. one card per atomic fact. split anything compound into separate cards.
3. pick the kind:
   - **Q/A** for term→definition, "state X", "what does N mean" — anything you want to recall from a cue.
   - **cloze** for a fact that only makes sense in its sentence, for notation tables, and for keeping surrounding context visible while blanking the operative token.
4. reuse the note's own `[[wikilinks]]` and `$katex$` so each card points back to where the idea is defined.
5. write to `<basename>.fc` in the same directory. **if the deck exists, merge additively**: keep already-correct cards byte-for-byte and only add or fix. card ids are a content hash of the normalized faces (`quartz/util/flashcards.ts:hashCard`), so rewording a card resets its FSRS schedule — whitespace reflow is safe, semantic edits are not. don't churn cards you didn't mean to reset.

## what makes a good card

practitioner rules (Wozniak's "twenty rules", trimmed to what bites here):

- **minimum information** — the smallest stand-alone fact. "what's the union of $A$ and $\emptyset$" beats "state the identities for union and intersection with the empty set".
- **no enumerations** — a bulleted list memorized whole is the worst card. turn a list into one cloze per item, or several Q/A.
- **cloze for context** — when the fact lives in a sentence, blank the token and keep the sentence: `the complement of the union is the [intersection] of the complements`.
- **be unambiguous** — a prompt with several defensible answers trains nothing. name the exact thing asked.
- **no yes/no** — rephrase "is $\emptyset$ a subset of every set?" into "what is $A \cap \emptyset$?".
- **cite the source** — a wikilink on the card both grounds it and makes the deck a study index back into the garden.

## worked example (`content/thoughts/Sets.fc`)

```
Q: What does $A \subseteq B$ mean?
A: Every element of $A$ is also an element of $B$.

---

Q: State the cartesian product $A \times B$.
A:

$$
A \times B = \{(a,b) \mid a \in A \text{ and } b \in B\}
$$

---

C: De Morgan's law: $A \setminus (B \cup C) = (A \setminus B) [\cap] (A \setminus C)$.

---

C: The [axiom of separation|which axiom] only lets you carve a subset out of an [existing set], which is how ZFC blocks [[thoughts/Wittgenstein#Russell's paradox and the vicious circle principle|Russell's paradox]].
```

reading the cards: the first is plain Q/A. the second shows a multi-line back whose answer is display math. the third clozes a single operator (`$[\cap]$` blanks `\cap` inside the katex). the fourth is the stress test — two deletions (so two sibling cards), a `|hint` on the first, and a `[[wikilink]]` the parser must leave intact.

## conventions

- card faces are Obsidian-flavored Markdown. Preserve the source note's wikilinks and KaTeX. Use the markdown skill for syntax details when available and needed.
- bare `.fc`/`.flashcards`, same basename and folder as the note. nothing else discovers the deck.
- validate with the existing parser and relevant tests. For browser inspection, wait for the matching watcher `build:ready`, confirm HTTP availability, and load `/<slug>/flashcards`; do not launch a full build. the drill (grade buttons, due queue) only lights up once you're signed in via GitHub, since scheduling is per-login in D1.
