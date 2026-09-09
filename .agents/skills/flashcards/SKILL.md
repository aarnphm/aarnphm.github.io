---
name: flashcards
description: Create or revise Garden .fc study decks from notes while preserving existing card identities.
---

# Garden flashcards

Write a sibling deck for the requested source note: `content/thoughts/Sets.md` pairs with `content/thoughts/Sets.fc`. Use the same basename and directory, with a bare `.fc` or `.flashcards` extension. One deck belongs to each note; preserve an existing deck's extension.

Read the source and existing deck. Select the requested facts and use the note's notation, wikilinks, and citations. Each prompt should test one unambiguous recall target. Use `Q:`/`A:` for a question and answer, or `C:` with `[answer|optional hint]` for contextual recall. Each deletion in a cloze produces a sibling card. See [format and examples](references/format.md) when authoring card syntax.

Keep already-correct cards byte-for-byte when adding cards. Rewording changes content-derived IDs and can disconnect existing FSRS scheduling. Cloze IDs include the whole sentence and deletion index, so changing a hint or one deletion can change every sibling's ID. Whitespace normalization preserves IDs only while parsing the same card boundaries; it can still change rendered Markdown. Make requested corrections and report affected identities.

Validate the edited deck with `parseFlashcards` in `quartz/util/flashcards.ts`: check errors, expected card count, duplicate IDs, and preservation of untouched IDs. Fix problems introduced by the edit before finishing. Deck authoring normally needs parser validation; changes to rendering or scheduling also need checks at their owning boundary.

For a requested preview, the viewer is `/<slug>/flashcards`; wait for the matching watcher `build:ready` and HTTP availability. The source note's review link is discovered from the sibling file. Grading and due queues require GitHub login and the D1-backed Worker; authoring a deck does not require changing stored schedules.
