# Flashcards

The flashcard deck view: the deck bar, the current card and the note rail.

**Source:** `quartz/components/FlashcardsContent.tsx` · script: `quartz/components/scripts/flashcards.inline.ts` · styles: `quartz/components/styles/flashcards.scss`

## Preview

A copy of the server-rendered markup from `/courses/18.905-fall-2016/lectures/01/flashcards` (selector `.flashcards-root`), styled by `site/garden.css`, the site's compiled CSS pruned to the rules these previews use. Links point to `#` and the rail's note is trimmed. The preview is static because its script reads and writes review state through `/api/flashcards`.

## Rules

- Every `<note>/flashcards` page and `/courses/review` use the flashcards shell: no title, byline or footer (`pageLayout: flashcards` flags the body).
- The note rail beside the card shows the page the active card came from, fetched once per note. `n` toggles it; it opens by default from 64rem. `notes` in the bar shows its state with `aria-pressed`.
- `/courses/review` is the one merged drill across decks. `?in=<course-dir>` narrows it and `?new=` caps new cards (default 10). Grading one cloze buries its siblings for the session; undo restores them.
- Scheduling runs at run time. Card text and identities are stable; editing them affects scheduling. `ID:` pins keep a reworded card's schedule.
