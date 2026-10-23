# Courses

`/courses/` holds mirrors of MIT OpenCourseWare courses with a study layer on top: lecture notes, one flashcard deck per unit, due counts from the FSRS review store, and converted LaTeX notes. The mirrors stay regenerable by `pnpm vendor:ocw`. What aarnphm writes (notes, decks, psets and statements) sits beside them in git.

The course pages use the tool register: mono lowercase labels, the shared `line` / `surface` / `selected` recipe, and underlined text links.

## Views

| Route                                  | Component                     | Shape                                                                            |
| -------------------------------------- | ----------------------------- | -------------------------------------------------------------------------------- |
| `/courses/`                            | CoursesIndex                  | Trail bar, a bar of totals, bordered rows with counts beneath                    |
| `/courses/<dir>/`                      | CourseHome                    | Trail bar, masthead, rail, schedule table grouped by part                        |
| `/courses/<dir>/lectures/<nn>`         | CourseUnit, CourseWorkspace   | Unit bar, the note, unit foot; the workspace beside or under it                  |
| `/courses/<dir>/notes/<nn>`            | CourseUnit, LatexCallout      | Converted LaTeX notes with numbered statements                                   |
| `/courses/<dir>/resources/<slug>/`     | CourseCollection, CourseWatch | Course strip over a collection table, a video with transcript, or the PDF reader |
| `/courses/<dir>/statements/`           | not in the system             | Theorems and definitions with the cards that use them                            |
| `<note>/flashcards`, `/courses/review` | Flashcards                    | Card with the note rail; the merged drill                                        |

The home takes `--layout-full-column` (`5 / span 5` on desktop, the reading column on mobile), wider than the 740px reading column. The strip and stacked views share a 66rem centred column. The site breadcrumbs stay off every page with a course crumb, because the trail bar replaces them.

## Trail bar

`nav.course-unit-bar` is the one top bar of the course pages: a crumb (`courses / 18.905 / i. singular homology / 01/38`) and, at the right, what the page offers (dates, study toggles, the due count, actions). The courses index, the course home, units, the university pages and the français home and units all use it. The crumb separator is `/` in `gray`; the bar never fills on hover.

## Study toggles

On a unit bar, `read` and `problems` are toggles with `aria-pressed`, followed by a `log` link to the study log. In dev they post to the study endpoint and the title carries the date. In a build, a set date shows as a link to the log and an unset one is left out, so the row never grows with save feedback. The schedule `state` column carries the same two marks as `r` and `p`. Totals read `read x/y · notes x/y` on the index and in each part head.

## Colour

Course views declare the shared recipe under their own prefix:

| Role               | System token | Course name               | Value in the code                   |
| ------------------ | ------------ | ------------------------- | ----------------------------------- |
| Hairline           | `line`       | `--course-line`           | `gray` at 55%                       |
| Raised ground      | `surface`    | `--course-surface`        | `light` 86% with `lightgray`        |
| Selected ground    | `selected`   | `--course-selected`       | `secondary` at 18% over the surface |
| Workspace hairline | `line`       | `--course-workspace-line` | `gray` at 45%                       |

`--course-control-height` is 1.75rem and rises to 44px under `pointer: coarse`.

Four source rules draw text in `gray` (the strip table head, the license line, the empty schedule slot and the unselected workspace tab). The system uses `darkgray` for each.

## Links

The site's `a:hover` inverts links to a dark fill. Course views turn that off inside their own roots and underline instead: `text-decoration-color` moves from `gray` to `dark` over 120ms. Rows and cells never fill on hover. Actions are text links ("review due –", "open in reader").

## Owner-only cells

`due`, `last` and `recall` belong to the signed-in owner. `course.inline.ts` fills them from one `GET /api/flashcards/state?prefix=` per `[data-course-prefix]` root and counts only card ids embedded at build time. `due` shows the due count and `+n` for cards never seen. Signed out, each cell is `–` and the action reads "sign in to schedule". `recall` is mean FSRS retrievability over seen cards, an estimate, and its title says so. The study toggles are owner-only too: they exist only where a study endpoint is set.

The harvested previews ran in dev, signed in, on 2026-10-23, so they show aarnphm's counts and the toggles.

## Overlays

- The lecture workspace is a fixed side panel above 1400px, with a drag divider. At 1400px and below it is an inline panel under the note. It is not a modal and has no `<dialog>`.
- The strip's links table becomes a native `popover` bottom sheet opened by a `links` button below 800px.
- A link popover on a course link reads only the page's `[data-course-card]` head (crumb, title, meta line) and optional `[data-course-rows]`. The generic `.popover-hint` lift never applies under `/courses/`.

## Focus

The workspace tabs use the `tri-map-tab` ring: 2px `focus`, outside the tab, over the selected fill (`focus.md`, Tabs). Study toggles and links keep the 2px `focus` ring at `2px`.

## Rules

- Course directory names are frozen. The review store keys decks by path, so a rename orphans every review.
- Unit state is derived from the repo: `new`, `notes`, `deck`. The first unit without `study.read` is the frontier. Never show a state the repo cannot back.
- Nothing is colour-only. A due count keeps its word ("due 3 of 24").
