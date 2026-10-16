# Courses

`/courses/` holds mirrors of MIT OpenCourseWare courses with a study layer on top: lecture notes, one flashcard deck per unit, due counts from the FSRS review store, and converted LaTeX notes. The mirrors stay regenerable by `pnpm vendor:ocw`. What aarnphm writes (notes and decks) sits beside them in git.

The course pages use the tool register: mono lowercase labels, the shared `line` / `surface` / `selected` recipe, and underlined text links.

## Views

| Route | Component | Shape |
| --- | --- | --- |
| `/courses/` | CoursesIndex | Bordered rows, one per course, counts beneath |
| `/courses/<dir>/` | CourseHome | Masthead, rail, schedule table grouped by part |
| `/courses/<dir>/lectures/<nn>` | CourseUnit, CourseWorkspace | Unit bar, the note, unit foot; the workspace beside it |
| `/courses/<dir>/notes/<nn>` | CourseUnit, LatexCallout | Converted LaTeX notes with numbered statements |
| `/courses/<dir>/resources/<slug>/` | CourseCollection, CourseWatch | Course strip over a collection table, a video with transcript, or the PDF reader |
| `<note>/flashcards`, `/courses/review` | Flashcards | Card with the note rail; the merged drill |

The home spans `grid-column: 5 / -3`, wider than the 740px reading column. The strip and stacked views share its 66rem centred column. The site breadcrumbs stay off every page with a course crumb.

## Colour

Course views declare the shared recipe under their own prefix:

| Role | System token | Course name | Value in the code |
| --- | --- | --- | --- |
| Hairline | `line` | `--course-line` | `gray` at 55% |
| Raised ground | `surface` | `--course-surface` | `light` 86% with `lightgray` |
| Selected ground | `selected` | `--course-selected` | `secondary` at 18% over the surface |
| Workspace hairline | `line` | `--course-workspace-line` | `gray` at 45% |

`--course-control-height` is 1.75rem and rises to 44px under `pointer: coarse`.

## Links

The site's `a:hover` inverts links to a dark fill. Course views turn that off inside their own roots and underline instead: `text-decoration-color` moves from `gray` to `dark` over 120ms. Rows and cells never fill on hover. Actions are text links ("review due –", "open in reader").

## Owner-only cells

`due`, `last` and `recall` belong to the signed-in owner. `course.inline.ts` fills them from one `GET /api/flashcards/state?prefix=` per `[data-course-prefix]` root and counts only card ids embedded at build time. Signed out, each cell is `–` and the action reads "sign in to schedule". `recall` is mean FSRS retrievability over seen cards, an estimate, and its title says so.

The harvested previews ran signed in, so they show aarnphm's counts on 2026-10-11.

## Overlays

- The lecture workspace is a side panel above 1400px and a native modal `<dialog>` below.
- The strip's links table becomes a native `popover` bottom sheet opened by a `links` button below 800px.
- A link popover on a course link reads only the page's `[data-course-card]` head (crumb, title, meta line) and optional `[data-course-rows]`. The generic `.popover-hint` lift never applies under `/courses/`.

## Rules

- Course directory names are frozen. The review store keys decks by path, so a rename orphans every review.
- Unit state is derived from the repo: `new`, `notes`, `deck`. The first unit without `study.read` is the frontier. Never show a state the repo cannot back.
- Nothing is colour-only. A due count keeps its word ("due 3 of 24").
