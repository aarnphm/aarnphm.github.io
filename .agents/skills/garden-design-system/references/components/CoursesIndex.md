# CoursesIndex

The list of mirrored courses at `/courses/`: a trail bar, a bar of study counts, and one bordered row per course.

**Source:** `quartz/components/CoursesIndex.tsx` · styles: `quartz/components/styles/course.scss` (partials in `course/`: `foundations-and-index.scss`, `trail-bar.scss`) · data: `quartz/util/course.ts` (`courseView`)

## Preview

A copy of the server-rendered markup from `/courses/` (selector `.course-home`), styled by `site/garden.css`, the site's compiled CSS pruned to the rules these previews use. Links point to `#`. The preview is static because `course.inline.ts` fills the owner-only cells from `GET /api/flashcards/state?prefix=`. The harvest ran in dev, signed in, on 2026-10-23, so `due` and `last` show aarnphm's state.

## Rules

- The page opens with the trail bar (`nav.course-unit-bar`): `courses` on the left, `mit opencourseware` on the right. The bar is shared with the course pages and the français home.
- Under the bar, the section head reads `5 courses`, with the totals `read 2/127 · notes 127/127 · cards 32 · due 0 +32 · last –` and a `review due` action.
- A row reads `number · title · term · instructor`, with `read 1/38 · notes 38/38 · cards 32 · due 0 +32 · last –` under it in `meta-mono`. `+32` counts cards never seen.
- Signed out, every owner-only cell is `–` and the action reads `sign in to schedule`.
- The action is a text link (`review due –`). Rows and cells do not fill on hover.
- The markdown body of `content/courses/index.md` is not rendered.
