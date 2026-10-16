# CoursesIndex

The list of mirrored courses at `/courses/`: one bordered row per course with its study counts beneath.

**Source:** `quartz/components/CoursesIndex.tsx` · styles: `quartz/components/styles/course.scss` · data: `quartz/util/course.ts` (`courseView`)

## Preview

A copy of the server-rendered markup from `/courses/` (selector `.course-home`), styled by `site/garden.css`, the site's compiled CSS pruned to the rules these previews use. Links point to `#`. The preview is static because `course.inline.ts` fills the owner-only cells from `GET /api/flashcards/state?prefix=`. The harvest ran signed in, so `due` and `recall` show aarnphm's state on 2026-10-11.

## Rules

- A row reads `number · title · term · instructor`, with `notes 1/38 · cards 24 · due – · last –` under it in `meta-mono`.
- Signed out, every owner-only cell is `–` and the action reads `sign in to schedule`.
- The action is a text link (`review due –`). Rows and cells do not fill on hover.
- The markdown body of `content/courses/index.md` is not rendered.
