# CourseHome

The home of one course: the trail bar, a masthead, a rail of course pages, and the schedule table.

**Source:** `quartz/components/CourseHome.tsx` and `quartz/components/CourseHome/rows.tsx` (unit and problem rows) · styles: `quartz/components/styles/course.scss` (`course/home-and-unit.scss`, `course/foundations-and-index.scss`, `course-study.scss`) · data: the `course:` frontmatter that `pnpm vendor:ocw` writes on the course `index.md`

## Preview

A copy of the server-rendered markup from `/courses/18.905-fall-2016/` (selector `.course-home`), styled by `site/garden.css`, the site's compiled CSS pruned to the rules these previews use. Links point to `#` and each part keeps its first 6 rows. The preview is static for the same reason as CoursesIndex: `due`, `last` and `recall` come from one state fetch per `[data-course-prefix]` root. The harvest ran in dev, signed in, on 2026-10-23.

## Rules

- The home takes the full column (`--layout-full-column`, `5 / span 5` on desktop), wider than the 740px reading column. Below the mobile breakpoint it takes the reading column.
- Trail bar first: `courses / 18.905`, then `fall 2016 · graduate · prof. haynes miller · mit opencourseware`. Then the masthead: number, name, the description.
- Rail: the vendored `pages/*` (assignments, calendar, lecture notes, syllabus), `study` and `study log`, `statements` with a count, the resource collections with counts, topics and license. It sits beside the table above a 56rem container width and folds under the masthead below it.
- Schedule: a `<table>` grouped by part, with columns `# · topic · state · src · marks · cards · due · last`. A part head carries `read 1/13 · notes 13/13`. Psets form a trailing group (`solved 0/6`).
- `state` shows the read and problems marks (`r`, `p`; the header title spells them out). In dev they are toggles that post to the study endpoint. In a build a set date links to the study log and an unset one is left out.
- Container queries on the table hide `last` below 36rem and `src` below 24rem.
- Unit state comes from the repo: `new` (no note), `notes` (note, no deck), `deck` (sibling `.fc`). The first unit without `study.read` is the frontier.
