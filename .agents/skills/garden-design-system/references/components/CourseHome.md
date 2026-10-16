# CourseHome

The home of one course: a masthead, a rail of course pages, and the schedule table.

**Source:** `quartz/components/CourseHome.tsx` · styles: `quartz/components/styles/course.scss` · data: the `course:` frontmatter that `pnpm vendor:ocw` writes on the course `index.md`

## Preview

A copy of the server-rendered markup from `/courses/18.905-fall-2016/` (selector `.course-home`), styled by `site/garden.css`, the site's compiled CSS pruned to the rules these previews use. Links point to `#`. The preview is static for the same reason as CoursesIndex: `due`, `last` and `recall` come from one state fetch per `[data-course-prefix]` root. The harvest ran signed in.

## Rules

- The home spans the reading edge through the side column (`grid-column: 5 / -3`), wider than the 740px full column.
- Masthead: course number, name, then `term · level · instructor · mit opencourseware` in `meta`, then the description.
- Rail: the vendored `pages/*`, the resource collections, topics and license. It sits beside the table above a 56rem container width and folds under the masthead below it.
- Schedule: a `<table>` grouped by part, with columns `# · topic · src · marks · cards · due · last`. Key dates sit under the topic. Psets form a trailing group.
- Container queries hide `last` below 36rem and `src` below 24rem.
- Unit state comes from the repo: `new` (no note), `notes` (note, no deck), `deck` (sibling `.fc`). The first unit without `study.read` is the frontier.
