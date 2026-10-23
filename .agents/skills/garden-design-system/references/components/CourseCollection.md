# CourseCollection

A resource collection page of a course, such as its lecture videos: the strip and one table row per resource.

**Source:** `quartz/components/CourseResource.tsx` · styles: `quartz/components/styles/course.scss` (`course/resource-strip.scss`), `course-resource.scss`

## Preview

A copy of the server-rendered markup from `/courses/18-156-projection-theory-spring-2025/resources/lecture-videos/` (selector `.course-resource`), styled by `site/garden.css`, the site's compiled CSS pruned to the rules these previews use. Links point to `#` and the table keeps its first 8 rows. The preview is static; the page has no script.

## Rules

- The strip head shows the crumb (`courses / 18.156 / collection`), the title, and the `links`, source (`mit ocw`) and license (`cc by-nc-sa`) links at the right.
- A count line sits under the title in `meta-mono`: `24 videos · 32 h · 24 transcripts`.
- The table frames with a 1px border and reads `# · title · length · transcript`. Lengths are tabular. Text and pdf links share one cell, separated by `·`.
- The license line is `gray` in the source; this rendition uses `darkgray`.
