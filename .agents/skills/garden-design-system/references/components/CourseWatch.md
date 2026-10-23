# CourseWatch

A lecture video resource: the course strip, the player and the transcript.

**Source:** `quartz/components/CourseWatch.tsx` and `quartz/components/CourseResource.tsx` (strip) · styles: `quartz/components/styles/course.scss` (`course/resource-strip.scss`), `course-resource.scss`

## Preview

A copy of the server-rendered markup from `/courses/18-156-projection-theory-spring-2025/resources/mit18_156s25_lec04_1080p_mp4/` (selector `.course-resource`), styled by `site/garden.css`, the site's compiled CSS pruned to the rules these previews use. Links point to `#`. The video element is a grey placeholder and the transcript keeps its first 8 paragraphs. The preview is static because the transcript follows the player time.

## Rules

- The strip sits above every course resource: a crumb (`courses / 18.156 / lecture 04 of 24 · video`), the title, a meta line (`video · 1:19:44 · 1134 cues`), and the links as a key/value table. `prev` and `next` rows come first, then `lecture pdf`, `study` (`read`, `problems`, `log`), `notes` (`notes · latex §4`), `in` and `source`.
- Below 800px the links table becomes a native `popover` bottom sheet opened by a `links` button.
- The transcript has a search field, `‹ prev` and `next ›`, a `follow` toggle and `copy`. Rows start with a time stamp in `meta-mono`. The active row has the `selected` ground. A stamp seeks the player.
- A lecture without a vendored captions file has no transcript section.
