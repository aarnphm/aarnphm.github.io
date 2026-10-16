# CourseUnit

The bar above a lecture note and the foot below it.

**Source:** `quartz/components/CourseUnit.tsx` (`CourseUnitBar`, `CourseUnitFoot`) · styles: `quartz/components/styles/course.scss`

## Preview

A copy of the server-rendered markup from `/courses/18.905-fall-2016/lectures/01` (selector `.course-unit-bar`), styled by `site/garden.css`, the site's compiled CSS pruned to the rules these previews use. Links point to `#`. and `.course-unit-foot`. The two parts sit on one card; on the site the note body sits between them. The preview is static because the due count is an owner-only fetch.

## Rules

- The bar is a crumb: `18.905 / i. singular homology / lecture 01 of 38 · due – of 24`. Separators are `/` in `gray`.
- The foot lists what is due here, then the lecture pdf, `open in reader` and prev/next.
- Notes pages (`notes/<nn>`) add `guth's notes`, the taught date and a `fold proofs` toggle with `aria-pressed` to the bar. Their foot adds `full notes pp. 18–24`, the video, the tex source and a `<details class="course-latex">` that fetches the section's lines with copy.
- Links underline on hover. The base `a:hover` dark fill is off inside course views.
