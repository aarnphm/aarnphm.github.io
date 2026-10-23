# CourseUnit

The bar above a lecture note and the foot below it.

**Source:** `quartz/components/CourseUnit.tsx` (`CourseUnitBar`, `CourseUnitFoot`) and `quartz/components/CourseStudy.tsx` (`StudyState`) · styles: `quartz/components/styles/course.scss` (`course/home-and-unit.scss`, `course/trail-bar.scss`), `course-study.scss`

## Preview

A copy of the server-rendered markup from `/courses/18.905-fall-2016/lectures/01` (selectors `nav.course-unit-bar` and `.course-unit-foot`), styled by `site/garden.css`, the site's compiled CSS pruned to the rules these previews use. Links point to `#`. The preview shows the two parts as two `main` blocks; on the site the note body sits between them. The preview is static because the study toggles post to a dev endpoint and the due count is an owner-only fetch.

## Rules

- The bar is a crumb: `courses / 18.905 / i. singular homology / 01/38`, then the actions: `read`, `problems`, a `log` link and `due 0 +16 of 16`. Separators are `/` in `gray`.
- `read` and `problems` are toggles with `aria-pressed`. In dev they post to the study endpoint and the label carries the date in its title. In a build a set date links to the study log and an unset one is left out. The row does not grow with save feedback.
- The same `nav.course-unit-bar` is the trail on the courses index, the university pages and the français home and units (`fr-unit-bar` adds its own accents).
- The foot lists what is due here, `sources`, `open in reader` and prev/next (`02 homology`).
- Notes pages (`notes/<nn>`) add the taught date (`taught 2025-04-15`) and a `fold proofs` toggle with `aria-pressed` to the bar. Their foot adds `full notes pp. 18–24`, the video, the tex source and a `<details class="course-latex">` that fetches the section's lines with copy.
- Links underline on hover. The base `a:hover` dark fill is off inside course views.
