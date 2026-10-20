import { JSX } from 'preact'
import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../types/component'
import { classNames } from '../util/lang'
import { FullSlug, resolveRelative } from '../util/path'
import {
  UNIVERSITY_ROOT,
  UNIVERSITY_SLUG,
  UniversityPlace,
  courseCode,
  termLabel,
  universityFolderName,
  universityPlaceOf,
} from '../util/university'
import { isUniversityPdfPage } from '../util/university-pdf'

const sep = <span class="course-unit-sep">/</span>

/**
 * `university / 2024–25 / SFWR 4ML3 / a1`: every step links up the trail. On a folder page the
 * last folder is the page itself, so it is plain text.
 */
export function UniversityCrumb({
  from,
  place,
  paths,
}: {
  from: FullSlug
  place: UniversityPlace
  paths: readonly string[]
}): JSX.Element {
  const home = `${UNIVERSITY_ROOT}${place.term}/${place.course}`
  const universityHref = resolveRelative(from, UNIVERSITY_SLUG as FullSlug)
  const link = (slug: string, text: string) => (
    <a
      class="internal"
      href={resolveRelative(from, slug as FullSlug)}
      data-slug={slug}
      data-no-popover
    >
      {text}
    </a>
  )
  const atHome = place.isFolder && place.folders.length === 0
  return (
    <>
      <a class="internal" href={universityHref} data-slug={UNIVERSITY_SLUG} data-no-popover>
        university
      </a>
      {sep}
      <a
        class="internal"
        href={`${universityHref}#term-${place.term}`}
        data-slug={UNIVERSITY_SLUG}
        data-no-popover
      >
        {termLabel(place.term)}
      </a>
      {sep}
      {atHome ? (
        <span>{courseCode(place.course)}</span>
      ) : (
        link(`${home}/index`, courseCode(place.course))
      )}
      {place.folders.map((_, i) => {
        const folder = `${home}/${place.folders.slice(0, i + 1).join('/')}`
        const name = universityFolderName(paths, folder)
        const current = place.isFolder && i === place.folders.length - 1
        return (
          <>
            {sep}
            {current ? <span>{name}</span> : link(`${folder}/index`, name)}
          </>
        )
      })}
    </>
  )
}

/** The trail above a note or notebook inside a university course; folder pages draw their own. */
export default (() => {
  const UniversityCrumbBar: QuartzComponent = ({
    fileData,
    allFiles,
    ctx,
    displayClass,
  }: QuartzComponentProps) => {
    const from = fileData.slug!
    const place = universityPlaceOf(from)
    if (!place || place.isFolder || isUniversityPdfPage(from)) return null
    const homeSlug = `${UNIVERSITY_ROOT}${place.term}/${place.course}/index`
    const courseTitle = allFiles.find(file => file.slug === homeSlug)?.frontmatter?.title
    return (
      <nav class={classNames(displayClass, 'course-unit-bar')} aria-label="course trail">
        <span class="course-crumb">
          <UniversityCrumb from={from} place={place} paths={ctx.allFiles} />
        </span>
        {courseTitle ? <span class="course-count course-unit-dates">{courseTitle}</span> : null}
      </nav>
    )
  }
  return UniversityCrumbBar
}) satisfies QuartzComponentConstructor
