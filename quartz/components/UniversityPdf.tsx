import { JSX } from 'preact'
import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../types/component'
import { formatBytes } from '../util/course'
import { classNames } from '../util/lang'
import { FullSlug, resolveRelative } from '../util/path'
import { concatenateResources } from '../util/resources'
import { courseCode, termLabel, UNIVERSITY_SLUG } from '../util/university'
// @ts-ignore
import script from './scripts/course.inline'
import studyStyle from './styles/course-study.scss'
import style from './styles/course.scss'

const MAX_CITED = 6

/**
 * The row above the PDF reader on a university PDF page: the course crumb, title, meta line and
 * a links table, in the markup of the MIT resource strip so both share a stylesheet and the
 * `[data-course-card]` popover contract.
 */
export const UniversityPdfStrip = (() => {
  const Strip: QuartzComponent = ({ fileData, displayClass }: QuartzComponentProps) => {
    const page = fileData.universityPdf
    if (!page) return null
    const from = fileData.slug as FullSlug
    const internal = (slug: string, text: string, extra: Record<string, string> = {}) => (
      <a
        class="internal"
        href={resolveRelative(from, slug as FullSlug)}
        data-slug={slug}
        {...extra}
      >
        {text}
      </a>
    )
    const pagerLink = (rel: 'prev' | 'next', sibling: { title: string; path: string }) => (
      <a
        rel={rel}
        class="internal"
        href={sibling.path}
        data-slug={sibling.path.replace(/^\/|\/$/g, '')}
        data-course-key={rel}
        data-no-popover
        title={sibling.title}
      >
        {rel === 'prev' ? '‹ prev' : 'next ›'}
      </a>
    )
    const cited = page.citedBy.slice(0, MAX_CITED)
    const rows: [string, JSX.Element[]][] = []
    const pager = [
      ...(page.prev ? [pagerLink('prev', page.prev)] : []),
      ...(page.next ? [pagerLink('next', page.next)] : []),
    ]
    if (pager.length > 0) rows.push(['pdfs', pager])
    if (page.home)
      rows.push(['course', [internal(page.home, page.homeTitle ?? courseCode(page.course))]])
    if (cited.length > 0)
      rows.push(['cited by', cited.map(note => internal(note.from, note.title.toLowerCase()))])
    rows.push([
      'file',
      [
        <a href={page.rawPath} data-no-popover>
          raw pdf
        </a>,
      ],
    ])
    const meta = ['pdf', formatBytes(page.bytes), `${page.position[0]} of ${page.position[1]}`]
      .filter(Boolean)
      .join(' · ')
    return (
      <nav class={classNames(displayClass, 'course-resource-strip')} aria-label="university pdf">
        <div class="course-strip" data-course-card>
          <div class="course-strip-head">
            <p class="course-crumb">
              {internal(UNIVERSITY_SLUG, 'university', { 'data-no-popover': '' })}
              <span class="course-unit-sep">/</span>
              {page.termHome ? (
                internal(page.termHome, termLabel(page.term))
              ) : (
                <span>{termLabel(page.term)}</span>
              )}
              <span class="course-unit-sep">/</span>
              {page.home ? (
                <a
                  class="internal"
                  href={resolveRelative(from, page.home as FullSlug)}
                  data-slug={page.home}
                  data-course-key="up"
                >
                  {courseCode(page.course)}
                </a>
              ) : (
                <span>{courseCode(page.course)}</span>
              )}
              <span class="course-unit-sep">/</span>
              <span class="course-num">pdf</span>
            </p>
            <h1 class="course-card-title">{page.title}</h1>
            <p class="course-count">{meta}</p>
          </div>
          <button
            type="button"
            class="course-strip-toggle"
            popovertarget="course-strip-links"
            aria-label="links for this pdf"
          >
            links
          </button>
          <div class="course-strip-links" id="course-strip-links" popover="auto">
            <table class="course-strip-table">
              <tbody>
                {rows.map(([label, items]) => (
                  <tr key={label}>
                    <th scope="row">{label}</th>
                    <td>
                      {items.map((item, i) => (
                        <>
                          {i > 0 ? <span class="course-sep">·</span> : null}
                          {item}
                        </>
                      ))}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </nav>
    )
  }
  Strip.displayName = 'UniversityPdfStrip'
  Strip.css = concatenateResources(style, studyStyle)
  Strip.afterDOMLoaded = script
  return Strip
}) satisfies QuartzComponentConstructor
