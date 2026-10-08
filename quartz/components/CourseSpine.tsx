import { JSX } from 'preact'
import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../types/component'
import {
  CourseUnitView,
  CourseView,
  courseView,
  formatCount,
  resourceSlug,
  unitDeckSlug,
  unitPdf,
} from '../util/course'
import { classNames } from '../util/lang'
import { FullSlug, resolveRelative } from '../util/path'
// @ts-ignore
import script from './scripts/course.inline'
import style from './styles/course.scss'

const roman = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x']

const pad = (n: number) => String(n).padStart(2, '0')

export const psetNumber = (id: string): number | undefined => {
  const match = /(?:pset|ps|problem[-_ ]?set)[-_ ]?0*(\d+)/i.exec(id)
  return match ? Number(match[1]) : undefined
}

/** Bar shared by the spine and the courses index: counts, then one review action. */
export function CourseBar({
  from,
  heading,
  headingId,
  notes,
  total,
  cards,
  cardIds,
  reviewSlug,
  hasDecks,
}: {
  from: FullSlug
  heading: string
  headingId: string
  notes: number
  total: number
  cards: number
  cardIds: string[]
  reviewSlug: FullSlug
  hasDecks: boolean
}): JSX.Element {
  return (
    <header class="course-bar">
      <h2 id={headingId}>{heading}</h2>
      <p class="course-count" data-cards={cardIds.join(' ')} data-unit="all">
        notes {notes}/{total} · cards {hasDecks ? formatCount(cards) : '–'} · due{' '}
        <span data-due="">–</span> · last{' '}
        <span data-last="" data-for="all">
          –
        </span>{' '}
        · recall{' '}
        <span data-retention="" data-for="all">
          –
        </span>
      </p>
      {hasDecks ? (
        <a
          class="course-action internal"
          href={resolveRelative(from, reviewSlug)}
          data-slug={reviewSlug}
          data-no-popover
          data-course-action="review"
          data-cards={cardIds.join(' ')}
          data-unit="all-action"
        >
          review due <span data-due="">–</span>
        </a>
      ) : (
        <span class="course-action" aria-disabled="true">
          no decks yet
        </span>
      )}
    </header>
  )
}

function unitRow(view: CourseView, unit: CourseUnitView, from: FullSlug, frontier: boolean) {
  const pdf = unitPdf(unit.record)
  const pdfHref = pdf ? resolveRelative(from, resourceSlug(view.dir, pdf.id)) : undefined
  const topicHref = unit.state === 'new' ? pdfHref : resolveRelative(from, unit.slug)
  const dates = unit.record.keyDates.map((date, i) => {
    const dueId = unit.record.due[i]
    return dueId ? (
      <a class="internal" href={resolveRelative(from, resourceSlug(view.dir, dueId))}>
        {date}
      </a>
    ) : (
      <span>{date}</span>
    )
  })
  return (
    <tr
      class="course-unit-row"
      data-state={unit.state}
      data-frontier={frontier ? 'true' : undefined}
    >
      <td class="course-num course-c-n">{pad(unit.record.n)}</td>
      <td>
        {topicHref ? (
          <a class="internal course-unit-title" href={topicHref}>
            {unit.record.title}
          </a>
        ) : (
          <span class="course-unit-title">{unit.record.title}</span>
        )}
        {unit.record.sessions ? (
          <span class="course-unit-dates">{unit.record.sessions}</span>
        ) : null}
        {dates.length > 0 ? <span class="course-unit-dates">{dates}</span> : null}
      </td>
      <td class="course-c-src">
        {pdfHref ? (
          <a class="internal" href={pdfHref}>
            pdf
          </a>
        ) : (
          '–'
        )}
      </td>
      <td class="course-c-cards">
        {unit.deck ? (
          <a
            class="internal"
            href={resolveRelative(from, unitDeckSlug(view.dir, unit.record.n))}
            data-no-popover
          >
            {unit.cardIds.length}
          </a>
        ) : (
          '–'
        )}
      </td>
      <td
        class="course-c-due"
        data-due=""
        data-cards={unit.cardIds.length > 0 ? unit.cardIds.join(' ') : undefined}
        data-unit={String(unit.record.n)}
      >
        –
      </td>
      <td class="course-c-last" data-last="" data-for={String(unit.record.n)}>
        –
      </td>
    </tr>
  )
}

function psetRows(view: CourseView, from: FullSlug) {
  const seen = new Set<string>()
  const rows: JSX.Element[] = []
  for (const unit of view.units) {
    unit.record.due.forEach((id, i) => {
      if (seen.has(id)) return
      seen.add(id)
      const n = psetNumber(id)
      const solution = n === undefined ? undefined : view.psets.get(n)
      rows.push(
        <tr class="course-unit-row" data-state={solution ? 'notes' : 'new'}>
          <td class="course-num course-c-n">{n === undefined ? '–' : pad(n)}</td>
          <td>
            <a
              class="internal course-unit-title"
              href={resolveRelative(from, resourceSlug(view.dir, id))}
            >
              {unit.record.keyDates[i] ?? id}
            </a>
            <span class="course-unit-dates">
              due at {view.record.unitKind} {pad(unit.record.n)}
            </span>
          </td>
          <td class="course-c-src">
            <a class="internal" href={resolveRelative(from, resourceSlug(view.dir, id))}>
              pdf
            </a>
          </td>
          <td class="course-c-cards" colSpan={3}>
            {solution ? (
              <a class="internal" href={resolveRelative(from, solution.slug as FullSlug)}>
                our solutions
              </a>
            ) : (
              '–'
            )}
          </td>
        </tr>,
      )
    })
  }
  return rows
}

export default (() => {
  const CourseSpine: QuartzComponent = ({
    ctx,
    fileData,
    allFiles,
    displayClass,
  }: QuartzComponentProps) => {
    const view = courseView(fileData, allFiles, ctx.decks)
    if (!view) return null
    const from = fileData.slug!
    const total = view.units.length
    const notes = view.units.filter(unit => unit.note).length
    const cardIds = view.units.flatMap(unit => unit.cardIds)
    const hasDecks = view.units.some(unit => unit.deck)
    // the frontier is the first unit without notes after the last one that has them
    const lastNoted = view.units.map(u => Boolean(u.note)).lastIndexOf(true)
    const frontierIdx = view.units.findIndex((u, i) => i > lastNoted && !u.note)

    const parts = new Map<number | undefined, CourseUnitView[]>()
    for (const unit of view.units) {
      const key = view.record.parts.length > 0 ? unit.record.part : undefined
      const list = parts.get(key) ?? []
      list.push(unit)
      parts.set(key, list)
    }
    const psets = psetRows(view, from)
    const kindPlural = `${view.record.unitKind}s`

    return (
      <section
        class={classNames(displayClass, 'course-spine')}
        data-course-prefix={`courses/${view.dir}/`}
        aria-labelledby="course-spine-h"
      >
        <CourseBar
          from={from}
          heading={kindPlural}
          headingId="course-spine-h"
          notes={notes}
          total={total}
          cards={cardIds.length}
          cardIds={cardIds}
          reviewSlug={`courses/review?in=${view.dir}` as FullSlug}
          hasDecks={hasDecks}
        />
        <table class="course-spine-table">
          <thead>
            <tr>
              <th scope="col" class="course-c-n">
                #
              </th>
              <th scope="col">topic</th>
              <th scope="col" class="course-c-src">
                src
              </th>
              <th scope="col" class="course-c-cards">
                cards
              </th>
              <th scope="col" class="course-c-due">
                due
              </th>
              <th scope="col" class="course-c-last">
                last
              </th>
            </tr>
          </thead>
          {Array.from(parts.entries()).map(([part, units]) => (
            <tbody class="course-part">
              {part !== undefined && view.record.parts[part - 1] ? (
                <tr class="course-part-head">
                  <th scope="rowgroup" colSpan={6}>
                    {roman[part - 1] ?? part}. {view.record.parts[part - 1]}
                    <span class="course-count">
                      notes {units.filter(u => u.note).length}/{units.length}
                    </span>
                  </th>
                </tr>
              ) : null}
              {units.map(unit =>
                unitRow(view, unit, from, view.units.indexOf(unit) === frontierIdx),
              )}
            </tbody>
          ))}
          {psets.length > 0 ? (
            <tbody class="course-part course-part--psets">
              <tr class="course-part-head">
                <th scope="rowgroup" colSpan={6}>
                  problem sets
                  <span class="course-count">
                    solved {view.psets.size}/{psets.length}
                  </span>
                </th>
              </tr>
              {psets}
            </tbody>
          ) : null}
        </table>
        {view.record.trailing.length > 0 ? (
          <p class="course-spine-trailing">{view.record.trailing.join(' · ')}</p>
        ) : null}
      </section>
    )
  }
  CourseSpine.css = style
  CourseSpine.afterDOMLoaded = script
  return CourseSpine
}) satisfies QuartzComponentConstructor
