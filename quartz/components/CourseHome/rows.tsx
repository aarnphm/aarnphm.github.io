import {
  CourseResourceRef,
  CourseUnitView,
  CourseView,
  courseFileSlug,
  formatDuration,
  psetNumber,
  psetTextSlug,
  resourceSlug,
  unitDeckSlug,
  unitPdf,
  unitSources,
} from '../../util/course'
import { parseProblems } from '../../util/course-study'
import { FullSlug, resolveRelative } from '../../util/path'
import { psetProblems } from '../CourseDesk'
import { StudySlots } from '../CourseStudy'

const pad = (n: number) => String(n).padStart(2, '0')

export type Slot = 'pdf' | 'tex' | 'video' | 'txt'

/** `1:18` for the fixed-width slot: hours and minutes, or minutes and seconds under an hour. */
function slotDuration(seconds: number | undefined): string | undefined {
  const full = formatDuration(seconds)
  if (!full) return undefined
  const parts = full.split(':')
  return parts.length === 3 ? `${parts[0]}:${parts[1]}` : full
}
const slotOrder: Slot[] = ['pdf', 'tex', 'video', 'txt']

interface SlotLink {
  slot: Slot
  href?: string
  text: string
  slug?: FullSlug
  title?: string
}

/** The `src` cell: fixed slots in course order, `–` where the unit has nothing for a slot. */
export function unitSlots(
  view: CourseView,
  unit: CourseUnitView,
  from: FullSlug,
  resources: Map<string, CourseResourceRef>,
): SlotLink[] {
  const sources = unitSources(unit, resources)
  const pdfUnit = unitPdf(unit.record)
  const pdfSlug = sources.pdf?.slug ?? (pdfUnit ? resourceSlug(view.dir, pdfUnit.id) : undefined)
  const length = slotDuration(sources.video?.record.duration)
  return [
    {
      slot: 'pdf',
      text: 'pdf',
      ...(pdfSlug ? { href: resolveRelative(from, pdfSlug), slug: pdfSlug } : {}),
    },
    {
      slot: 'tex',
      text: 'tex',
      ...(unit.text
        ? {
            href: resolveRelative(from, unit.text.slug as FullSlug),
            slug: unit.text.slug as FullSlug,
          }
        : {}),
    },
    {
      slot: 'video',
      text: length ?? 'video',
      ...(sources.video
        ? {
            href: resolveRelative(from, sources.video.slug),
            slug: sources.video.slug,
            ...(sources.videos.length > 1 ? { title: `${sources.videos.length} videos` } : {}),
          }
        : {}),
    },
    {
      slot: 'txt',
      text: 'txt',
      ...(sources.transcript
        ? { href: resolveRelative(from, sources.transcript.slug), slug: sources.transcript.slug }
        : {}),
    },
  ]
}

function slotCell(links: SlotLink[], slots: Set<Slot>, className: string) {
  return (
    <span class={className}>
      {slotOrder
        .filter(slot => slots.has(slot))
        .map(slot => {
          const link = links.find(entry => entry.slot === slot)!
          return link.href ? (
            <a
              class="internal course-slot"
              href={link.href}
              data-slug={link.slug}
              title={link.title}
            >
              {link.text}
            </a>
          ) : (
            <span class="course-slot course-slot--empty" aria-hidden="true">
              –
            </span>
          )
        })}
    </span>
  )
}

export function unitRow(
  view: CourseView,
  unit: CourseUnitView,
  from: FullSlug,
  frontier: boolean,
  slots: Set<Slot>,
  resources: Map<string, CourseResourceRef>,
) {
  const links = unitSlots(view, unit, from, resources)
  const pdf = links.find(link => link.slot === 'pdf')
  const video = links.find(link => link.slot === 'video')
  // a new unit's topic falls back note → pdf → watch page
  const topicHref =
    unit.state === 'new' ? (pdf?.href ?? video?.href) : resolveRelative(from, unit.slug)
  const pdfUnit = unitPdf(unit.record)
  const marksSrc = pdfUnit?.file ? courseFileSlug(view.dir, pdfUnit.file) : undefined
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
        {slotCell(links, slots, 'course-unit-src')}
      </td>
      <td class="course-c-state">
        <StudySlots unit={unit} />
      </td>
      <td class="course-c-src">{slotCell(links, slots, 'course-slots')}</td>
      <td
        class="course-num course-c-marks"
        data-marks-src={marksSrc}
        title={marksSrc ? 'marks · questions in the reader' : undefined}
      >
        –
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

/**
 * Problem sets in due order: those the calendar names at a unit, then the rest of the course's
 * `pset` resources (18.156 and 18.100B list problem sets with no schedule), one per number.
 */
export function psetEntries(
  view: CourseView,
  resources: Map<string, CourseResourceRef>,
): { id: string; label: string; due?: number }[] {
  const entries: { id: string; label: string; due?: number }[] = []
  const seen = new Set<string>()
  for (const unit of view.units) {
    unit.record.due.forEach((id, i) => {
      if (seen.has(id)) return
      seen.add(id)
      entries.push({ id, label: unit.record.keyDates[i] ?? id, due: unit.record.n })
    })
  }
  const numbers = new Set(entries.map(entry => psetNumber(entry.id)))
  // the shortest id is the term's own set; `pset1-1` style ids are older terms
  const undated = Array.from(resources.values())
    .filter(ref => ref.record.kind === 'pset' && ref.record.n !== undefined)
    .sort((a, b) => a.record.n! - b.record.n! || a.id.length - b.id.length)
  for (const ref of undated) {
    if (numbers.has(ref.record.n!)) continue
    numbers.add(ref.record.n!)
    entries.push({ id: ref.id, label: `problem set ${ref.record.n}` })
  }
  return entries
}

/** Sets whose every desk problem is solved; a seeded `psets/<nn>.md` alone is a set started. */
export function solvedSets(view: CourseView): number {
  let solved = 0
  for (const [n, file] of view.psets) {
    const problems = psetProblems(view.psetText.get(n) ?? file)
    const log = parseProblems(file.frontmatter)
    if (problems.length > 0 && problems.every(problem => log.get(problem.n)?.state === 'solved'))
      solved++
  }
  return solved
}

export function psetRows(
  view: CourseView,
  from: FullSlug,
  slots: Set<Slot>,
  resources: Map<string, CourseResourceRef>,
) {
  return psetEntries(view, resources).map(({ id, label, due }) => {
    const n = psetNumber(id)
    const solution = n === undefined ? undefined : view.psets.get(n)
    const text = n === undefined ? undefined : view.psetText.get(n)
    const ref = resources.get(id)
    const pdfSlug = resourceSlug(view.dir, id)
    const marksSrc = ref?.record.file ? courseFileSlug(view.dir, ref.record.file) : undefined
    const links: SlotLink[] = [
      { slot: 'pdf', text: 'pdf', href: resolveRelative(from, pdfSlug), slug: pdfSlug },
      {
        slot: 'tex',
        text: 'tex',
        ...(text
          ? {
              href: resolveRelative(from, psetTextSlug(view.dir, n!)),
              slug: psetTextSlug(view.dir, n!),
            }
          : {}),
      },
      { slot: 'video', text: 'video' },
      { slot: 'txt', text: 'txt' },
    ]
    return (
      <tr class="course-unit-row" data-state={solution ? 'notes' : 'new'}>
        <td class="course-num course-c-n">{n === undefined ? '–' : pad(n)}</td>
        <td>
          <a class="internal course-unit-title" href={resolveRelative(from, pdfSlug)}>
            {label}
          </a>
          <span class="course-unit-dates">
            {due === undefined ? 'no due date' : `due at ${view.record.unitKind} ${pad(due)}`}
          </span>
          {slotCell(links, slots, 'course-unit-src')}
        </td>
        <td class="course-c-state" />
        <td class="course-c-src">{slotCell(links, slots, 'course-slots')}</td>
        <td class="course-num course-c-marks" data-marks-src={marksSrc}>
          –
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
      </tr>
    )
  })
}
