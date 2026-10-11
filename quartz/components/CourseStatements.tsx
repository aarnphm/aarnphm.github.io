import { QuartzPluginData } from '../plugins/vfile'
import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../types/component'
import { CourseView, courseDirOf, courseOf } from '../util/course'
import { BOLD_LEAD, STATEMENT_KINDS, statementRowId, termAnchor } from '../util/course-statements'
import { classNames } from '../util/lang'
import { MathText } from '../util/math-text'
import { FullSlug, resolveRelative } from '../util/path'
import { concatenateResources } from '../util/resources'
import { psetProblems } from './CourseDesk'
import { CoursesCrumb } from './CourseResource'
import studyStyle from './styles/course-study.scss'
import style from './styles/course.scss'

const pad = (n: number) => String(n).padStart(2, '0')

export interface Statement {
  kind: string
  /** `3.1` for a numbered callout; empty for a bold-lead term. */
  number: string
  name: string
  unit: number
  slug: FullSlug
  /** Block anchor without `^`; a bold lead's comes from the CourseStatements transformer. */
  anchor: string
  cards: number
  usedBy: string[]
}

export function statementsSlug(dir: string): FullSlug {
  return `courses/${dir}/statements/index` as FullSlug
}

const body = (file: QuartzPluginData) =>
  (file.rawMarkdownSource ?? '').replace(/^---\n[\s\S]*?\n---\n/, '')

/** `> [!theorem|latex] 3.1 Name` callouts paired with the `^theorem-3-1` anchor after them. */
function calloutStatements(file: QuartzPluginData, unit: number) {
  const out: Omit<Statement, 'cards' | 'usedBy'>[] = []
  // by quote depth: a lemma nested in a proof closes with its anchor one level up
  const pending = new Map<number, { kind: string; number: string; name: string } | null>()
  for (const line of body(file).split('\n')) {
    const [quotes = ''] = /^(?:> ?)*/.exec(line) ?? []
    const depth = (quotes.match(/>/g) ?? []).length
    const rest = line.slice(quotes.length)
    const head = /^\[!(\w+)\|latex\][+-]?\s*(.*)$/.exec(rest)
    if (head && depth > 0) {
      const kind = head[1].toLowerCase()
      const [, number = '', name = ''] = /^([\d.]+[a-z]?)?\s*(.*)$/i.exec(head[2].trim()) ?? []
      pending.set(depth, STATEMENT_KINDS.has(kind) ? { kind, number, name } : null)
      continue
    }
    const anchor = /^\^([\w-]+)\s*$/.exec(rest)?.[1]
    const open = pending.get(depth + 1)
    if (anchor && open && anchor.startsWith(`${open.kind}-`)) {
      out.push({ ...open, unit, slug: file.slug as FullSlug, anchor })
      pending.set(depth + 1, null)
    }
  }
  return out
}

/** `**Name.**` at the start of a paragraph in an authored note: a term, often a definition. */
function boldLeads(file: QuartzPluginData, unit: number): Omit<Statement, 'cards' | 'usedBy'>[] {
  return Array.from(body(file).matchAll(BOLD_LEAD), match => ({
    kind: 'term',
    number: '',
    name: match[1],
    unit,
    slug: file.slug as FullSlug,
    anchor: termAnchor(match[1]),
  }))
}

/**
 * Every statement of a course: numbered callouts in the generated `notes/<nn>` pages, bold-lead
 * definitions in the authored `lectures/<nn>` notes. `cards` counts deck cards that link to the
 * anchor (bold leads: cards in that unit's deck naming the term); `usedBy` lists pset problems
 * whose links reach the anchor.
 */
export function courseStatements(view: CourseView): Statement[] {
  const found: Omit<Statement, 'cards' | 'usedBy'>[] = view.units.flatMap(unit => [
    ...(unit.text ? calloutStatements(unit.text, unit.record.n) : []),
    ...(unit.note ? boldLeads(unit.note, unit.record.n) : []),
  ])
  // card blocks of each unit deck, split on the `---` separator of the `.fc` source
  const cards = view.units.flatMap(unit =>
    (unit.deck?.rawMarkdownSource ?? '')
      .split(/^---\s*$/m)
      .filter(raw => raw.trim())
      .map(raw => ({ unit: unit.record.n, raw })),
  )
  const uses = new Map<string, string[]>()
  for (const [n, text] of view.psetText)
    for (const problem of psetProblems(text))
      for (const use of problem.uses) {
        // a use the pset restates is the pset's own statement, not the lecture's
        if (use.slug) continue
        uses.set(use.anchor, [...(uses.get(use.anchor) ?? []), `pset ${n} #${problem.n}`])
      }
  return found.map(statement => {
    const anchor = statement.kind === 'term' ? undefined : statement.anchor
    const cited = anchor
      ? cards.filter(card => card.raw.includes(`${statement.slug}#^${anchor}`)).length
      : cards.filter(
          card =>
            card.unit === statement.unit &&
            card.raw.toLowerCase().includes(statement.name.toLowerCase()),
        ).length
    return { ...statement, cards: cited, usedBy: anchor ? (uses.get(anchor) ?? []) : [] }
  })
}

/** `courses/<dir>/statements/index.md`: the course's statements in lecture order. */
export default (() => {
  const CourseStatements: QuartzComponent = (props: QuartzComponentProps) => {
    const { ctx, fileData, allFiles, displayClass } = props
    const from = fileData.slug!
    const dir = courseDirOf(from)
    const view = dir ? courseOf(dir, allFiles, ctx.decks).view : null
    if (!view) return null
    const statements = courseStatements(view)
    const kinds = new Map<string, number>()
    for (const statement of statements)
      kinds.set(statement.kind, (kinds.get(statement.kind) ?? 0) + 1)
    const carded = statements.filter(statement => statement.cards > 0).length
    const title = String(fileData.frontmatter?.title ?? `${view.record.number} statements`)
    return (
      <div class={classNames(displayClass, 'course-home', 'course-statements')}>
        <header class="course-masthead" data-course-card>
          <p class="course-masthead-num course-num course-crumb">
            <CoursesCrumb from={from} />
            <a class="internal" href={resolveRelative(from, view.slug)} data-slug={view.slug}>
              {view.record.number}
            </a>
            <span class="course-unit-sep">/</span>
            <span>statements</span>
          </p>
          <h1 class="course-masthead-title course-card-title">{title}</h1>
          <p class="course-count">
            {Array.from(kinds, ([kind, count]) => `${kind} ${count}`).join(' · ') || 'none yet'}
            {statements.length > 0 ? ` · on cards ${carded}/${statements.length}` : ''}
          </p>
        </header>
        {statements.length === 0 ? (
          <p class="course-count">
            no statements yet: numbered callouts in the notes and bold-lead terms in{' '}
            <code>lectures/&lt;nn&gt;.md</code> land here.
          </p>
        ) : (
          <div class="course-desk">
            <table class="course-spine-table" data-course-rows>
              <thead>
                <tr>
                  <th scope="col" class="course-c-lec">
                    {view.record.unitKind.slice(0, 3)}
                  </th>
                  <th scope="col" class="course-c-kind">
                    kind
                  </th>
                  <th scope="col">statement</th>
                  <th
                    scope="col"
                    class="course-c-cards"
                    title="cards that link to the statement; for a bold lead, cards in that unit's deck naming it"
                  >
                    cards
                  </th>
                  <th scope="col" class="course-c-used">
                    used by
                  </th>
                </tr>
              </thead>
              <tbody>
                {statements.map(statement => {
                  const label = [statement.number, statement.name].filter(Boolean).join(' ')
                  const href = `${resolveRelative(from, statement.slug)}#${statement.anchor}`
                  return (
                    <tr
                      class="course-unit-row"
                      id={statementRowId(statement.unit, statement.anchor)}
                    >
                      <td class="course-num course-c-lec">{pad(statement.unit)}</td>
                      <td class="course-c-kind">{statement.kind}</td>
                      <td>
                        <a
                          class="internal course-unit-title"
                          href={href}
                          data-slug={statement.slug}
                        >
                          <MathText text={label || statement.kind} />
                        </a>
                      </td>
                      <td
                        class={statement.cards ? 'course-c-cards' : 'course-c-cards course-muted'}
                      >
                        {statement.cards || '–'}
                      </td>
                      <td
                        class={
                          statement.usedBy.length ? 'course-c-used' : 'course-c-used course-muted'
                        }
                      >
                        {statement.usedBy.join(', ') || '–'}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    )
  }
  CourseStatements.css = concatenateResources(style, studyStyle)
  return CourseStatements
}) satisfies QuartzComponentConstructor
