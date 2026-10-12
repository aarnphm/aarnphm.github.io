import { JSX } from 'preact'
import { QuartzPluginData } from '../plugins/vfile'
import { CourseResourceRef, CourseView, psetRefs } from '../util/course'
import { ProblemRecord, ProblemState, parseProblems } from '../util/course-study'
import { MathText } from '../util/math-text'
import { FullSlug, resolveRelative } from '../util/path'

const pad = (n: number) => String(n).padStart(2, '0')
const states: ProblemState[] = ['open', 'stuck', 'solved']

export interface DeskUse {
  anchor: string
  label: string
  /** The anchor on the pset page itself, when the statement is restated there. */
  slug?: FullSlug
}

export interface DeskProblem {
  n: number
  title: string
  uses: DeskUse[]
  /** The `problem-<n>` block on the page the problems come from: the row links to its statement. */
  statement?: { slug: FullSlug; anchor: string }
}

/** `theorem-3` → `theorem 3`, `proposition-2-1` → `proposition 2.1`. */
export function anchorLabel(anchor: string): string {
  const [kind, ...rest] = anchor.split('-')
  return rest.length > 0 ? `${kind} ${rest.join('.')}` : kind
}

function leadTitle(text: string): string {
  const flat = text
    .replace(/\[\[[^\]|]*\|([^\]]*)\]\]/g, '$1')
    .replace(/\s+/g, ' ')
    .trim()
  const named = /^\(([^)]{2,80})\)/.exec(flat)
  if (named) return named[1]
  const sentence = /^(.{1,90}?[.?!:])(\s|$)/.exec(flat)?.[1] ?? flat
  return sentence.length > 90 ? `${sentence.slice(0, 89)}…` : sentence
}

/**
 * Problems of one set from its generated `notes/pset-<nn>` page (`**problem N.**` leads) or from
 * the `## N. title` sections of `psets/<nn>.md`. `uses` are the statement links inside each
 * problem. The LaTeX converter points `\ref` at the pset's own statement, or at the notes statement
 * it restates; an anchor on the pset page links there, anything else keeps its label without a link.
 */
export function psetProblems(text: QuartzPluginData | undefined): DeskProblem[] {
  const source = text?.rawMarkdownSource ?? ''
  const slug = text?.slug as FullSlug | undefined
  const body = source.replace(/^---\n[\s\S]*?\n---\n/, '')
  const own = new Set(Array.from(body.matchAll(/^\^([\w-]+)\s*$/gm), match => match[1]))
  const starts = Array.from(body.matchAll(/^\*\*problem (\d+)\.\*\*\s*(.*)$/gim), match => ({
    n: Number(match[1]),
    title: leadTitle(match[2]),
    at: match.index ?? 0,
  }))
  const heads =
    starts.length > 0
      ? starts
      : Array.from(body.matchAll(/^##\s+(?:problem\s+)?(\d+)\b\.?\s*(.*)$/gim), match => ({
          n: Number(match[1]),
          // a section heading is already a title; only a long one is cut to its lead
          title: match[2].length <= 90 ? match[2].trim() : leadTitle(match[2]),
          at: match.index ?? 0,
        }))
  if (heads.length === 0) return []
  return heads.map((head, i) => {
    const span = body.slice(head.at, heads[i + 1]?.at ?? body.length)
    const uses = new Map<string, DeskUse>()
    for (const match of span.matchAll(/\[\[[^\]|#]*#\^([\w-]+)(?:\|[^\]]*)?\]\]/g)) {
      const anchor = match[1]
      if (anchor.startsWith('problem-') || uses.has(anchor)) continue
      uses.set(anchor, {
        anchor,
        label: anchorLabel(anchor),
        ...(own.has(anchor) && slug ? { slug } : {}),
      })
    }
    const anchor = `problem-${head.n}`
    return {
      n: head.n,
      title: head.title,
      uses: Array.from(uses.values()),
      ...(own.has(anchor) && slug ? { statement: { slug, anchor } } : {}),
    }
  })
}

/** `solutionTerm:` in `psets/<nn>.md`: the term an untitled official solution belongs to. */
export function solutionTerm(pset: QuartzPluginData | undefined): string | undefined {
  const term = pset?.frontmatter?.solutionTerm
  return typeof term === 'string' || typeof term === 'number' ? String(term) : undefined
}

function stateLog(record: ProblemRecord | undefined): string {
  if (!record) return '–'
  const parts: string[] = []
  if (record.state !== 'open') parts.push(`${record.state}${record.date ? ` ${record.date}` : ''}`)
  if (record.reveal) parts.push(`revealed ${record.reveal}`)
  return parts.join(' · ') || '–'
}

/**
 * The problem set desk: one row per problem with its statement links, its state and the day the
 * official solution was opened. States live in `psets/<nn>.md` (`problems:`); in dev the row
 * buttons post to the study endpoint, elsewhere the states are text.
 */
export function CourseDesk({
  view,
  n,
  from,
  resources,
  endpoint,
}: {
  view: CourseView
  n: number
  from: FullSlug
  resources: Map<string, CourseResourceRef>
  endpoint?: string
}): JSX.Element | null {
  const problems = psetProblems(view.psetText.get(n) ?? view.psets.get(n))
  if (problems.length === 0) return null
  const log = parseProblems(view.psets.get(n)?.frontmatter)
  const refs = psetRefs(resources, n, solutionTerm(view.psets.get(n)))
  const solution = refs.solution
  const link = (ref: CourseResourceRef, text: string) => (
    <a class="internal" href={resolveRelative(from, ref.slug)} data-slug={ref.slug}>
      {text}
    </a>
  )
  const sources: JSX.Element[] = []
  if (refs.pset) sources.push(link(refs.pset, 'set pdf'))
  if (solution) sources.push(link(solution, 'solutions'))
  for (const other of refs.other) sources.push(link(other.solution, `${other.term} solutions`))
  const solved = problems.filter(problem => log.get(problem.n)?.state === 'solved').length
  const revealed = problems.filter(problem => log.get(problem.n)?.reveal).length
  const file = `psets/${pad(n)}.md`
  return (
    <section
      class="course-desk"
      aria-labelledby={`course-desk-${n}`}
      data-desk-endpoint={endpoint}
      data-desk-dir={view.dir}
      data-desk-n={String(n)}
    >
      <div class="course-desk-head">
        <h2 id={`course-desk-${n}`}>desk</h2>
        <span>
          solved {solved}/{problems.length} · revealed {revealed} · states in{' '}
          <code class="course-path">{file}</code>
        </span>
        {sources.length > 0 ? (
          <span class="course-desk-sources">
            {sources.map((source, i) => (
              <>
                {i > 0 ? ' · ' : ''}
                {source}
              </>
            ))}
          </span>
        ) : null}
        <span class="course-study-status" role="status" aria-live="polite" />
      </div>
      <table class="course-spine-table">
        <thead>
          <tr>
            <th scope="col" class="course-c-n">
              #
            </th>
            <th scope="col">problem</th>
            <th scope="col" class="course-c-uses">
              uses
            </th>
            <th scope="col" class="course-c-state">
              state
            </th>
            <th scope="col" class="course-c-log">
              log
            </th>
          </tr>
        </thead>
        <tbody>
          {problems.map(problem => {
            const record = log.get(problem.n)
            const state = record?.state ?? 'open'
            const href = solution ? resolveRelative(from, solution.slug) : undefined
            return (
              <tr class="course-unit-row" data-desk-problem={String(problem.n)}>
                <td class="course-num course-c-n">{pad(problem.n)}</td>
                <td>
                  {problem.statement ? (
                    <a
                      class="internal course-unit-title"
                      href={`${resolveRelative(from, problem.statement.slug)}#${problem.statement.anchor}`}
                      data-slug={problem.statement.slug}
                    >
                      <MathText text={problem.title} />
                    </a>
                  ) : (
                    <span class="course-unit-title">
                      <MathText text={problem.title} />
                    </span>
                  )}
                </td>
                <td class="course-c-uses">
                  {problem.uses.length === 0
                    ? '–'
                    : problem.uses.map((use, i) => (
                        <>
                          {i > 0 ? ', ' : ''}
                          {use.slug ? (
                            <a
                              class="internal"
                              href={`${resolveRelative(from, use.slug)}#${use.anchor}`}
                              data-slug={use.slug}
                            >
                              {use.label}
                            </a>
                          ) : (
                            <span>{use.label}</span>
                          )}
                        </>
                      ))}
                </td>
                <td class="course-c-state">
                  {endpoint ? (
                    <span
                      class="course-desk-states"
                      role="group"
                      aria-label={`problem ${problem.n}`}
                    >
                      {states.map(value => (
                        <button
                          type="button"
                          class="course-desk-toggle"
                          aria-pressed={state === value ? 'true' : 'false'}
                          data-desk-state={value}
                        >
                          {value}
                        </button>
                      ))}
                    </span>
                  ) : (
                    state
                  )}
                </td>
                <td class="course-c-log">
                  <span data-desk-log>{stateLog(record)}</span>
                  {href ? (
                    <div class="course-desk-reveal">
                      {endpoint ? (
                        <button
                          type="button"
                          class="course-desk-toggle"
                          data-desk-reveal={href}
                          title="log the reveal, then open the official solution"
                        >
                          reveal
                        </button>
                      ) : (
                        <a class="internal" href={href} data-slug={solution!.slug}>
                          solution
                        </a>
                      )}
                    </div>
                  ) : null}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </section>
  )
}
