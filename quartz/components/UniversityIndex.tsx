import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../types/component'
import { BuildCtx } from '../util/ctx'
import { classNames } from '../util/lang'
import { FullSlug, resolveRelative } from '../util/path'
import style from './styles/university.scss'

const root = 'thoughts/university/'
const termIndexRe = /^thoughts\/university\/([^/]+)\/index$/
const homeRe = /^thoughts\/university\/([^/]+)\/([^/]+)\/index$/

interface Counts {
  notes: number
  pdfs: number
  notebooks: number
}

const emptyCounts = (): Counts => ({ notes: 0, pdfs: 0, notebooks: 0 })

// `twenty-five-twenty-six` -> `2025–26`
const numberWords: Record<string, number> = { three: 3, four: 4, five: 5, six: 6, seven: 7 }
function termLabel(dir: string): string {
  const [a, b] = dir.split('-twenty-').map(part => numberWords[part.replace(/^twenty-/, '')])
  return a && b ? `20${20 + a}–${20 + b}` : dir
}

// `sfwr-4ml3` -> `SFWR 4ML3`
const courseCode = (dir: string): string => dir.replace('-', ' ').toUpperCase()

let counted: { allFiles: BuildCtx['allFiles']; byCourse: Map<string, Counts> } | undefined

// Notes, PDFs and notebooks per course, counted from every source path: PDFs and
// notebooks are not in the markdown file list.
function countsByCourse(ctx: BuildCtx): Map<string, Counts> {
  if (counted?.allFiles === ctx.allFiles) return counted.byCourse
  const byCourse = new Map<string, Counts>()
  for (const fp of ctx.allFiles) {
    const rel = String(fp)
    if (!rel.startsWith('content/' + root) && !rel.startsWith(root)) continue
    const parts = rel.slice(rel.indexOf(root) + root.length).split('/')
    if (parts.length < 3 || parts.some(part => part === '.ipynb_checkpoints')) continue
    const key = `${parts[0]}/${parts[1]}`
    const counts = byCourse.get(key) ?? emptyCounts()
    const name = parts[parts.length - 1]
    const ext = name.slice(name.lastIndexOf('.')).toLowerCase()
    if (ext === '.md' && !(parts.length === 3 && name === 'index.md')) counts.notes++
    else if (ext === '.pdf') counts.pdfs++
    else if (ext === '.ipynb') counts.notebooks++
    byCourse.set(key, counts)
  }
  counted = { allFiles: ctx.allFiles, byCourse }
  return byCourse
}

const sum = (list: Counts[]): Counts =>
  list.reduce(
    (acc, c) => ({
      notes: acc.notes + c.notes,
      pdfs: acc.pdfs + c.pdfs,
      notebooks: acc.notebooks + c.notebooks,
    }),
    emptyCounts(),
  )

const stats = (c: Counts): string =>
  [`notes ${c.notes}`, `pdf ${c.pdfs}`, c.notebooks > 0 ? `notebooks ${c.notebooks}` : '']
    .filter(Boolean)
    .join(' · ')

export default (() => {
  const UniversityIndex: QuartzComponent = ({
    ctx,
    fileData,
    allFiles,
    displayClass,
  }: QuartzComponentProps) => {
    const from = fileData.slug!
    const counts = countsByCourse(ctx)
    const terms = new Map<string, { home: (typeof allFiles)[number]; dir: string }[]>()
    // a term page (`thoughts/university/<term>/`) lists only its own courses
    const scope = termIndexRe.exec(from)?.[1]
    for (const home of allFiles) {
      const match = homeRe.exec(home.slug ?? '')
      if (!match || (scope && match[1] !== scope)) continue
      const list = terms.get(match[1]) ?? []
      list.push({ home, dir: match[2] })
      terms.set(match[1], list)
    }
    // newest term first: the folder names sort by year only through their words, so use the label
    const ordered = [...terms.entries()].sort(([a], [b]) =>
      termLabel(b).localeCompare(termLabel(a)),
    )
    const total = sum([...counts.values()])
    const courseTotal = [...terms.values()].reduce((n, list) => n + list.length, 0)
    const description = String(fileData.frontmatter?.description ?? '')

    return (
      <div class={classNames(displayClass, 'course-home', 'course-home--index')}>
        <section class="course-index university-index" aria-labelledby="university-h">
          <header class="course-bar">
            <h2 id="university-h">
              {scope
                ? termLabel(scope)
                : `${courseTotal} ${courseTotal === 1 ? 'course' : 'courses'}`}
            </h2>
            <p class="course-count">
              {scope
                ? `${courseTotal} ${courseTotal === 1 ? 'course' : 'courses'} · `
                : `${terms.size} terms · `}
              {stats(
                scope
                  ? sum([...counts].filter(([k]) => k.startsWith(`${scope}/`)).map(([, c]) => c))
                  : total,
              )}
            </p>
          </header>
          {description && <p class="university-lede">{description}</p>}
          {ordered.map(([term, list]) => {
            const label = termLabel(term)
            const termCounts = sum(
              list.map(({ dir }) => counts.get(`${term}/${dir}`) ?? emptyCounts()),
            )
            list.sort((a, b) => a.dir.localeCompare(b.dir, undefined, { numeric: true }))
            return (
              <section class="university-term" aria-labelledby={`term-${term}`}>
                {!scope && (
                  <header class="course-bar">
                    <h3 id={`term-${term}`}>{label}</h3>
                    <p class="course-count">
                      {list.length} {list.length === 1 ? 'course' : 'courses'} · {stats(termCounts)}
                    </p>
                  </header>
                )}
                <ol class="course-index-list">
                  {list.map(({ home, dir }) => (
                    <li class="course-row">
                      <a
                        class="course-row-link internal"
                        href={resolveRelative(from, home.slug as FullSlug)}
                        data-slug={home.slug}
                      >
                        <span class="course-num">{courseCode(dir)}</span>
                        <span class="course-row-title">
                          {String(home.frontmatter?.title ?? dir)}
                        </span>
                        <span class="course-stats">
                          {stats(counts.get(`${term}/${dir}`) ?? emptyCounts())}
                        </span>
                      </a>
                    </li>
                  ))}
                </ol>
              </section>
            )
          })}
        </section>
      </div>
    )
  }
  UniversityIndex.css = style
  return UniversityIndex
}) satisfies QuartzComponentConstructor
