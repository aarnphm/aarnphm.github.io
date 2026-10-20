import type { Root } from 'hast'
import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../types/component'
import { BuildCtx } from '../util/ctx'
import { htmlToJsx } from '../util/jsx'
import { classNames } from '../util/lang'
import { FilePath, FullSlug, resolveRelative, slugifyFilePath } from '../util/path'
import { normalizePdfSlug } from '../util/pdf-marks'
import { pdfTitlesFor } from '../util/pdf-titles'
import { concatenateResources } from '../util/resources'
import {
  UNIVERSITY_ROOT,
  UNIVERSITY_SLUG,
  courseCode,
  semesterOf,
  termLabel,
  universityHomeRe,
} from '../util/university'
import { universityPdfOf } from '../util/university-pdf'
import SeeAlsoConstructor from './SeeAlso'
import style from './styles/university-course.scss'

type Kind = 'note' | 'pdf' | 'ipynb'

interface Entry {
  kind: Kind
  slug: FullSlug
  title: string
  /** path below the group folder, shown when the file sits deeper than the group */
  sub: string
  date?: Date
}

interface Group {
  /** first folder under the course; empty for files at the course root */
  dir: string
  title: string
  slug?: FullSlug
  entries: Entry[]
}

const kindOf: Record<string, Kind> = { '.md': 'note', '.pdf': 'pdf', '.ipynb': 'ipynb' }
const kindLabel: Record<Kind, string> = { note: 'notes', pdf: 'pdf', ipynb: 'notebooks' }
const kinds: Kind[] = ['note', 'pdf', 'ipynb']

const skipped = (parts: string[]) =>
  parts.some(part => part === 'images' || part === '.ipynb_checkpoints' || part.startsWith('.'))

const collator = new Intl.Collator('en', { numeric: true, sensitivity: 'base' })
const byPath = (a: string, b: string) => collator.compare(a, b)

const anchor = (dir: string) =>
  'g-' +
  (dir
    ? dir
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, '-')
        .replace(/^-|-$/g, '')
    : 'root')

const str = (value: unknown): string => (typeof value === 'string' ? value : '')

/** Course files grouped by their first folder, built from every source path. */
function courseGroups(
  ctx: BuildCtx,
  allFiles: QuartzComponentProps['allFiles'],
  prefix: string,
): Group[] {
  const bySlug = new Map(allFiles.map(file => [file.slug, file]))
  const pdfTitles = pdfTitlesFor(allFiles)
  const groups = new Map<string, Group>()
  for (const fp of ctx.allFiles) {
    const rel = String(fp).replace(/^content\//, '')
    if (!rel.startsWith(prefix)) continue
    const parts = rel.slice(prefix.length).split('/')
    const name = parts[parts.length - 1]
    const kind = kindOf[name.slice(name.lastIndexOf('.')).toLowerCase()]
    if (!kind || skipped(parts)) continue
    const dir = parts.length > 1 ? parts[0] : ''
    let group = groups.get(dir)
    if (!group) {
      const index = dir
        ? bySlug.get(slugifyFilePath(`${prefix}${dir}/index.md` as FilePath))
        : undefined
      group = {
        dir,
        title: str(index?.frontmatter?.title) || dir || 'course',
        slug: index?.slug as FullSlug | undefined,
        entries: [],
      }
      groups.set(dir, group)
    }
    // the course home and folder indexes are the page and the group heading
    if (kind === 'note' && name === 'index.md' && parts.length <= 2) continue
    const file = slugifyFilePath(rel as FilePath, kind === 'ipynb')
    // a PDF opens in its reader page under `<course>/pdf/`
    const slug = ((kind === 'pdf' && universityPdfOf(file)?.pageSlug) || file) as FullSlug
    const md = kind === 'note' ? bySlug.get(file) : undefined
    if (kind === 'note' && (!md || md.frontmatter?.noindex === true)) continue
    const base = name.slice(0, name.lastIndexOf('.'))
    group.entries.push({
      kind,
      slug,
      title:
        str(md?.frontmatter?.title) ||
        (kind === 'pdf' ? pdfTitles.get(normalizePdfSlug(file) ?? file) : undefined) ||
        base,
      sub: parts.slice(dir ? 1 : 0, -1).join('/'),
      date: md?.dates?.created,
    })
  }
  for (const group of groups.values()) {
    group.entries.sort((a, b) => byPath(a.sub, b.sub) || byPath(a.title, b.title))
  }
  return [...groups.values()]
    .filter(group => group.entries.length > 0)
    .sort((a, b) => (a.dir === '' ? -1 : b.dir === '' ? 1 : byPath(a.dir, b.dir)))
}

const countOf = (entries: Entry[]) =>
  kinds
    .map(kind => [kind, entries.filter(entry => entry.kind === kind).length] as const)
    .filter(([, n]) => n > 0)
    .map(([kind, n]) => `${kindLabel[kind]} ${n}`)
    .join(' · ')

const dateFmt = new Intl.DateTimeFormat('en-CA', {
  year: 'numeric',
  month: 'short',
  day: 'numeric',
})

/** Whether the course page has authored body text beyond whitespace. */
const hasBody = (tree: QuartzComponentProps['tree']) =>
  (tree as Root).children.some(node => node.type !== 'text' || node.value.trim() !== '')

export default (() => {
  const SeeAlso = SeeAlsoConstructor()

  const UniversityCourse: QuartzComponent = (props: QuartzComponentProps) => {
    const { ctx, fileData, allFiles, tree, displayClass } = props
    const from = fileData.slug!
    const match = universityHomeRe.exec(from)
    if (!match) return null
    const [, term, dir] = match
    const prefix = `${UNIVERSITY_ROOT}${term}/${dir}/`
    const groups = courseGroups(ctx, allFiles, prefix)
    const all = groups.flatMap(group => group.entries)

    const fm: Record<string, unknown> = fileData.frontmatter ?? {}
    const course = (
      typeof fm.university === 'object' && fm.university ? fm.university : {}
    ) as Record<string, unknown>
    const summary = str(course.summary) || str(fm.description)
    const inferred = course.inferred === true
    const instructors = Array.isArray(course.instructors)
      ? course.instructors.map(str).filter(Boolean)
      : []
    const semester = str(course.term) || semesterOf(fm.date)
    const site = str((fm.socials as Record<string, unknown> | undefined)?.site)
    const meta = [semester, ...instructors].filter(Boolean)
    const universityHref = resolveRelative(from, UNIVERSITY_SLUG as FullSlug)

    return (
      <div class={classNames(displayClass, 'course-home', 'university-course')}>
        <header class="course-masthead">
          <p class="course-masthead-num course-num course-crumb">
            <a class="internal" href={universityHref} data-slug={UNIVERSITY_SLUG} data-no-popover>
              university
            </a>
            <span class="course-unit-sep">/</span>
            <a
              class="internal"
              href={`${universityHref}#term-${term}`}
              data-slug={UNIVERSITY_SLUG}
              data-no-popover
            >
              {termLabel(term)}
            </a>
            <span class="course-unit-sep">/</span>
            <span>{courseCode(dir)}</span>
          </p>
          <h1 class="course-masthead-title">{str(fm.title) || courseCode(dir)}</h1>
          <p class="course-masthead-meta">
            {meta.map((item, i) => (
              <>
                {i > 0 ? <span class="course-sep">·</span> : null}
                <span>{item}</span>
              </>
            ))}
            {meta.length > 0 ? <span class="course-sep">·</span> : null}
            <span>mcmaster university</span>
            {site ? (
              <>
                <span class="course-sep">·</span>
                <a href={site} target="_blank" rel="noopener noreferrer">
                  course site
                </a>
              </>
            ) : null}
          </p>
          {summary ? (
            <p class="course-masthead-desc">
              {summary}
              {inferred ? (
                <span class="university-course-inferred" title="no course outline in this folder">
                  summary inferred from the notes and files
                </span>
              ) : null}
            </p>
          ) : null}
        </header>
        <div class="course-body">
          <nav class="course-rail" aria-label="course sections">
            {groups.length > 1 ? (
              <section>
                <h2>sections</h2>
                <ul>
                  {groups.map(group => (
                    <li>
                      <a href={`#${anchor(group.dir)}`}>{group.title}</a>
                      <span class="course-count"> {group.entries.length}</span>
                    </li>
                  ))}
                </ul>
              </section>
            ) : null}
            <section aria-label="course metadata">
              <dl>
                <dt>files</dt>
                <dd>{countOf(all) || 'none'}</dd>
                <dt>archive</dt>
                <dd>
                  <a class="internal" href={universityHref} data-slug={UNIVERSITY_SLUG}>
                    all courses
                  </a>
                </dd>
              </dl>
            </section>
          </nav>
          <section class="course-main" aria-labelledby="university-course-h">
            <header class="course-bar">
              <h2 id="university-course-h">index</h2>
              <p class="course-count">{countOf(all)}</p>
            </header>
            {hasBody(tree) ? (
              <article class="university-course-body popover-hint">
                {htmlToJsx(fileData.filePath!, tree)}
              </article>
            ) : null}
            <table class="course-spine-table university-course-table">
              <thead>
                <tr>
                  <th scope="col" class="university-c-kind">
                    kind
                  </th>
                  <th scope="col">title</th>
                  <th scope="col" class="university-c-date">
                    date
                  </th>
                </tr>
              </thead>
              {groups.map(group => (
                <tbody class="course-part" id={anchor(group.dir)}>
                  {groups.length > 1 ? (
                    <tr class="course-part-head">
                      <th scope="rowgroup" colSpan={3}>
                        {group.slug ? (
                          <a
                            class="internal"
                            href={resolveRelative(from, group.slug)}
                            data-slug={group.slug}
                          >
                            {group.title}
                          </a>
                        ) : (
                          group.title
                        )}
                        <span class="course-count">{countOf(group.entries)}</span>
                      </th>
                    </tr>
                  ) : null}
                  {group.entries.map(entry => (
                    <tr class="course-unit-row" data-kind={entry.kind}>
                      <td class="university-c-kind">{entry.kind}</td>
                      <td>
                        <a
                          class={entry.kind === 'pdf' ? undefined : 'internal'}
                          href={resolveRelative(from, entry.slug)}
                          data-slug={entry.slug}
                        >
                          {entry.sub ? (
                            <span class="university-course-sub">{entry.sub}/</span>
                          ) : null}
                          <span class="course-unit-title">{entry.title}</span>
                        </a>
                      </td>
                      <td class="university-c-date">
                        {entry.date && entry.date.getTime() > 0 ? dateFmt.format(entry.date) : '–'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              ))}
            </table>
            <SeeAlso {...props} />
          </section>
        </div>
      </div>
    )
  }
  UniversityCourse.css = concatenateResources(style, SeeAlso.css)
  return UniversityCourse
}) satisfies QuartzComponentConstructor
