import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import type { StatementText } from './vendor-ocw-latex/converter-model'
import {
  compileFigures,
  walk,
  which,
  type FigureJob,
  type ZipSources,
} from './vendor-ocw-latex/artifacts'
import { convertLatexDocument } from './vendor-ocw-latex/converter'
import {
  convertFullLectureNotes,
  flow,
  flowMap,
  frontmatterYaml,
  pad,
  problemSetNumber,
  problemSetNumberRe,
  yamlQuote,
  type CourseDraft,
  type LatexCourseInput,
  type LatexNoteSummary,
  type LatexResource,
} from './vendor-ocw-latex/course-notes'
import { LatexGateError } from './vendor-ocw-latex/parser'
import { scanPreamble, type Preamble } from './vendor-ocw-latex/preamble'
export { LatexGateError } from './vendor-ocw-latex/parser'
export { convertLatexDocument } from './vendor-ocw-latex/converter'
export { pageOfLine, parseAux, parseSynctex, parseToc } from './vendor-ocw-latex/artifacts'
export { parseLatex } from './vendor-ocw-latex/parser'
export { expandMacros, scanPreamble, splitRows, stripComments } from './vendor-ocw-latex/preamble'

export interface LatexResult {
  notes: LatexNoteSummary[]
  psets: LatexNoteSummary[]
  figures: number
  report: string[]
  /** Resource relDir → the note slug its pointer page links. */
  pointers: Map<string, string>
}

/** Normalized token bigrams of a statement body; spelling variants of one symbol collapse. */
function statementBigrams(text: string): Set<string> {
  const words = Array.from(
    text
      .replace(/\uE000REF\d+\uE000/g, ' ')
      .replace(/\\\|/g, '||')
      .replace(/\\(?:le|leq)\b/g, '≤')
      .replace(/\\(?:ge|geq)\b/g, '≥')
      .replace(/\\rightarrow\b/g, '\\to')
      .replace(/\\[dt]frac\b/g, '\\frac')
      .replace(/\\(?:left|right|big|Big|bigg|Bigg)\b/g, '')
      .matchAll(/\\[A-Za-z]+|[A-Za-z]+|[^\sA-Za-z0-9{}$>]/g),
    match => match[0].toLowerCase(),
  )
  const out = new Set<string>()
  for (let i = 0; i + 1 < words.length; i++) out.add(`${words[i]} ${words[i + 1]}`)
  return out
}

function setJaccard(left: Set<string>, right: Set<string>): number {
  let shared = 0
  for (const item of left) if (right.has(item)) shared++
  return shared === 0 ? 0 : shared / (left.size + right.size - shared)
}

/**
 * Pairs each pset statement with the notes statement it restates. On 18.156 restatements score
 * 0.46 to 0.97 and the best wrong candidate 0.36, so 0.4 separates them. Pairs are one to one,
 * best score first, so a pset variant beside its restatement stays the pset's own; the notes
 * repeat some statements word for word, and only the first copy is a candidate.
 */
function restatementMatcher(
  notes: StatementText[],
  noteSlug: (section: number) => string,
  report: string[],
  source: string,
): (statements: StatementText[]) => Map<string, string> {
  const seen = new Set<string>()
  const candidates = notes.flatMap(statement => {
    const key = statementBigrams(statement.text)
    const fingerprint = Array.from(key).sort().join('\n')
    if (seen.has(fingerprint)) return []
    seen.add(fingerprint)
    return [{ statement, key }]
  })
  return statements => {
    const pairs = statements
      .flatMap(statement => {
        const key = statementBigrams(statement.text)
        return candidates.map(candidate => ({
          statement,
          note: candidate.statement,
          score: setJaccard(key, candidate.key),
        }))
      })
      .filter(pair => pair.score >= 0.4)
      .sort((a, b) => b.score - a.score)
    const out = new Map<string, string>()
    const taken = new Set<string>()
    for (const pair of pairs) {
      if (out.has(pair.statement.anchor) || taken.has(pair.note.anchor)) continue
      out.set(pair.statement.anchor, `${noteSlug(pair.note.section)}#^${pair.note.anchor}`)
      taken.add(pair.note.anchor)
      report.push(
        `${source}: ${pair.statement.anchor} restates ${pair.note.anchor} (${pair.score.toFixed(2)})`,
      )
    }
    return out
  }
}

const lectureNumberRe = /\blec(?:ture)?[\s_#]*0*(\d+)\b/i

async function convertProblemSets(draft: CourseDraft): Promise<void> {
  const input = draft.input
  const report = draft.report
  const pointers = draft.pointers
  const notesDir = draft.notesDir
  const noteSlug = draft.noteSlug
  const psets = draft.psets
  const files = draft.files
  const psetNotes = draft.psetNotes
  const figureJobs = draft.figureJobs
  const noteStatements = draft.noteStatements

  for (const resource of psets) {
    const n = problemSetNumber(resource)!
    const sourcePath = path.join(input.root, 'static_resources', resource.localFile)
    const source = await fs.readFile(sourcePath, 'utf8')
    const slug = `${input.slug}/notes/pset-${pad(n)}`
    const doc = convertLatexDocument(source, {
      mode: 'pset',
      noteSlug: () => slug,
      restates: restatementMatcher(noteStatements, noteSlug, report, resource.localFile),
    })
    report.push(...doc.report.map(entry => `${resource.localFile}: ${entry}`))
    if (doc.katexErrors.length > 0) {
      throw new LatexGateError(
        `${resource.localFile}: KaTeX refused ${doc.katexErrors.length} expression(s):\n  ${doc.katexErrors.slice(0, 12).join('\n  ')}`,
      )
    }
    const section = doc.sections[0]
    if (!section) continue
    let problem = 0
    const markdown = section.markdown
      .split('\n\n')
      .map(block => {
        const match = /^\\?(\d+)\.\s+([\s\S]*)$/.exec(block)
        if (!match) return block
        problem = Number(match[1])
        return `**problem ${problem}.** ${match[2]}\n\n^problem-${problem}`
      })
      .join('\n\n')
    // the heading line `{\bf 18.156, Projection theory, problem set 1}` duplicates the title
    const body = markdown.replace(/^\*\*[^*\n]*problem set \d+\*\*\n\n/i, '')
    const head = frontmatterYaml([
      ['title', yamlQuote(`${input.number} problem set ${pad(n)}`)],
      [
        'description',
        yamlQuote(section.description || `${input.number} problem set ${pad(n)}, as assigned`),
      ],
      ['tags', flow(['course', 'mit', 'ocw', 'problem set'])],
      ['cssclasses', flow(['course-notes'])],
      ['license', yamlQuote(input.licenseName)],
      ['license_url', yamlQuote(input.licenseUrl)],
      [
        'latex',
        flowMap([
          ['source', `static_resources/${resource.localFile}`],
          ['lines', [section.startLine, section.endLine]],
          ['pset', n],
          [
            'pdf',
            (() => {
              const pdf = input.resources.find(
                r =>
                  /\.pdf$/i.test(r.localFile) &&
                  problemSetNumber(r) === n &&
                  !/sol/i.test(r.localFile),
              )
              return pdf ? path.basename(pdf.relDir) : ''
            })(),
          ],
          ['unhandled', section.unhandled.length > 0 ? section.unhandled : undefined],
        ]),
      ],
    ])
    files.set(path.join(notesDir, `pset-${pad(n)}.md`), `${head}\n\n${body}\n`)
    psetNotes.push({
      slug,
      n,
      title: `problem set ${pad(n)}`,
      date: '',
      pages: null,
      statements: section.statements,
    })
    pointers.set(resource.relDir, slug)
    for (const figure of section.figures) figureJobs.push({ ...figure, section })
  }
}

export async function convertLatexCourse(input: LatexCourseInput): Promise<LatexResult | null> {
  const texResources = input.resources.filter(resource => /\.tex$/i.test(resource.localFile))
  if (texResources.length === 0) return null
  const report: string[] = []
  const pointers = new Map<string, string>()
  const notesDir = path.join(input.root, 'notes')
  const latexDir = path.join(input.root, 'static_resources', 'latex')
  const noteSlug = (n: number) => `${input.slug}/notes/${pad(n)}`

  const full = texResources.find(
    resource =>
      resource.resourceTypes.includes('Lecture Notes') && !problemSetNumberRe.test(resource.title),
  )
  const psets = texResources.filter(resource => problemSetNumber(resource) !== null)

  // staged output: nothing is written until every gate passes
  const files = new Map<string, string>()
  const notes: LatexNoteSummary[] = []
  const psetNotes: LatexNoteSummary[] = []
  let figureCount = 0
  const figureJobs: FigureJob[] = []
  let zip: ZipSources | null = null
  const year = /\b(\d{4})\b/.exec(input.term)?.[1] ?? ''
  const lecturer =
    (input.instructors[0] ?? '')
      .replace(/^prof\.?\s*/i, '')
      .split(/\s+/)
      .at(-1) ?? ''

  const pdfResource = (n: number | 'full') =>
    input.resources.find(
      resource =>
        /\.pdf$/i.test(resource.localFile) &&
        (n === 'full'
          ? /full/i.test(resource.localFile) && /lec/i.test(resource.localFile)
          : lectureNumber(resource) === n && !/transcript|full/i.test(resource.localFile)),
    )

  let preambleForFigures: Preamble | null = null
  let noteStatements: StatementText[] = []
  const courseDraft: CourseDraft = {
    input,
    report,
    pointers,
    notesDir,
    noteSlug,
    full,
    psets,
    files,
    notes,
    psetNotes,
    figureJobs,
    zip,
    pdfResource,
    year,
    preambleForFigures,
    noteStatements,
  }
  await convertFullLectureNotes(courseDraft)
  zip = courseDraft.zip
  preambleForFigures = courseDraft.preambleForFigures
  noteStatements = courseDraft.noteStatements
  await convertProblemSets(courseDraft)

  // figures
  if (figureJobs.length > 0) {
    const preamble = preambleForFigures ?? scanPreamble([])
    const haveTools = (await which('latex')) && (await which('dvisvgm'))
    if (haveTools) await fs.rm(latexDir, { recursive: true, force: true })
    await fs.mkdir(latexDir, { recursive: true })
    const results = await compileFigures(figureJobs, zip, preamble, latexDir, report)
    figureCount = Array.from(results.values()).filter(value => value !== null).length
    for (const [file, content] of files) {
      files.set(
        file,
        content.replace(
          /\uE001FIG:(input|inline|graphics):([^:]+):(\d+)\uE001/g,
          (_, _kind: string, name: string, line: string) => {
            const output = results.get(name)
            if (output) return `![[${input.slug}/static_resources/latex/${output}]]`
            return `_(figure \`${name}\` is not in the source; see the pdf, source line ${line}.)_`
          },
        ),
      )
    }
  }

  // dangling anchors: existing content linking notes/<nn>#^id must still resolve
  const defined = new Map<string, Set<string>>()
  for (const [file, content] of files) {
    const slug = `${input.slug}/notes/${path.basename(file, '.md')}`
    // anchors inside nested callouts carry a `> ` prefix per level
    defined.set(
      slug,
      new Set(Array.from(content.matchAll(/^(?:> )*\^([-A-Za-z0-9]+)$/gm), match => match[1])),
    )
    for (const match of content.matchAll(/<a id="([^"]+)" class="latex-eq">/g))
      defined.get(slug)!.add(match[1])
  }
  const dangling: string[] = []
  const linkRe = new RegExp(
    `\\[\\[(${escapeRegExp(input.slug)}/notes/[^\\]|#]+)#\\^?([-A-Za-z0-9]+)`,
    'g',
  )
  const resolvedNotesDir = path.resolve(notesDir)
  for (const file of await walk(path.resolve(input.contentRoot), name => /\.(md|fc)$/.test(name))) {
    const text = await fs.readFile(file, 'utf8')
    // generated notes are replaced by this run; authored files under notes/ still count
    if (file.startsWith(resolvedNotesDir) && /^latex:|^resource: \{kind: 'notes-index'/m.test(text))
      continue
    for (const match of text.matchAll(linkRe)) {
      const slug = match[1].replace(/\/$/, '')
      const anchors = defined.get(slug)
      if (anchors && !anchors.has(match[2]))
        dangling.push(`${path.relative(input.contentRoot, file)} → ${slug}#^${match[2]}`)
    }
  }
  if (dangling.length > 0) {
    throw new LatexGateError(
      `anchors referenced by existing notes are no longer defined:\n  ${dangling.join('\n  ')}`,
    )
  }

  // index page
  if (notes.length > 0) {
    const rows = notes.map(note => {
      const pages = note.pages ? `${note.pages[0]}–${note.pages[1]}` : '–'
      return `| ${pad(note.n)} | [[${note.slug}\\|${note.title}]] | ${note.date || '–'} | ${pages} | ${note.statements} |`
    })
    const psetRows = psetNotes.map(note => `- [[${note.slug}|problem set ${pad(note.n)}]]`)
    const head = frontmatterYaml([
      ['title', yamlQuote(`${input.number} notes`)],
      [
        'description',
        yamlQuote(
          `${lecturer ? `${lecturer}'s` : 'the'} lecture notes for ${input.number}, one page per lecture, converted from the OCW LaTeX source`,
        ),
      ],
      ['tags', flow(['course', 'mit', 'ocw', 'lecture notes'])],
      ['layout', 'resource'],
      ['resource', flowMap([['kind', 'notes-index']])],
      ['license', yamlQuote(input.licenseName)],
      ['license_url', yamlQuote(input.licenseUrl)],
    ])
    files.set(
      path.join(notesDir, 'index.md'),
      [
        head,
        '',
        `up: [[${input.slug}/|${input.number}]]`,
        '',
        '| # | lecture | taught | pp. | statements |',
        '| --- | --- | --- | --- | --- |',
        ...rows,
        '',
        ...(psetRows.length > 0 ? ['## problem sets', '', ...psetRows, ''] : []),
      ].join('\n'),
    )
  }

  // write: drop vendored notes first (authored files carry no `latex:` block)
  await fs.mkdir(notesDir, { recursive: true })
  for (const entry of await fs.readdir(notesDir)) {
    if (!entry.endsWith('.md')) continue
    const file = path.join(notesDir, entry)
    const text = await fs.readFile(file, 'utf8')
    if (/^latex:|^resource: \{kind: 'notes-index'/m.test(text)) await fs.rm(file)
  }
  for (const [file, content] of files) await fs.writeFile(file, content, 'utf8')
  if (zip)
    await fs
      .rm(path.dirname(zip.dir) === os.tmpdir() ? zip.dir : path.resolve(zip.dir, '..'), {
        recursive: true,
        force: true,
      })
      .catch(() => {})

  return { notes, psets: psetNotes, figures: figureCount, report, pointers }
}

function lectureNumber(resource: LatexResource): number | null {
  const match =
    path.posix.basename(resource.relDir).match(lectureNumberRe) ??
    resource.title.match(lectureNumberRe)
  return match ? Number(match[1]) : null
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}
