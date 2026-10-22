import fs from 'node:fs/promises'
import path from 'node:path'
import type { ConvertedSection, StatementText } from './converter-model'
import { extractZip, pageOfLine, run, which, type FigureJob, type ZipSources } from './artifacts'
import { convertLatexDocument } from './converter'
import { LatexGateError, parseLatex, type LatexNode } from './parser'
import { scanPreamble, stripComments, type Preamble } from './preamble'

// ---------------------------------------------------------------------------
// public contract

export interface LatexUnit {
  n: number
  title: string
}

export interface LatexResource {
  relDir: string
  title: string
  localFile: string
  resourceTypes: string[]
}

export interface LatexCourseInput {
  root: string
  slug: string
  number: string
  term: string
  instructors: string[]
  licenseName: string
  licenseUrl: string
  units: LatexUnit[]
  resources: LatexResource[]
  /** Markdown and flashcard files that may link `notes/<nn>#^id` anchors; checked by the dangling-anchor gate. */
  contentRoot: string
}

export interface LatexNoteSummary {
  slug: string
  n: number
  title: string
  date: string
  pages: [number, number] | null
  statements: number
}

export function yamlQuote(value: string): string {
  return `'${value.replace(/'/g, "''")}'`
}

function tokens(value: string): Set<string> {
  return new Set(
    value
      .toLowerCase()
      .replace(/\$[^$]*\$/g, ' ')
      .split(/[^a-z0-9]+/)
      .filter(word => word.length > 1),
  )
}

function jaccard(a: string, b: string): number {
  const left = tokens(a)
  const right = tokens(b)
  if (left.size === 0 && right.size === 0) return 1
  let shared = 0
  for (const word of left) if (right.has(word)) shared++
  return shared / (left.size + right.size - shared)
}

export const problemSetNumberRe = /\b(?:pset|problem set|ps)[\s_#]*0*(\d+)\b/i

export function pad(n: number): string {
  return String(n).padStart(2, '0')
}

export function frontmatterYaml(fields: [string, string][]): string {
  return ['---', ...fields.map(([key, value]) => `${key}: ${value}`), '---'].join('\n')
}

export function flowMap(entries: [string, unknown][]): string {
  return `{${entries
    .filter(([, value]) => value !== undefined && value !== null && value !== '')
    .map(([key, value]) => `${key}: ${flow(value)}`)
    .join(', ')}}`
}

export function flow(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(flow).join(', ')}]`
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  return yamlQuote(String(value))
}

export interface CourseDraft {
  input: LatexCourseInput
  report: string[]
  pointers: Map<string, string>
  notesDir: string
  noteSlug: (n: number) => string
  full: LatexResource | undefined
  psets: LatexResource[]
  files: Map<string, string>
  notes: LatexNoteSummary[]
  psetNotes: LatexNoteSummary[]
  figureJobs: FigureJob[]
  zip: ZipSources | null
  pdfResource: (n: number | 'full') => LatexResource | undefined
  year: string
  preambleForFigures: Preamble | null
  noteStatements: StatementText[]
}

export async function convertFullLectureNotes(draft: CourseDraft): Promise<void> {
  const input = draft.input
  const report = draft.report
  const pointers = draft.pointers
  const notesDir = draft.notesDir
  const noteSlug = draft.noteSlug
  const full = draft.full
  const year = draft.year
  const files = draft.files
  const notes = draft.notes
  const figureJobs = draft.figureJobs
  let zip = draft.zip
  const pdfResource = draft.pdfResource
  let preambleForFigures = draft.preambleForFigures
  let noteStatements = draft.noteStatements

  if (full) {
    const sourcePath = path.join(input.root, 'static_resources', full.localFile)
    const source = await fs.readFile(sourcePath, 'utf8')
    const zipResource = input.resources.find(resource => /\.zip$/i.test(resource.localFile))
    if (zipResource) {
      zip = await extractZip(
        path.join(input.root, 'static_resources', zipResource.localFile),
        source,
      )
      if (!zip)
        report.push(
          `zip ${zipResource.localFile} has no .tex matching ${full.localFile}; no aux/synctex gates`,
        )
    }
    const nodes = parseLatex(stripComments(source))
    preambleForFigures = scanPreamble(nodes)
    const sectionCount = countSections(nodes)
    if (sectionCount !== input.units.length) {
      throw new LatexGateError(
        `${full.localFile}: ${sectionCount} \\section commands for ${input.units.length} units`,
      )
    }
    const fullPdf = pdfResource('full')
    const pageLink = (line: number) => {
      if (!zip?.synctex || !zip.toc) return null
      const page = pageOfLine(zip.synctex, line - zip.lineOffset)
      if (page === null) return null
      return { page, href: '' }
    }
    const pending: { section: ConvertedSection; links: { line: number }[] }[] = []
    const doc = convertLatexDocument(source, {
      mode: 'notes',
      noteSlug,
      pageLink: line => {
        const page = pageLink(line)
        if (!page) return null
        return { page: page.page, href: `\uE002PAGE:${page.page}\uE002` }
      },
    })
    void pending
    report.push(...doc.report)
    noteStatements = doc.statements
    if (doc.katexErrors.length > 0) {
      throw new LatexGateError(
        `${full.localFile}: KaTeX refused ${doc.katexErrors.length} expression(s):\n  ${doc.katexErrors.slice(0, 12).join('\n  ')}`,
      )
    }
    const titleMismatches = doc.sections
      .map((section, index) => ({ section, unit: input.units[index] }))
      .filter(({ section, unit }) => jaccard(section.titlePlain, unit.title) < 0.5)
    if (titleMismatches.length > 0) {
      throw new LatexGateError(
        `${full.localFile}: section titles do not match the units:\n` +
          titleMismatches
            .map(
              ({ section, unit }) =>
                `  §${section.section} "${section.titlePlain}" vs unit ${unit.n} "${unit.title}"`,
            )
            .join('\n'),
      )
    }
    if (zip?.aux) {
      const mismatches: string[] = []
      for (const [key, entry] of zip.aux) {
        const ours = doc.labels.get(key)
        if (!ours) mismatches.push(`${key}: aux=${entry.value} ours=(missing)`)
        else if (ours.value !== entry.value)
          mismatches.push(`${key}: aux=${entry.value} ours=${ours.value}`)
      }
      if (mismatches.length > 0) {
        throw new LatexGateError(
          `${full.localFile}: numbering differs from the aux file:\n  ${mismatches.join('\n  ')}`,
        )
      }
      report.push(`aux gate: ${zip.aux.size} labels match`)
    }
    // pages per section and the lecture pdf page offsets
    const sectionPages = new Map<number, [number, number] | null>()
    const pageChecks = new Map<number, boolean>()
    if (zip?.toc) {
      const totalPages = zip.synctex ? Math.max(...zip.synctex.map(page => page.page)) : null
      for (const section of doc.sections) {
        const start = zip.toc.get(section.section)
        const next = zip.toc.get(section.section + 1)
        if (start === undefined) {
          sectionPages.set(section.section, null)
          continue
        }
        const end = next !== undefined ? next - 1 : (totalPages ?? start)
        sectionPages.set(section.section, [start, end])
      }
      const haveMutool = await which('mutool')
      for (const section of doc.sections) {
        const unit = input.units[section.section - 1]
        const lecturePdf = pdfResource(unit.n)
        const range = sectionPages.get(section.section)
        if (!lecturePdf || !range) {
          pageChecks.set(section.section, false)
          continue
        }
        if (!haveMutool) {
          pageChecks.set(section.section, true)
          continue
        }
        const pdfPath = path.join(input.root, 'static_resources', lecturePdf.localFile)
        try {
          // the running head of page 1 carries the page number the section starts on in the full notes
          const { stdout } = await run('mutool', ['draw', '-F', 'txt', pdfPath], {
            maxBuffer: 16 * 1024 * 1024,
          })
          const text = stdout
            .replace(/\s+/g, ' ')
            .replace(/[\u2018\u2019]/g, "'")
            .toLowerCase()
          const firstNumber = /\b(\d{1,3})\b/.exec(text)?.[1]
          const firstWords = section.titlePlain.split(/\s+/).slice(0, 2).join(' ')
          const ok =
            firstNumber === String(range[0]) || text.includes(`${section.section}. ${firstWords}`)
          pageChecks.set(section.section, ok)
          if (!ok)
            report.push(
              `lecture pdf ${lecturePdf.localFile} page 1 does not open with section ${section.section} (full page ${range[0]}); links target the full pdf`,
            )
        } catch {
          pageChecks.set(section.section, false)
          report.push(`mutool could not read ${lecturePdf.localFile}; links target the full pdf`)
        }
      }
    }
    for (const section of doc.sections) {
      const unit = input.units[section.section - 1]
      const range = sectionPages.get(section.section) ?? null
      const lecturePdf = pdfResource(unit.n)
      const usesLecturePdf = pageChecks.get(section.section) === true && lecturePdf && range
      const markdown = section.markdown.replace(
        /\[p\. (\d+)\]\(\uE002PAGE:(\d+)\uE002\)/g,
        (_, _p: string, page: string) => {
          const full = Number(page)
          if (usesLecturePdf) {
            const k = full - range![0] + 1
            return `[p. ${k}](${input.slug}/resources/${path.basename(lecturePdf!.relDir)}/#page=${k})`
          }
          if (fullPdf)
            return `[p. ${full}](${input.slug}/resources/${path.basename(fullPdf.relDir)}/#page=${full})`
          return ''
        },
      )
      for (const figure of section.figures) figureJobs.push({ ...figure, section })
      notes.push({
        slug: noteSlug(unit.n),
        n: unit.n,
        title: section.title,
        date: '',
        pages: range,
        statements: section.statements,
      })
      files.set(path.join(notesDir, `${pad(unit.n)}.md`), markdown)
      section.markdown = markdown
    }
    // dates: section 7 has no date line and inherits
    let previous: string | null = null
    for (let i = 0; i < doc.sections.length; i++) {
      const section = doc.sections[i]
      const unit = input.units[i]
      let date = section.date ? `${year}-${section.date}` : null
      let dateSource: 'line' | 'previous' | null = date ? 'line' : null
      if (!date && previous) {
        date = previous
        dateSource = 'previous'
        report.push(`section ${section.section} has no date line; inherits ${previous}`)
      }
      if (date && previous && date < previous)
        report.push(`section ${section.section} date ${date} precedes ${previous}`)
      if (date) previous = date
      const summary = notes[i]
      summary.date = date ?? ''
      const latex: [string, unknown][] = [
        ['source', `static_resources/${full.localFile}`],
        ['lines', [section.startLine, section.endLine]],
        ['section', section.section],
        ['unit', unit.n],
        ['pdf', fullPdf ? path.basename(fullPdf.relDir) : ''],
        ['pages', summary.pages ?? undefined],
        ['dateSource', dateSource === 'previous' ? 'previous' : undefined],
        ['unhandled', section.unhandled.length > 0 ? section.unhandled : undefined],
      ]
      const head = frontmatterYaml([
        ['title', yamlQuote(`${input.number} notes ${pad(unit.n)}: ${section.title}`)],
        [
          'description',
          yamlQuote(section.description || `${input.number} lecture ${pad(unit.n)}: ${unit.title}`),
        ],
        ...(date ? [['date', date] as [string, string]] : []),
        ['tags', flow(['course', 'mit', 'ocw', 'lecture notes'])],
        ['cssclasses', flow(['course-notes'])],
        ['license', yamlQuote(input.licenseName)],
        ['license_url', yamlQuote(input.licenseUrl)],
        ['latex', flowMap(latex)],
      ])
      files.set(path.join(notesDir, `${pad(unit.n)}.md`), `${head}\n\n${section.markdown}\n`)
    }
    pointers.set(full.relDir, `${input.slug}/notes/`)
  }
  draft.zip = zip
  draft.preambleForFigures = preambleForFigures
  draft.noteStatements = noteStatements
}

function countSections(nodes: LatexNode[]): number {
  let count = 0
  const visit = (list: LatexNode[]) => {
    for (const node of list) {
      if (node.type === 'cmd' && node.name === 'section') count++
      else if (node.type === 'env' && node.name === 'document') visit(node.children)
    }
  }
  visit(nodes)
  return count
}

export function problemSetNumber(resource: LatexResource): number | null {
  const match =
    path.posix.basename(resource.relDir).match(problemSetNumberRe) ??
    resource.title.match(problemSetNumberRe)
  return match ? Number(match[1]) : null
}
