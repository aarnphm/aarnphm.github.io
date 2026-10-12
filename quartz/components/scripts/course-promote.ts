import type { PdfMark, PdfMarkTarget } from '../../util/pdf-marks'

/** What the strip knows about a unit without notes (`data-promote-*` on `.course-promote`). */
export interface PromoteSource {
  dir: string
  n: number
  unitKind: string
  title: string
  /** Resource page slug without `/index`: the note's `Source:` link. */
  resource: string
  /** The lecture pdf: marks `src` and the page links. */
  pdf: string
  /** `courses/<dir>/lectures/<nn>`: marks on the unit's passages. */
  unit: string
  /** `courses/<dir>/notes/<nn>`, the rendered OCW notes: the workspace marks its passages. */
  text?: string
  date: string
}

export interface PromoteDrafts {
  md: string
  fc: string
  marks: number
  questions: number
  /** Marks on the pdf (text, region and page marks). */
  pdf: number
  /** Marks on passages of the unit or the notes page, saved or browser drafts. */
  passages: number
}

const pad = (n: number) => String(n).padStart(2, '0')

/** Pinned card id: the first 8 hex of sha256(mark id), so a re-run keeps review history. */
export async function cardId(markId: string): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(markId))
  return Array.from(new Uint8Array(digest).slice(0, 4), byte =>
    byte.toString(16).padStart(2, '0'),
  ).join('')
}

const yamlString = (value: string) => `'${value.replaceAll("'", "''")}'`
const oneLine = (value: string) => value.replace(/\s+/g, ' ').trim()
const blockquote = (value: string) =>
  value
    .trim()
    .split('\n')
    .map(line => (line.trim() ? `> ${line}` : '>'))
    .join('\n')

function quoteOf(target: PdfMarkTarget): string | null {
  if (target.type === 'text') return target.quote.exact
  if (target.type === 'passage') return target.text ?? target.quote?.exact ?? null
  return null
}

function citation(source: PromoteSource, mark: PdfMark): string | null {
  const target = mark.target
  if (target.type !== 'passage')
    return target.page > 0 ? `[[${source.pdf}#page=${target.page}|p. ${target.page}]]` : null
  // a passage of the note being written needs no link to itself
  if (!mark.src || mark.src === source.unit) return null
  return `[[${mark.src}${target.heading ? `#${target.heading}` : ''}|notes]]`
}

function entry(source: PromoteSource, mark: PdfMark): string {
  const lines: string[] = []
  const quote = quoteOf(mark.target)
  if (quote) lines.push(blockquote(quote), '')
  const cite = citation(source, mark)
  const lead =
    mark.kind === 'question' ? '**question.** ' : mark.kind === 'contra' ? '**disagree.** ' : ''
  const body = mark.body.trim()
  if (body || lead) lines.push(`${lead}${body}${cite ? ` (${cite})` : ''}`)
  else if (cite) lines.push(cite)
  return lines.join('\n')
}

const byPosition = (a: PdfMark, b: PdfMark) =>
  a.target.page - b.target.page || a.createdAt - b.createdAt || a.id.localeCompare(b.id)

/** One copy per id: a mark can come back from more than one source. */
const unique = (marks: PdfMark[]) => [...new Map(marks.map(mark => [mark.id, mark])).values()]

/**
 * A `lectures/<nn>.md` skeleton with the marks in page order, and a `.fc` draft with one `Q:`
 * per question mark. Answers stay empty: the deck reports them until they are written.
 */
export async function promoteDrafts(
  source: PromoteSource,
  pdfMarks: PdfMark[],
  passageMarks: PdfMark[],
): Promise<PromoteDrafts> {
  const pdf = unique(pdfMarks).sort(byPosition)
  const passages = unique(passageMarks).sort(byPosition)
  const md = [
    '---',
    `title: ${yamlString(`${source.unitKind} ${pad(source.n)}: ${source.title.toLowerCase()}`)}`,
    `description: ${yamlString(`notes on ${source.unitKind} ${source.n}, drafted from ${pdf.length + passages.length} marks`)}`,
    `date: ${yamlString(source.date)}`,
    'tags:',
    '  - course',
    '  - mit',
    '---',
    '',
    `Source: [[${source.resource}|${source.unitKind} ${source.n}]].`,
    '',
  ]
  let page = -1
  for (const mark of pdf) {
    if (mark.target.page !== page) {
      page = mark.target.page
      md.push(`## p. ${page}`, '')
    }
    md.push(entry(source, mark), '')
  }
  if (passages.length > 0) {
    md.push('## passages', '')
    for (const mark of passages) md.push(entry(source, mark), '')
  }
  if (pdf.length + passages.length === 0) md.push('## notes', '')

  const cards: string[] = []
  for (const mark of [...pdf, ...passages]) {
    if (mark.kind !== 'question') continue
    const prompt = oneLine(mark.body) || oneLine(quoteOf(mark.target) ?? '')
    if (!prompt) continue
    const lines = [`Q: ${prompt}`, 'A:']
    const cite = citation(source, mark)
    if (cite) lines.push(`N: ${cite}`)
    lines.push(`ID: ${await cardId(mark.id)}`)
    cards.push(lines.join('\n'))
  }
  return {
    md: `${md.join('\n').trimEnd()}\n`,
    fc: cards.length > 0 ? `${cards.join('\n---\n')}\n` : '',
    marks: pdf.length + passages.length,
    questions: cards.length,
    pdf: pdf.length,
    passages: passages.length,
  }
}
