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
  date: string
}

export interface PromoteDrafts {
  md: string
  fc: string
  marks: number
  questions: number
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
  const page = mark.target.page
  return page > 0 ? `[[${source.pdf}#page=${page}|p. ${page}]]` : null
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

/**
 * A `lectures/<nn>.md` skeleton with the marks in page order, and a `.fc` draft with one `Q:`
 * per question mark. Answers stay empty: the deck reports them until they are written.
 */
export async function promoteDrafts(
  source: PromoteSource,
  pdfMarks: PdfMark[],
  unitMarks: PdfMark[],
): Promise<PromoteDrafts> {
  const pdf = [...pdfMarks].sort(byPosition)
  const passages = [...unitMarks].sort(byPosition)
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
  }
}
