/** Callout kinds that state something; proofs, examples, remarks and figures stay out. */
export const STATEMENT_KINDS = new Set([
  'theorem',
  'lemma',
  'corollary',
  'proposition',
  'definition',
  'conjecture',
])

/** `**Name.**` at the start of a paragraph in an authored note: a term, often a definition. */
export const BOLD_LEAD = /^\*\*([^*\n]{2,60}?)\.\*\*/gm

/** `courses/<dir>/notes/<nn>` (callouts) or `courses/<dir>/lectures/<nn>` (bold leads). */
export const STATEMENT_SOURCE = /^courses\/([^/]+)\/(notes|lectures)\/(\d+)$/

/** Anchor of a bold-lead paragraph: `Singular simplex` → `term-singular-simplex`. */
export function termAnchor(name: string): string {
  const slug = name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '')
  return `term-${slug}`
}

/** Row id on the statement index; anchors repeat across units, so the unit leads. */
export function statementRowId(unit: number, anchor: string): string {
  return `s${String(unit).padStart(2, '0')}-${anchor}`
}
