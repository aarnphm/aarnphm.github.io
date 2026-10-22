import type { LatexNode } from './parser'
import type { Preamble, TheoremEnv } from './preamble'

// ---------------------------------------------------------------------------
// numbering and emission

export interface LabelEntry {
  value: string
  anchor: string | null
  section: number
  line: number
  kind: 'statement' | 'equation' | 'figure' | 'section' | 'subsection' | 'item' | 'other'
}

export interface Counters {
  section: number
  subsection: number
  subsubsection: number
  equation: number
  figure: number
  subfigure: number
  theorem: Map<string, number>
  enumerate: number[]
}

export interface ConvertOptions {
  /** Where a statement's `p. k` link points; null drops the link. */
  pageLink?: (line: number) => { href: string; page: number } | null
  /** `notes/<nn>` slug for section n; used by `\ref` targets. */
  noteSlug: (section: number) => string
  mode: 'notes' | 'pset'
  /** Statement anchor → `slug#^anchor` of the notes statement it restates; `\ref`s follow it. */
  restates?: (statements: StatementText[]) => Map<string, string>
}

export interface ConvertedSection {
  section: number
  title: string
  titlePlain: string
  date: string | null
  description: string
  markdown: string
  startLine: number
  endLine: number
  statements: number
  figures: { name: string; kind: 'input' | 'inline' | 'graphics'; body: string; line: number }[]
  unhandled: string[]
}

/** A numbered statement's rendered body, for matching a pset's restatement to the notes. */
export interface StatementText {
  anchor: string
  section: number
  number: string
  title: string
  text: string
  line: number
}

export interface ConvertedDocument {
  sections: ConvertedSection[]
  labels: Map<string, LabelEntry>
  statements: StatementText[]
  report: string[]
  katexErrors: string[]
}

export interface Converter {
  readonly preamble: Preamble
  readonly options: ConvertOptions
  readonly labels: Map<string, LabelEntry>
  readonly statements: StatementText[]
  readonly report: string[]
  readonly katexErrors: string[]
  readonly refs: { placeholder: string; key: string; kind: 'ref' | 'eqref'; line: number }[]
  counters: Counters
  lastLabelTarget: Omit<LabelEntry, 'line'> | null
  lastStatement: { kind: string; number: string; anchor: string } | null
  proofsFor: Map<string, number>
  section: ConvertedSection | null
  figureStepped: boolean
  subfigureLetter: number
  refCounter: number
  stepTheorem(env: TheoremEnv): string
  resetSectionCounters(): void
  bindLabel(key: string, line: number): void
  setTarget(entry: Omit<LabelEntry, 'line'>): void
  convert(nodes: LatexNode[]): ConvertedSection[]
  documentBody(nodes: LatexNode[]): LatexNode[]
  convertSection(
    cmd: Extract<LatexNode, { type: 'cmd' }> | null,
    nodes: LatexNode[],
    startLine: number,
    endLine: number,
  ): ConvertedSection
  /** Renders a node list into markdown blocks (strings; containers are already indented or quoted). */
  blocks(nodes: LatexNode[]): string[]
  isBlockGroup(node: Extract<LatexNode, { type: 'group' }>): boolean
  cmdBlocks(node: Extract<LatexNode, { type: 'cmd' }>): string[] | null
  figureImage(node: Extract<LatexNode, { type: 'cmd' }>): string
  displayMath(
    src: string,
    line: number,
    env: 'plain' | 'equation' | 'align' | 'eqnarray' | 'gather' | 'multline',
    numbered: boolean | null,
  ): string[]
  gate(src: string, display: boolean, line: number): void
  envBlocks(node: Extract<LatexNode, { type: 'env' }>): string[] | null
  statement(node: Extract<LatexNode, { type: 'env' }>, theorem: TheoremEnv): string[]
  proof(node: Extract<LatexNode, { type: 'env' }>): string[]
  list(node: Extract<LatexNode, { type: 'env' }>): string[]
  figure(node: Extract<LatexNode, { type: 'env' }>): string[]
  subfigure(node: Extract<LatexNode, { type: 'env' }>): string[]
  figureBody(children: LatexNode[], parent: boolean): string[]
  inline(nodes: LatexNode[], mode: 'text' | 'title'): string
  accent(name: string, letter: string): string
  inlineCmd(node: Extract<LatexNode, { type: 'cmd' }>, mode: 'text' | 'title'): string
  /** Replaces `\ref` placeholders once every label is known. */
  resolveRefs(sections: ConvertedSection[]): void
}
