import type {
  ConvertedDocument,
  ConvertedSection,
  Converter,
  ConvertOptions,
  Counters,
  LabelEntry,
  StatementText,
} from './converter-model'
import {
  displayMath,
  envBlocks,
  figure,
  figureBody,
  list,
  proof,
  statement,
  subfigure,
} from './environments'
import {
  accent,
  blocks,
  cmdBlocks,
  figureImage,
  gate,
  inline,
  inlineCmd,
  isBlockGroup,
  normalizeSpaces,
  titleCaseLower,
} from './inline'
import { parseLatex, type LatexNode } from './parser'
import {
  expandMacros,
  scanPreamble,
  stripComments,
  type Preamble,
  type TheoremEnv,
} from './preamble'

const months: Record<string, number> = {
  jan: 1,
  feb: 2,
  mar: 3,
  apr: 4,
  may: 5,
  jun: 6,
  jul: 7,
  aug: 8,
  sep: 9,
  oct: 10,
  nov: 11,
  dec: 12,
}

const dateLineRe =
  /^(?:(?:mon|tue|wed|thu|fri|sat|sun)[a-z]*\.?,?\s+)?(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?,?\s+\$?(\d{1,2})\$?\.?\s*$/i

function mathToPlain(src: string): string {
  return src
    .replace(/\\mathbb\s*\{?([A-Za-z])\}?/g, '$1')
    .replace(/\\(mathrm|mathbf|text|textrm|operatorname)\s*\{([^}]*)\}/g, '$2')
    .replace(/\\[A-Za-z]+/g, '')
    .replace(/[{}^_]/g, '')
    .trim()
}

class LatexConverter implements Converter {
  readonly labels = new Map<string, LabelEntry>()
  readonly statements: StatementText[] = []
  readonly report: string[] = []
  readonly katexErrors: string[] = []
  readonly refs: { placeholder: string; key: string; kind: 'ref' | 'eqref'; line: number }[] = []
  counters: Counters = {
    section: 0,
    subsection: 0,
    subsubsection: 0,
    equation: 0,
    figure: 0,
    subfigure: 0,
    theorem: new Map(),
    enumerate: [],
  }
  lastLabelTarget: Omit<LabelEntry, 'line'> | null = null
  lastStatement: { kind: string; number: string; anchor: string } | null = null
  proofsFor = new Map<string, number>()
  section: ConvertedSection | null = null
  figureStepped = false
  subfigureLetter = 0
  refCounter = 0

  constructor(
    readonly preamble: Preamble,
    readonly options: ConvertOptions,
  ) {}

  // ---- numbering

  stepTheorem(env: TheoremEnv): string {
    return stepTheorem.call(this, env)
  }

  resetSectionCounters(): void {
    return resetSectionCounters.call(this)
  }

  bindLabel(key: string, line: number): void {
    return bindLabel.call(this, key, line)
  }

  setTarget(entry: Omit<LabelEntry, 'line'>): void {
    return setTarget.call(this, entry)
  }

  // ---- document

  convert(nodes: LatexNode[]): ConvertedSection[] {
    return convert.call(this, nodes)
  }

  documentBody(nodes: LatexNode[]): LatexNode[] {
    return documentBody(nodes)
  }

  convertSection(
    cmd: Extract<LatexNode, { type: 'cmd' }> | null,
    nodes: LatexNode[],
    startLine: number,
    endLine: number,
  ): ConvertedSection {
    return convertSection.call(this, cmd, nodes, startLine, endLine)
  }

  // ---- blocks

  /** Renders a node list into markdown blocks (strings; containers are already indented or quoted). */
  blocks(nodes: LatexNode[]): string[] {
    return blocks.call(this, nodes)
  }

  isBlockGroup(node: Extract<LatexNode, { type: 'group' }>): boolean {
    return isBlockGroup(node)
  }

  cmdBlocks(node: Extract<LatexNode, { type: 'cmd' }>): string[] | null {
    return cmdBlocks.call(this, node)
  }

  figureImage(node: Extract<LatexNode, { type: 'cmd' }>): string {
    return figureImage.call(this, node)
  }

  displayMath(
    src: string,
    line: number,
    env: 'plain' | 'equation' | 'align' | 'eqnarray' | 'gather' | 'multline',
    numbered: boolean | null,
  ): string[] {
    return displayMath.call(this, src, line, env, numbered)
  }

  gate(src: string, display: boolean, line: number): void {
    return gate.call(this, src, display, line)
  }

  envBlocks(node: Extract<LatexNode, { type: 'env' }>): string[] | null {
    return envBlocks.call(this, node)
  }

  statement(node: Extract<LatexNode, { type: 'env' }>, theorem: TheoremEnv): string[] {
    return statement.call(this, node, theorem)
  }

  proof(node: Extract<LatexNode, { type: 'env' }>): string[] {
    return proof.call(this, node)
  }

  list(node: Extract<LatexNode, { type: 'env' }>): string[] {
    return list.call(this, node)
  }

  figure(node: Extract<LatexNode, { type: 'env' }>): string[] {
    return figure.call(this, node)
  }

  subfigure(node: Extract<LatexNode, { type: 'env' }>): string[] {
    return subfigure.call(this, node)
  }

  figureBody(children: LatexNode[], parent: boolean): string[] {
    return figureBody.call(this, children, parent)
  }

  // ---- inline

  inline(nodes: LatexNode[], mode: 'text' | 'title'): string {
    return inline.call(this, nodes, mode)
  }

  accent(name: string, letter: string): string {
    return accent(name, letter)
  }

  inlineCmd(node: Extract<LatexNode, { type: 'cmd' }>, mode: 'text' | 'title'): string {
    return inlineCmd.call(this, node, mode)
  }

  /** Replaces `\ref` placeholders once every label is known. */
  resolveRefs(sections: ConvertedSection[]): void {
    return resolveRefs.call(this, sections)
  }
}
function stepTheorem(this: Converter, env: TheoremEnv): string {
  const current = (this.counters.theorem.get(env.counter) ?? 0) + 1
  this.counters.theorem.set(env.counter, current)
  return env.within === 'section' ? `${this.counters.section}.${current}` : String(current)
}

function resetSectionCounters(this: Converter): void {
  this.counters.subsection = 0
  this.counters.subsubsection = 0
  for (const env of this.preamble.theorems.values()) {
    if (env.within === 'section') this.counters.theorem.set(env.counter, 0)
  }
}

function bindLabel(this: Converter, key: string, line: number): void {
  if (!this.lastLabelTarget) {
    this.report.push(`label ${key} has no numbered target (line ${line})`)
    return
  }
  if (this.labels.has(key)) {
    this.report.push(
      `duplicate label ${key} (lines ${this.labels.get(key)!.line} and ${line}); the later one wins`,
    )
  }
  this.labels.set(key, { ...this.lastLabelTarget, line })
}

function setTarget(this: Converter, entry: Omit<LabelEntry, 'line'>): void {
  this.lastLabelTarget = entry
}

function convert(this: Converter, nodes: LatexNode[]): ConvertedSection[] {
  const body = this.documentBody(nodes)
  const sections: ConvertedSection[] = []
  let current: LatexNode[] = []
  let currentCmd: Extract<LatexNode, { type: 'cmd' }> | null = null
  const flushSection = (endLine: number) => {
    if (this.options.mode === 'pset') {
      sections.push(this.convertSection(null, current, body[0]?.line ?? 1, endLine))
    } else if (currentCmd) {
      sections.push(this.convertSection(currentCmd, current, currentCmd.line, endLine))
    }
    current = []
  }
  for (const node of body) {
    if (node.type === 'cmd' && node.name === 'section' && this.options.mode === 'notes') {
      flushSection(node.line - 1)
      currentCmd = node
      continue
    }
    current.push(node)
  }
  const last = body.at(-1)
  flushSection(last ? endLineOf(last) : 1)
  return sections
}

function documentBody(nodes: LatexNode[]): LatexNode[] {
  const doc = nodes.find(node => node.type === 'env' && node.name === 'document')
  return doc && doc.type === 'env' ? doc.children : nodes
}

function convertSection(
  this: Converter,
  cmd: Extract<LatexNode, { type: 'cmd' }> | null,
  nodes: LatexNode[],
  startLine: number,
  endLine: number,
): ConvertedSection {
  if (cmd) {
    this.counters.section++
    this.resetSectionCounters()
  }
  const titleNodes = cmd ? cmd.args[0] : []
  const section: ConvertedSection = {
    section: this.counters.section,
    title: cmd ? titleCaseLower(normalizeSpaces(this.inline(titleNodes, 'title')).trim()) : '',
    titlePlain: cmd ? plainSectionTitle(titleNodes, this.preamble) : '',
    date: null,
    description: '',
    markdown: '',
    startLine,
    endLine,
    statements: 0,
    figures: [],
    unhandled: [],
  }
  this.section = section
  if (cmd) {
    this.setTarget({
      value: String(this.counters.section),
      anchor: null,
      section: this.counters.section,
      kind: 'section',
    })
  }
  const blocks = this.blocks(nodes)
  // date line: the first paragraph of a section
  const firstIndex = blocks.findIndex(block => block.trim().length > 0)
  if (firstIndex !== -1) {
    const first = blocks[firstIndex].replace(/\\\$/g, '$').replace(/\$([^$]*)\$/g, '$1')
    const match = dateLineRe.exec(first.trim())
    if (match) {
      section.date = `${String(months[match[1].slice(0, 3).toLowerCase()]).padStart(2, '0')}-${match[2].padStart(2, '0')}`
      blocks.splice(firstIndex, 1)
    }
  }
  section.description = describe(blocks)
  section.markdown = blocks
    .join('\n\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim()
  return section
}

function resolveRefs(this: Converter, sections: ConvertedSection[]): void {
  const restated = this.options.restates?.(this.statements) ?? new Map<string, string>()
  const resolve = (ref: (typeof this.refs)[number]): string => {
    const entry = this.labels.get(ref.key)
    if (!entry) {
      this.report.push(`undefined reference ${ref.key} (line ${ref.line})`)
      return '??'
    }
    const value = ref.kind === 'eqref' ? `(${entry.value})` : entry.value
    const slug = this.options.noteSlug(entry.section)
    if (entry.kind === 'section') return `[[${slug}|${value}]]`
    if (entry.kind === 'equation' && entry.anchor) return `[[${slug}#${entry.anchor}|${value}]]`
    const target = entry.anchor ? restated.get(entry.anchor) : undefined
    if (target) return `[[${target}|${value}]]`
    if (entry.anchor) return `[[${slug}#^${entry.anchor}|${value}]]`
    return value
  }
  const table = new Map(this.refs.map(ref => [ref.placeholder, resolve(ref)]))
  for (const section of sections) {
    section.markdown = section.markdown.replace(
      /\uE000REF\d+\uE000/g,
      match => table.get(match) ?? '??',
    )
    section.description = section.description.replace(/\uE000REF\d+\uE000/g, match =>
      (table.get(match) ?? '??').replace(/\[\[[^\]|]*\|([^\]]*)\]\]/g, '$1'),
    )
  }
}

function endLineOf(node: LatexNode): number {
  return node.type === 'env' ? node.endLine : node.line
}

function plainSectionTitle(nodes: LatexNode[], preamble: Preamble): string {
  return normalizeSpaces(
    nodes
      .map(node => {
        if (node.type === 'text') return node.value
        if (node.type === 'math') return mathToPlain(expandMacros(node.src, preamble))
        if (node.type === 'group') return plainSectionTitle(node.children, preamble)
        if (node.type === 'cmd')
          return node.args.map(arg => plainSectionTitle(arg, preamble)).join('')
        return ''
      })
      .join(''),
  )
    .replace(/[~]/g, ' ')
    .trim()
    .toLowerCase()
}

/** First sentence of the first prose paragraph whose math is at most single symbols. */
function describe(blocks: string[]): string {
  for (const block of blocks) {
    if (/^(>|\$\$|#|[-*]|\d+\.|<a |```|\uE001|_Figure|\^)/.test(block)) continue
    const maths = Array.from(block.matchAll(/\$([^$]*)\$/g), match => match[1])
    if (maths.some(math => !/^\\?[A-Za-z0-9]{1,3}(?:[_^]\{?[A-Za-z0-9]\}?)?$/.test(math.trim())))
      continue
    const text = block
      .replace(/\$([^$]*)\$/g, (_, math: string) => math.replace(/^\\/, '').replace(/[{}^_]/g, ''))
      .replace(/\\([*_`[\]<$~|\\])/g, '$1')
      .replace(/\[\[[^\]|]*\|([^\]]*)\]\]/g, '$1')
    const sentence = /^(.+?[.?!])(?:\s|$)/.exec(text)?.[1] ?? text
    if (sentence.length > 240 || sentence.length < 20) continue
    return sentence.trim()
  }
  return ''
}

export function convertLatexDocument(source: string, options: ConvertOptions): ConvertedDocument {
  const nodes = parseLatex(stripComments(source))
  const preamble = scanPreamble(nodes)
  const converter = new LatexConverter(preamble, options)
  const sections = converter.convert(nodes)
  converter.resolveRefs(sections)
  return {
    sections,
    labels: converter.labels,
    statements: converter.statements,
    report: [
      ...converter.report,
      ...preamble.bodyDefinitions.map(entry => `definition in body: ${entry}`),
    ],
    katexErrors: converter.katexErrors,
  }
}
