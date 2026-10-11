import katex from 'katex'
// Converts the LaTeX OCW ships as "editable files" into per-lecture markdown notes.
// Plan: docs/plans/course-resources.md §4. Filesystem and TeX tools stay in this script.
import { execFile } from 'node:child_process'
import { createHash } from 'node:crypto'
import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { promisify } from 'node:util'
import { gunzipSync } from 'node:zlib'
import { customMacros, katexOptions } from '../cfg'

const run = promisify(execFile)

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

export interface LatexResult {
  notes: LatexNoteSummary[]
  psets: LatexNoteSummary[]
  figures: number
  report: string[]
  /** Resource relDir → the note slug its pointer page links. */
  pointers: Map<string, string>
}

export class LatexGateError extends Error {}

// ---------------------------------------------------------------------------
// source model

export type LatexNode =
  | { type: 'text'; value: string; line: number }
  | { type: 'par'; line: number }
  | { type: 'cmd'; name: string; args: LatexNode[][]; opt: string | null; line: number }
  | { type: 'group'; children: LatexNode[]; line: number }
  | { type: 'math'; display: boolean; src: string; line: number }
  | {
      type: 'env'
      name: string
      opt: string | null
      args: string[]
      children: LatexNode[]
      raw: string
      line: number
      endLine: number
    }

const rawEnvs = new Set([
  'equation',
  'equation*',
  'align',
  'align*',
  'eqnarray',
  'eqnarray*',
  'gather',
  'gather*',
  'multline',
  'multline*',
  'displaymath',
  'tikzpicture',
  'verbatim',
  'lstlisting',
])

// Argument signatures: o optional [..], m mandatory {..}, d dimension (`\vskip10pt`), t one token.
const commandArgs: Record<string, string> = {
  textbf: 'm',
  textit: 'm',
  texttt: 'm',
  textsc: 'm',
  emph: 'm',
  underline: 'm',
  textrm: 'm',
  text: 'm',
  mbox: 'm',
  url: 'm',
  href: 'mm',
  section: 'om',
  'section*': 'om',
  subsection: 'om',
  'subsection*': 'om',
  subsubsection: 'om',
  'subsubsection*': 'om',
  paragraph: 'm',
  label: 'm',
  ref: 'm',
  eqref: 'm',
  pageref: 'm',
  cite: 'om',
  footnote: 'om',
  caption: 'om',
  subcaption: 'om',
  input: 'm',
  include: 'm',
  includegraphics: 'om',
  adjustbox: 'mm',
  vskip: 'd',
  hskip: 'd',
  hspace: 'm',
  'hspace*': 'm',
  vspace: 'm',
  'vspace*': 'm',
  setcounter: 'mm',
  addtocounter: 'mm',
  title: 'm',
  author: 'm',
  date: 'm',
  newcommand: 'tom',
  renewcommand: 'tom',
  providecommand: 'tom',
  def: 't',
  newenvironment: 'momm',
  newtheorem: 'momo',
  DeclareMathOperator: 'mm',
  'DeclareMathOperator*': 'mm',
  usepackage: 'om',
  usetikzlibrary: 'm',
  pgfplotsset: 'm',
  definecolor: 'mmm',
  geometry: 'm',
  documentclass: 'om',
  item: 'o',
  H: 't',
  "'": 't',
  '"': 't',
  '^': 't',
  '`': 't',
  '~': 't',
  '=': 't',
  '.': 't',
  u: 't',
  v: 't',
  c: 't',
  '\\': 'o',
  linebreak: 'o',
  pagebreak: 'o',
  fontsize: 'mm',
  rule: 'omm',
  resizebox: 'mmm',
  scalebox: 'om',
  raisebox: 'om',
  parbox: 'omm',
  makebox: 'oom',
  framebox: 'oom',
  textcolor: 'mm',
  color: 'm',
  bibliography: 'm',
  bibliographystyle: 'm',
}

const envArgs: Record<string, string> = {
  figure: 'o',
  'figure*': 'o',
  wrapfigure: 'omm',
  subfigure: 'om',
  minipage: 'om',
  tabular: 'om',
  table: 'o',
  'table*': 'o',
  array: 'om',
  thebibliography: 'm',
}

function lineStarts(text: string): number[] {
  const starts = [0]
  for (let i = 0; i < text.length; i++) if (text[i] === '\n') starts.push(i + 1)
  return starts
}

function lineAt(starts: number[], index: number): number {
  let lo = 0
  let hi = starts.length - 1
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1
    if (starts[mid] <= index) lo = mid
    else hi = mid - 1
  }
  return lo + 1
}

/** Removes `%` comments (not `\%`); keeps line count so source lines stay addressable. */
export function stripComments(text: string): string {
  const out: string[] = []
  for (const line of text.split('\n')) {
    let result = ''
    for (let i = 0; i < line.length; i++) {
      const ch = line[i]
      if (ch === '\\') {
        result += ch + (line[i + 1] ?? '')
        i++
        continue
      }
      if (ch === '%') break
      result += ch
    }
    out.push(result)
  }
  return out.join('\n')
}

class Parser {
  private pos = 0
  private readonly starts: number[]

  constructor(private readonly text: string) {
    this.starts = lineStarts(text)
  }

  private line(index = this.pos): number {
    return lineAt(this.starts, index)
  }

  private fail(message: string): never {
    throw new LatexGateError(`${message} (line ${this.line()})`)
  }

  parseAll(): LatexNode[] {
    const nodes = this.parseUntil(() => this.pos >= this.text.length)
    return nodes
  }

  private skipSpacesInline(): void {
    while (
      this.pos < this.text.length &&
      (this.text[this.pos] === ' ' || this.text[this.pos] === '\t')
    )
      this.pos++
  }

  /** Reads `{...}` with balanced braces; returns the raw inner text. */
  private readBraced(): string | null {
    this.skipSpacesInline()
    if (this.text[this.pos] !== '{') return null
    let depth = 0
    const start = this.pos
    for (let i = this.pos; i < this.text.length; i++) {
      const ch = this.text[i]
      if (ch === '\\') {
        i++
        continue
      }
      if (ch === '{') depth++
      else if (ch === '}') {
        depth--
        if (depth === 0) {
          this.pos = i + 1
          return this.text.slice(start + 1, i)
        }
      }
    }
    this.fail('unbalanced braces')
  }

  private readOptional(): string | null {
    const save = this.pos
    // an optional argument may follow a newline, as in `\begin{proof}\n[sketch]`? No: TeX reads it, we match.
    while (this.pos < this.text.length && /[ \t]/.test(this.text[this.pos])) this.pos++
    if (this.text[this.pos] !== '[') {
      this.pos = save
      return null
    }
    let depth = 0
    const start = this.pos
    for (let i = this.pos; i < this.text.length; i++) {
      const ch = this.text[i]
      if (ch === '\\') {
        i++
        continue
      }
      if (ch === '{') depth++
      else if (ch === '}') depth--
      else if (ch === ']' && depth === 0) {
        this.pos = i + 1
        return this.text.slice(start + 1, i)
      }
    }
    this.pos = save
    return null
  }

  private readDimension(): string {
    const match =
      /^\s*-?[\d.]+\s*(pt|mm|cm|in|ex|em|bp|pc|dd|cc|sp|\\[a-z]+)?(\s*plus\s*[\d.]+[a-z]*)?(\s*minus\s*[\d.]+[a-z]*)?/.exec(
        this.text.slice(this.pos),
      )
    if (!match) return ''
    this.pos += match[0].length
    return match[0].trim()
  }

  /** One token: a braced group, a control word, or a single character. */
  private readToken(): string {
    this.skipSpacesInline()
    const braced = this.readBraced()
    if (braced !== null) return braced
    const rest = this.text.slice(this.pos)
    const control = /^\\([A-Za-z]+|.)/.exec(rest)
    if (control) {
      this.pos += control[0].length
      return control[0]
    }
    const ch = this.text[this.pos] ?? ''
    this.pos += ch.length
    return ch
  }

  private parseArgs(signature: string, line: number): { args: LatexNode[][]; opt: string | null } {
    const args: LatexNode[][] = []
    let opt: string | null = null
    for (const kind of signature) {
      if (kind === 'o') {
        const value = this.readOptional()
        if (opt === null) opt = value
        continue
      }
      if (kind === 'd') {
        args.push([{ type: 'text', value: this.readDimension(), line }])
        continue
      }
      if (kind === 't') {
        args.push([{ type: 'text', value: this.readToken(), line }])
        continue
      }
      const raw = this.readBraced()
      if (raw === null) {
        args.push([])
        continue
      }
      const inner = new Parser(raw)
      inner.starts.splice(0, inner.starts.length, ...lineStarts(raw))
      const children = inner.parseAll().map(node => shiftLines(node, line - 1))
      args.push(children)
    }
    return { args, opt }
  }

  private parseUntil(done: () => boolean, stopAtClose = false): LatexNode[] {
    const nodes: LatexNode[] = []
    let text = ''
    let textLine = this.line()
    const flush = () => {
      if (text.length > 0) nodes.push({ type: 'text', value: text, line: textLine })
      text = ''
    }
    while (!done()) {
      const ch = this.text[this.pos]
      if (ch === undefined) break
      if (ch === '}') {
        if (stopAtClose) break
        this.pos++
        continue
      }
      if (ch === '{') {
        flush()
        const line = this.line()
        this.pos++
        const children = this.parseUntil(
          () => this.text[this.pos] === '}' || this.pos >= this.text.length,
          true,
        )
        if (this.text[this.pos] === '}') this.pos++
        nodes.push({ type: 'group', children, line })
        textLine = this.line()
        continue
      }
      if (ch === '\n') {
        // a blank line is a paragraph break; a single newline is a space
        const match = /^\n[ \t]*\n[\s]*/.exec(this.text.slice(this.pos))
        if (match) {
          flush()
          nodes.push({ type: 'par', line: this.line() })
          this.pos += match[0].length
          textLine = this.line()
        } else {
          text += ' '
          this.pos++
        }
        continue
      }
      if (ch === '$') {
        flush()
        const line = this.line()
        const display = this.text[this.pos + 1] === '$'
        this.pos += display ? 2 : 1
        const close = display ? '$$' : '$'
        const end = this.findMathEnd(close)
        nodes.push({ type: 'math', display, src: this.text.slice(this.pos, end), line })
        this.pos = end + close.length
        textLine = this.line()
        continue
      }
      if (ch === '\\') {
        const next = this.text[this.pos + 1] ?? ''
        if (next === '[' || next === '(') {
          flush()
          const line = this.line()
          const display = next === '['
          this.pos += 2
          const close = display ? '\\]' : '\\)'
          const end = this.text.indexOf(close, this.pos)
          if (end === -1) this.fail(`unterminated ${close}`)
          nodes.push({ type: 'math', display, src: this.text.slice(this.pos, end), line })
          this.pos = end + 2
          textLine = this.line()
          continue
        }
        const control = /^\\([A-Za-z]+\*?|.)/.exec(this.text.slice(this.pos))
        if (!control) {
          text += '\\'
          this.pos++
          continue
        }
        flush()
        const line = this.line()
        let name = control[1]
        this.pos += control[0].length
        if (name === 'begin') {
          const env = this.readBraced()
          if (env === null) this.fail('\\begin without a name')
          nodes.push(this.parseEnv(env, line))
          textLine = this.line()
          continue
        }
        if (name === 'end') {
          const env = this.readBraced()
          if (env === null) this.fail('\\end without a name')
          // an \end that closes an environment the caller owns
          nodes.push({ type: 'cmd', name: `end:${env}`, args: [], opt: null, line })
          textLine = this.line()
          continue
        }
        if (name.endsWith('*') && !(name in commandArgs)) {
          this.pos--
          name = name.slice(0, -1)
        }
        const signature = commandArgs[name] ?? ''
        const { args, opt } = this.parseArgs(signature, line)
        // a control word swallows the spaces after it
        if (/^[A-Za-z]+\*?$/.test(name) && signature.length === 0) this.skipSpacesInline()
        nodes.push({ type: 'cmd', name, args, opt, line })
        textLine = this.line()
        continue
      }
      text += ch
      this.pos++
    }
    flush()
    return nodes
  }

  private findMathEnd(close: string): number {
    let depth = 0
    for (let i = this.pos; i < this.text.length; i++) {
      const ch = this.text[i]
      if (ch === '\\') {
        i++
        continue
      }
      if (ch === '{') depth++
      else if (ch === '}') depth--
      else if (depth === 0 && this.text.startsWith(close, i)) {
        if (close === '$' && this.text[i + 1] === '$') continue
        return i
      }
    }
    this.fail(`unterminated ${close} math`)
  }

  private parseEnv(name: string, line: number): LatexNode {
    const signature = envArgs[name] ?? 'o'
    let opt: string | null = null
    const args: string[] = []
    for (const kind of signature) {
      if (kind === 'o') {
        const value = this.readOptional()
        if (opt === null) opt = value
      } else args.push(this.readBraced() ?? '')
    }
    if (rawEnvs.has(name)) {
      const end = this.findEnvEnd(name)
      const raw = this.text.slice(this.pos, end)
      const endLine = this.line(end)
      this.pos = end + `\\end{${name}}`.length
      return { type: 'env', name, opt, args, children: [], raw, line, endLine }
    }
    const start = this.pos
    const children = this.parseUntil(() => {
      const last = this.text.slice(this.pos)
      return last.startsWith(`\\end{${name}}`) || this.pos >= this.text.length
    })
    const endLine = this.line()
    const raw = this.text.slice(start, this.pos)
    if (this.text.startsWith(`\\end{${name}}`, this.pos)) this.pos += `\\end{${name}}`.length
    else this.fail(`missing \\end{${name}} for the environment opened at line ${line}`)
    return { type: 'env', name, opt, args, children, raw, line, endLine }
  }

  private findEnvEnd(name: string): number {
    let depth = 0
    const open = `\\begin{${name}}`
    const close = `\\end{${name}}`
    for (let i = this.pos; i < this.text.length; i++) {
      if (this.text.startsWith(open, i)) depth++
      else if (this.text.startsWith(close, i)) {
        if (depth === 0) return i
        depth--
      }
    }
    this.fail(`missing \\end{${name}}`)
  }
}

function shiftLines(node: LatexNode, by: number): LatexNode {
  if (by === 0) return node
  const shifted = { ...node, line: node.line + by } as LatexNode
  if (shifted.type === 'group' || shifted.type === 'env') {
    shifted.children = shifted.children.map(child => shiftLines(child, by))
    if (shifted.type === 'env') shifted.endLine += by
  }
  if (shifted.type === 'cmd')
    shifted.args = shifted.args.map(arg => arg.map(child => shiftLines(child, by)))
  return shifted
}

export function parseLatex(text: string): LatexNode[] {
  return new Parser(text).parseAll()
}

// ---------------------------------------------------------------------------
// preamble

export interface TheoremEnv {
  env: string
  name: string
  counter: string
  within: 'section' | null
}

export interface Preamble {
  theorems: Map<string, TheoremEnv>
  macros: Map<string, string>
  operators: Map<string, string>
  colors: string[]
  bodyDefinitions: string[]
}

function plainText(nodes: LatexNode[]): string {
  return nodes
    .map(node => {
      if (node.type === 'text') return node.value
      if (node.type === 'group') return `{${plainText(node.children)}}`
      if (node.type === 'math') return node.display ? `$$${node.src}$$` : `$${node.src}$`
      if (node.type === 'cmd')
        return `\\${node.name}${node.args.map(arg => `{${plainText(arg)}}`).join('')}`
      if (node.type === 'env') return node.raw
      return '\n\n'
    })
    .join('')
}

export function scanPreamble(nodes: LatexNode[]): Preamble {
  const theorems = new Map<string, TheoremEnv>()
  const macros = new Map<string, string>()
  const operators = new Map<string, string>()
  const colors: string[] = []
  const bodyDefinitions: string[] = []
  let inBody = false
  const visit = (list: LatexNode[]) => {
    for (const node of list) {
      if (node.type === 'env') {
        if (node.name === 'document') {
          inBody = true
          visit(node.children)
          continue
        }
        visit(node.children)
        continue
      }
      if (node.type === 'group') {
        visit(node.children)
        continue
      }
      if (node.type !== 'cmd') continue
      if (node.name === 'newtheorem') {
        const env = plainText(node.args[0]).trim()
        const name = plainText(node.args[1]).trim()
        // \newtheorem{env}[shared]{Name}[within]: the optional before the name shares a counter
        const opt = node.opt
        const within = node.opt && /^section$/.test(node.opt) ? 'section' : null
        const shared = opt && theorems.has(opt) ? theorems.get(opt)!.counter : null
        theorems.set(env, {
          env,
          name,
          counter: shared ?? env,
          within: shared ? theorems.get(opt!)!.within : within,
        })
        continue
      }
      if (
        node.name === 'newcommand' ||
        node.name === 'renewcommand' ||
        node.name === 'providecommand'
      ) {
        const macro = plainText(node.args[0]).trim()
        const body = plainText(node.args[1])
        if (node.opt !== null) {
          bodyDefinitions.push(`${macro} takes arguments (line ${node.line})`)
          continue
        }
        if (node.name === 'renewcommand' && inBody)
          bodyDefinitions.push(`\\renewcommand${macro} (line ${node.line})`)
        if (/^\\[A-Za-z]+$/.test(macro)) macros.set(macro.slice(1), body.trim())
        continue
      }
      if (node.name === 'def' || node.name === 'newenvironment') {
        if (inBody) bodyDefinitions.push(`\\${node.name} (line ${node.line})`)
        continue
      }
      if (node.name === 'DeclareMathOperator') {
        const macro = plainText(node.args[0]).trim()
        if (/^\\[A-Za-z]+$/.test(macro)) {
          operators.set(macro.slice(1), plainText(node.args[1]).trim())
        }
        continue
      }
      if (node.name === 'definecolor') {
        colors.push(
          `\\definecolor{${plainText(node.args[0])}}{${plainText(node.args[1])}}{${plainText(node.args[2])}}`,
        )
      }
    }
  }
  visit(nodes)
  return { theorems, macros, operators, colors, bodyDefinitions }
}

// ---------------------------------------------------------------------------
// math

const katexFixes: [RegExp, string][] = [
  [/\\Tilde(?![A-Za-z])/g, '\\widetilde'],
  [/\\qedhere(?![A-Za-z])/g, ''],
  [/\\textrm(?![A-Za-z])/g, '\\text'],
  [/\\bf(?![A-Za-z])/g, '\\mathbf'],
  [/\\displaystyle(?![A-Za-z])/g, ''],
]

/** Expands zero-argument macros on control-word boundaries, so `\T` leaves `\TT` and `\Theta` alone. */
export function expandMacros(src: string, preamble: Preamble): string {
  let out = src
  const table = new Map<string, string>()
  for (const [name, body] of preamble.macros) table.set(name, body)
  for (const [name, body] of preamble.operators) {
    table.set(name, `\\operatorname{${body.replace(/\\textrm\{([^}]*)\}/g, '$1')}}`)
  }
  for (let round = 0; round < 8; round++) {
    let changed = false
    out = out.replace(
      /\\([A-Za-z]+)(?![A-Za-z])/g,
      (match, name: string, offset: number, whole: string) => {
        const body = table.get(name)
        if (body === undefined) return match
        changed = true
        // a space keeps the expansion from fusing with a following letter (`\eps x` → `\epsilon x`)
        const next = whole[offset + match.length] ?? ''
        return /[A-Za-z]/.test(next) && /[A-Za-z]$/.test(body) ? `${body} ` : body
      },
    )
    if (!changed) break
  }
  for (const [pattern, replacement] of katexFixes) out = out.replace(pattern, replacement)
  return out.replace(/[ \t]+$/gm, '').trim()
}

export function checkKatex(src: string, display: boolean): string | null {
  try {
    katex.renderToString(src, {
      ...katexOptions,
      displayMode: display,
      macros: { ...customMacros },
    })
    return null
  } catch (error) {
    return error instanceof Error ? error.message : String(error)
  }
}

/** Splits an alignment body into rows at `\\` that sit outside braces and nested environments. */
export function splitRows(body: string): string[] {
  const rows: string[] = []
  let depth = 0
  let envDepth = 0
  let current = ''
  for (let i = 0; i < body.length; i++) {
    const ch = body[i]
    if (ch === '\\') {
      if (body.startsWith('\\begin{', i)) envDepth++
      else if (body.startsWith('\\end{', i)) envDepth--
      else if (body[i + 1] === '\\' && depth === 0 && envDepth === 0) {
        rows.push(current)
        current = ''
        i++
        const trailing = /^\[[^\]]*\]/.exec(body.slice(i + 1))
        if (trailing) i += trailing[0].length
        continue
      }
      current += ch + (body[i + 1] ?? '')
      i++
      continue
    }
    if (ch === '{') depth++
    else if (ch === '}') depth--
    current += ch
  }
  rows.push(current)
  return rows.map(row => row.replace(/^\s*\n|\n\s*$/g, ''))
}

// ---------------------------------------------------------------------------
// numbering and emission

export interface LabelEntry {
  value: string
  anchor: string | null
  section: number
  line: number
  kind: 'statement' | 'equation' | 'figure' | 'section' | 'subsection' | 'item' | 'other'
}

interface Counters {
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

const theoremCalloutTypes: Record<string, string> = {
  theorem: 'theorem',
  lemma: 'lemma',
  proposition: 'proposition',
  definition: 'definition',
  corollary: 'corollary',
  conjecture: 'conjecture',
  claim: 'claim',
  axiom: 'axiom',
}

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

const accents: Record<string, Record<string, string>> = {
  "'": {
    a: 'á',
    e: 'é',
    i: 'í',
    o: 'ó',
    u: 'ú',
    y: 'ý',
    c: 'ć',
    n: 'ń',
    s: 'ś',
    z: 'ź',
    A: 'Á',
    E: 'É',
    O: 'Ó',
  },
  '"': { a: 'ä', e: 'ë', i: 'ï', o: 'ö', u: 'ü', y: 'ÿ', A: 'Ä', O: 'Ö', U: 'Ü' },
  '`': { a: 'à', e: 'è', i: 'ì', o: 'ò', u: 'ù', A: 'À', E: 'È' },
  '^': { a: 'â', e: 'ê', i: 'î', o: 'ô', u: 'û' },
  '~': { a: 'ã', n: 'ñ', o: 'õ', N: 'Ñ' },
  H: { o: 'ő', u: 'ű', O: 'Ő', U: 'Ű' },
  c: { c: 'ç', C: 'Ç', s: 'ş' },
  v: { c: 'č', s: 'š', z: 'ž', C: 'Č', S: 'Š', Z: 'Ž', r: 'ř', e: 'ě' },
  u: { a: 'ă', g: 'ğ' },
  '=': { a: 'ā', e: 'ē', i: 'ī', o: 'ō', u: 'ū' },
  '.': { z: 'ż', e: 'ė' },
}

function escapeText(value: string): string {
  return value
    .replace(/\\/g, '\\\\')
    .replace(/\*/g, '\\*')
    .replace(/_/g, '\\_')
    .replace(/`/g, '\\`')
    .replace(/\[/g, '\\[')
    .replace(/\]/g, '\\]')
    .replace(/</g, '\\<')
    .replace(/==/g, '\\==')
    .replace(/\$/g, '\\$')
    .replace(/~/g, '\\~')
    .replace(/\|/g, '\\|')
}

function escapeLineStart(line: string): string {
  return /^(#|>|[-+*]\s|\d+[.)]\s)/.test(line) ? `\\${line}` : line
}

function normalizeSpaces(value: string): string {
  return value.replace(/[ \t]+/g, ' ')
}

function titleCaseLower(value: string): string {
  // text runs lowercased, math untouched
  return value.replace(
    /(\$[^$]*\$)|([^$]+)/g,
    (_, math: string | undefined, text: string | undefined) => math ?? (text ?? '').toLowerCase(),
  )
}

function mathToPlain(src: string): string {
  return src
    .replace(/\\mathbb\s*\{?([A-Za-z])\}?/g, '$1')
    .replace(/\\(mathrm|mathbf|text|textrm|operatorname)\s*\{([^}]*)\}/g, '$2')
    .replace(/\\[A-Za-z]+/g, '')
    .replace(/[{}^_]/g, '')
    .trim()
}

class Converter {
  readonly labels = new Map<string, LabelEntry>()
  readonly statements: StatementText[] = []
  readonly report: string[] = []
  readonly katexErrors: string[] = []
  readonly refs: { placeholder: string; key: string; kind: 'ref' | 'eqref'; line: number }[] = []
  private counters: Counters = {
    section: 0,
    subsection: 0,
    subsubsection: 0,
    equation: 0,
    figure: 0,
    subfigure: 0,
    theorem: new Map(),
    enumerate: [],
  }
  private lastLabelTarget: Omit<LabelEntry, 'line'> | null = null
  private lastStatement: { kind: string; number: string; anchor: string } | null = null
  private proofsFor = new Map<string, number>()
  private section: ConvertedSection | null = null
  private figureStepped = false
  private subfigureLetter = 0
  private refCounter = 0

  constructor(
    private readonly preamble: Preamble,
    private readonly options: ConvertOptions,
  ) {}

  // ---- numbering

  private stepTheorem(env: TheoremEnv): string {
    const current = (this.counters.theorem.get(env.counter) ?? 0) + 1
    this.counters.theorem.set(env.counter, current)
    return env.within === 'section' ? `${this.counters.section}.${current}` : String(current)
  }

  private resetSectionCounters(): void {
    this.counters.subsection = 0
    this.counters.subsubsection = 0
    for (const env of this.preamble.theorems.values()) {
      if (env.within === 'section') this.counters.theorem.set(env.counter, 0)
    }
  }

  private bindLabel(key: string, line: number): void {
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

  private setTarget(entry: Omit<LabelEntry, 'line'>): void {
    this.lastLabelTarget = entry
  }

  // ---- document

  convert(nodes: LatexNode[]): ConvertedSection[] {
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

  private documentBody(nodes: LatexNode[]): LatexNode[] {
    const doc = nodes.find(node => node.type === 'env' && node.name === 'document')
    return doc && doc.type === 'env' ? doc.children : nodes
  }

  private convertSection(
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

  // ---- blocks

  /** Renders a node list into markdown blocks (strings; containers are already indented or quoted). */
  blocks(nodes: LatexNode[]): string[] {
    const out: string[] = []
    let inline: string[] = []
    const flush = () => {
      const text = normalizeSpaces(inline.join('')).trim()
      inline = []
      if (text.length > 0) out.push(escapeLineStart(text))
    }
    for (let i = 0; i < nodes.length; i++) {
      const node = nodes[i]
      if (node.type === 'par') {
        flush()
        continue
      }
      if (node.type === 'math' && node.display) {
        flush()
        out.push(...this.displayMath(node.src, node.line, 'plain', null))
        continue
      }
      if (node.type === 'env') {
        const rendered = this.envBlocks(node)
        if (rendered !== null) {
          flush()
          out.push(...rendered)
          continue
        }
      }
      if (node.type === 'cmd') {
        const rendered = this.cmdBlocks(node)
        if (rendered !== null) {
          flush()
          out.push(...rendered)
          continue
        }
      }
      if (node.type === 'group' && this.isBlockGroup(node)) {
        flush()
        out.push(...this.blocks(node.children))
        continue
      }
      inline.push(this.inline([node], 'text'))
    }
    flush()
    return out
  }

  private isBlockGroup(node: Extract<LatexNode, { type: 'group' }>): boolean {
    return node.children.some(
      child => child.type === 'par' || (child.type === 'env' && !rawEnvs.has(child.name)),
    )
  }

  private cmdBlocks(node: Extract<LatexNode, { type: 'cmd' }>): string[] | null {
    switch (node.name) {
      case 'section':
      case 'section*':
      case 'subsection*':
      case 'subsubsection*': {
        // unnumbered headings; a pset \section is a heading since the file is one page
        const level = node.name.startsWith('section')
          ? 2
          : node.name.startsWith('subsection')
            ? 3
            : 4
        const title = titleCaseLower(normalizeSpaces(this.inline(node.args[0], 'title')).trim())
        return [`${'#'.repeat(level)} ${title}`]
      }
      case 'newcommand':
      case 'renewcommand':
      case 'providecommand':
      case 'DeclareMathOperator':
      case 'newtheorem':
      case 'def':
        // collected by scanPreamble; body definitions of argument macros are reported there
        return []
      case 'subsection':
      case 'subsubsection': {
        const level = node.name === 'subsection' ? 2 : 3
        if (level === 2) {
          this.counters.subsection++
          this.counters.subsubsection = 0
        } else this.counters.subsubsection++
        const number =
          level === 2
            ? `${this.counters.section}.${this.counters.subsection}`
            : `${this.counters.section}.${this.counters.subsection}.${this.counters.subsubsection}`
        this.setTarget({
          value: number,
          anchor: null,
          section: this.counters.section,
          kind: level === 2 ? 'subsection' : 'other',
        })
        const title = titleCaseLower(normalizeSpaces(this.inline(node.args[0], 'title')).trim())
        return [`${'#'.repeat(level)} ${number} ${title}`]
      }
      case 'paragraph':
        return [`**${normalizeSpaces(this.inline(node.args[0], 'text')).trim()}**`]
      case 'newpage':
      case 'tableofcontents':
      case 'maketitle':
      case 'title':
      case 'setcounter':
      case 'vskip':
      case 'hskip':
      case 'smallskip':
      case 'medskip':
      case 'bigskip':
      case 'clearpage':
      case 'pagebreak':
        return []
      case '\\':
      case 'newline':
      case 'linebreak':
        return []
      case 'input':
      case 'includegraphics':
        return [this.figureImage(node)]
      default:
        return null
    }
  }

  private figureImage(node: Extract<LatexNode, { type: 'cmd' }>): string {
    const name = plainText(node.args[node.name === 'input' ? 0 : 0]).trim()
    const kind = node.name === 'input' ? 'input' : 'graphics'
    this.section?.figures.push({ name, kind, body: '', line: node.line })
    return `\uE001FIG:${kind}:${name}:${node.line}\uE001`
  }

  private displayMath(
    src: string,
    line: number,
    env: 'plain' | 'equation' | 'align' | 'eqnarray' | 'gather' | 'multline',
    numbered: boolean | null,
  ): string[] {
    const blocks: string[] = []
    let body = src
    const anchors: string[] = []
    const labelsInRow = (row: string): string[] =>
      Array.from(row.matchAll(/\\label\{([^}]*)\}/g), match => match[1].trim())
    const stripRow = (row: string) =>
      row
        .replace(/\\label\{[^}]*\}/g, '')
        .replace(/\\(nonumber|notag)(?![A-Za-z])/g, '')
        .trim()
    if (env === 'plain') {
      for (const key of labelsInRow(body)) {
        this.report.push(`label ${key} inside unnumbered display math is ignored (line ${line})`)
      }
      body = stripRow(body)
      const expanded = expandMacros(body, this.preamble)
      this.gate(expanded, true, line)
      return [`$$\n${expanded}\n$$`]
    }
    const rows = env === 'equation' ? [body] : splitRows(body)
    const taggedRows = rows.map(row => {
      const labels = labelsInRow(row)
      const unnumbered = /\\(nonumber|notag)(?![A-Za-z])/.test(row) || numbered === false
      const tag = /\\tag\{([^}]*)\}/.exec(row)
      let cleaned = stripRow(row)
      if (env === 'eqnarray')
        cleaned = cleaned.replace(/&\s*=\s*&/g, '&=').replace(/&\s*([<>≤≥\\]+[^&]*?)\s*&/g, '&$1')
      if (cleaned.length === 0) return cleaned
      if (unnumbered && !tag) return cleaned
      let value: string
      if (tag) value = tag[1]
      else {
        this.counters.equation++
        value = String(this.counters.equation)
        cleaned = `${cleaned} \\tag{${value}}`
      }
      const anchor = `eq-${value.replace(/[^A-Za-z0-9]+/g, '-')}`
      anchors.push(anchor)
      this.setTarget({ value, anchor, section: this.counters.section, kind: 'equation' })
      for (const key of labels) this.bindLabel(key, line)
      return cleaned
    })
    const content = taggedRows.filter(row => row.length > 0).join(' \\\\\n')
    let expanded = expandMacros(content, this.preamble)
    if (env !== 'equation') {
      const wrapper = env === 'align' || env === 'eqnarray' ? 'align*' : `${env}*`
      expanded = `\\begin{${wrapper}}\n${expanded}\n\\end{${wrapper}}`
    }
    this.gate(expanded, true, line)
    if (anchors.length > 0) {
      blocks.push(anchors.map(anchor => `<a id="${anchor}" class="latex-eq"></a>`).join(''))
    }
    blocks.push(`$$\n${expanded}\n$$`)
    return blocks
  }

  private gate(src: string, display: boolean, line: number): void {
    const error = checkKatex(src, display)
    if (error)
      this.katexErrors.push(
        `line ${line}: ${error.replace(/^KaTeX parse error: /, '')}\n    ${src.slice(0, 160)}`,
      )
  }

  private envBlocks(node: Extract<LatexNode, { type: 'env' }>): string[] | null {
    const name = node.name
    const theorem = this.preamble.theorems.get(name)
    if (theorem) return this.statement(node, theorem)
    switch (name) {
      case 'proof':
        return this.proof(node)
      case 'equation':
      case 'equation*':
      case 'displaymath':
        return this.displayMath(
          node.raw,
          node.line,
          name === 'equation' ? 'equation' : 'plain',
          name === 'equation',
        )
      case 'align':
      case 'align*':
      case 'eqnarray':
      case 'eqnarray*':
      case 'gather':
      case 'gather*':
      case 'multline':
      case 'multline*': {
        const base = name.replace('*', '') as 'align' | 'eqnarray' | 'gather' | 'multline'
        return this.displayMath(node.raw, node.line, base, !name.endsWith('*'))
      }
      case 'itemize':
      case 'enumerate':
      case 'description':
        return this.list(node)
      case 'figure':
      case 'figure*':
      case 'wrapfigure':
        return this.figure(node)
      case 'subfigure':
        return this.subfigure(node)
      case 'center':
      case 'minipage':
      case 'flushleft':
      case 'flushright':
      case 'quote':
      case 'quotation':
      case 'small':
      case 'footnotesize':
        return this.blocks(node.children)
      case 'tikzpicture': {
        const body = node.raw
        const name = `tikz-${createHash('sha256').update(body).digest('hex').slice(0, 10)}`
        this.section?.figures.push({
          name,
          kind: 'inline',
          body: `\\begin{tikzpicture}${body}\\end{tikzpicture}`,
          line: node.line,
        })
        return [`\uE001FIG:inline:${name}:${node.line}\uE001`]
      }
      case 'abstract':
        return this.blocks(node.children)
      case 'document':
        return this.blocks(node.children)
      default: {
        this.section?.unhandled.push(`environment ${name} (line ${node.line})`)
        const raw = `\\begin{${name}}${node.opt ? `[${node.opt}]` : ''}${node.args.map(arg => `{${arg}}`).join('')}${node.raw}\\end{${name}}`
        return ['```latex\n' + raw.trim() + '\n```']
      }
    }
  }

  private statement(node: Extract<LatexNode, { type: 'env' }>, theorem: TheoremEnv): string[] {
    const number = this.stepTheorem(theorem)
    const kindWord = theorem.name.toLowerCase()
    const calloutType = theoremCalloutTypes[kindWord] ?? kindWord
    const anchor = `${calloutType}-${number.replace(/\./g, '-')}`
    this.setTarget({ value: number, anchor, section: this.counters.section, kind: 'statement' })
    this.lastStatement = { kind: kindWord, number, anchor }
    this.proofsFor.set(anchor, 0)
    if (this.section) this.section.statements++

    let children = node.children
    let title = node.opt ? this.inline(new Parser(node.opt).parseAll(), 'title').trim() : ''
    if (!title) {
      const lead = leadingParenthetical(children)
      if (lead) {
        title = this.inline(lead.nodes, 'title').trim()
        children = lead.rest
      }
    }
    children = dropLeadingBreaks(children)
    const mathInTitle = /\$/.test(title)
    const isMath = calloutType in theoremCalloutTypes
    const header = isMath
      ? `> [!${calloutType}|latex] ${number}${title && !mathInTitle ? ` ${title}` : ''}`
      : `> [!${calloutType}|latex] ${theorem.name} ${number}${title && !mathInTitle ? `. ${title}` : ''}`
    const body = this.blocks(children)
    this.statements.push({
      anchor,
      section: this.counters.section,
      number,
      title,
      text: body.join('\n\n'),
      line: node.line,
    })
    if (mathInTitle) body.unshift(`_(${title})_`)
    const page = this.options.pageLink?.(node.line) ?? null
    if (page) body.push(`[p. ${page.page}](${page.href})`)
    return [[header, '>', ...quote(body.join('\n\n'))].join('\n'), `^${anchor}`]
  }

  private proof(node: Extract<LatexNode, { type: 'env' }>): string[] {
    const statement = this.lastStatement
    let anchor = 'proof'
    if (statement) {
      const count = (this.proofsFor.get(statement.anchor) ?? 0) + 1
      this.proofsFor.set(statement.anchor, count)
      anchor = `proof-${statement.number.replace(/\./g, '-')}${count > 1 ? `-${String.fromCharCode(96 + count)}` : ''}`
    }
    const label = node.opt ? this.inline(new Parser(node.opt).parseAll(), 'title').trim() : ''
    const of = statement ? ` of ${statement.kind} ${statement.number}` : ''
    const title = label ? `${label}${of}` : of.trim()
    const header = `> [!proof|latex]+ ${title}`.trimEnd()
    const body = this.blocks(dropLeadingBreaks(node.children))
    const blocks = [[header, '>', ...quote(body.join('\n\n'))].join('\n')]
    if (statement) blocks.push(`^${anchor}`)
    return blocks
  }

  private list(node: Extract<LatexNode, { type: 'env' }>): string[] {
    const ordered = node.name === 'enumerate'
    if (ordered) this.counters.enumerate.push(0)
    const items: { label: string | null; nodes: LatexNode[] }[] = []
    let current: { label: string | null; nodes: LatexNode[] } | null = null
    for (const child of node.children) {
      if (child.type === 'cmd' && child.name === 'item') {
        current = { label: child.opt, nodes: [] }
        items.push(current)
        continue
      }
      if (!current) {
        if (child.type === 'par' || (child.type === 'text' && child.value.trim() === '')) continue
        current = { label: null, nodes: [] }
        items.push(current)
      }
      current.nodes.push(child)
    }
    const lines: string[] = []
    for (const item of items) {
      let marker: string
      if (ordered) {
        const depth = this.counters.enumerate.length - 1
        this.counters.enumerate[depth]++
        const n = this.counters.enumerate[depth]
        marker = `${n}.`
        this.setTarget({
          value: String(n),
          anchor: null,
          section: this.counters.section,
          kind: 'item',
        })
      } else marker = '-'
      const label = item.label !== null ? `**${escapeText(item.label)}** ` : ''
      const blocks = this.blocks(dropLeadingBreaks(item.nodes))
      const first = blocks.shift() ?? ''
      const indent = ' '.repeat(ordered ? marker.length + 1 : 2)
      lines.push(`${marker} ${label}${first}`.trimEnd())
      for (const block of blocks) {
        lines.push('')
        lines.push(...block.split('\n').map(line => (line.length > 0 ? indent + line : line)))
      }
    }
    if (ordered) this.counters.enumerate.pop()
    return [lines.join('\n')]
  }

  private figure(node: Extract<LatexNode, { type: 'env' }>): string[] {
    const outerStepped = this.figureStepped
    this.figureStepped = false
    this.subfigureLetter = 0
    const blocks = this.figureBody(node.children, true)
    this.figureStepped = outerStepped
    return blocks
  }

  private subfigure(node: Extract<LatexNode, { type: 'env' }>): string[] {
    return this.figureBody(node.children, false)
  }

  private figureBody(children: LatexNode[], parent: boolean): string[] {
    const out: string[] = []
    for (const child of children) {
      if (child.type === 'cmd' && child.name === 'caption') {
        let value: string
        let anchor: string
        if (parent) {
          if (!this.figureStepped) {
            this.counters.figure++
            this.figureStepped = true
          }
          value = String(this.counters.figure)
          anchor = `figure-${value}`
        } else {
          if (!this.figureStepped) {
            this.counters.figure++
            this.figureStepped = true
          }
          this.subfigureLetter++
          value = `${this.counters.figure}${String.fromCharCode(96 + this.subfigureLetter)}`
          anchor = `figure-${value}`
        }
        this.setTarget({ value, anchor, section: this.counters.section, kind: 'figure' })
        const text = normalizeSpaces(this.inline(child.args[0], 'text')).trim()
        out.push(`_Figure ${value}._ ${text}`.trimEnd(), `^${anchor}`)
        continue
      }
      if (
        child.type === 'cmd' &&
        (child.name === 'centering' || child.name === 'hfill' || child.name === 'label')
      ) {
        if (child.name === 'label') this.bindLabel(plainText(child.args[0]).trim(), child.line)
        continue
      }
      if (child.type === 'par' || (child.type === 'text' && child.value.trim() === '')) continue
      if (child.type === 'env' && child.name === 'subfigure') {
        out.push(...this.subfigure(child))
        continue
      }
      if (child.type === 'cmd' && child.name === 'adjustbox') {
        out.push(...this.blocks(child.args[1]))
        continue
      }
      out.push(...this.blocks([child]))
    }
    return out
  }

  // ---- inline

  inline(nodes: LatexNode[], mode: 'text' | 'title'): string {
    let out = ''
    for (let i = 0; i < nodes.length; i++) {
      const node = nodes[i]
      switch (node.type) {
        case 'text':
          out += escapeText(normalizeSpaces(node.value.replace(/~/g, ' ')))
            .replace(/\\~/g, ' ')
            .replace(/---/g, '—')
            .replace(/--/g, '–')
            .replace(/``|''/g, '"')
            .replace(/`/g, "'")
          break
        case 'par':
          out += ' '
          break
        case 'math': {
          const expanded = expandMacros(node.src, this.preamble)
          this.gate(expanded, false, node.line)
          out += `$${expanded}$`
          break
        }
        case 'group': {
          const [first, ...rest] = node.children
          if (first?.type === 'cmd' && (first.name === 'bf' || first.name === 'textbf')) {
            out += `**${this.inline(first.name === 'bf' ? rest : [...first.args[0], ...rest], mode).trim()}**`
          } else if (
            first?.type === 'cmd' &&
            (first.name === 'it' || first.name === 'em' || first.name === 'sl')
          ) {
            out += `_${this.inline(rest, mode).trim()}_`
          } else if (first?.type === 'cmd' && first.name in accents && rest.length > 0) {
            out += this.accent(first.name, this.inline(rest, mode))
          } else out += this.inline(node.children, mode)
          break
        }
        case 'env':
          if (rawEnvs.has(node.name) && node.name !== 'tikzpicture' && node.name !== 'verbatim') {
            // display math inside an inline context (a title): keep the source
            out += `$$${expandMacros(node.raw, this.preamble)}$$`
          } else out += this.blocks([node]).join('\n\n')
          break
        case 'cmd':
          out += this.inlineCmd(node, mode)
          break
      }
    }
    return out
  }

  private accent(name: string, letter: string): string {
    const base = letter.replace(/\\i(?![A-Za-z])/g, 'i').trim()
    const table = accents[name]
    const mapped = table?.[base]
    if (mapped) return mapped
    return base
  }

  private inlineCmd(node: Extract<LatexNode, { type: 'cmd' }>, mode: 'text' | 'title'): string {
    const arg = (index: number) => this.inline(node.args[index] ?? [], mode)
    switch (node.name) {
      case 'textbf':
      case 'bf':
        return node.args.length ? `**${arg(0).trim()}**` : ''
      case 'emph':
      case 'textit':
      case 'it':
      case 'em':
        return node.args.length ? `_${arg(0).trim()}_` : ''
      case 'underline':
        return `<u>${arg(0)}</u>`
      case 'texttt':
        return `\`${plainText(node.args[0])}\``
      case 'textrm':
      case 'text':
      case 'mbox':
      case 'textsc':
      case 'adjustbox':
        return arg(node.name === 'adjustbox' ? 1 : 0)
      case 'url':
        return `<${plainText(node.args[0]).trim()}>`
      case 'href':
        return `[${arg(1)}](${plainText(node.args[0]).trim()})`
      case 'label':
        this.bindLabel(plainText(node.args[0]).trim(), node.line)
        return ''
      case 'ref':
      case 'eqref': {
        const key = plainText(node.args[0]).trim()
        const placeholder = `\uE000REF${this.refCounter++}\uE000`
        this.refs.push({ placeholder, key, kind: node.name, line: node.line })
        return placeholder
      }
      case 'cite':
        this.section?.unhandled.push(`\\cite (line ${node.line})`)
        return `[${plainText(node.args[0])}]`
      case 'footnote':
        return `^[${arg(0).trim()}]`
      case 'ldots':
      case 'dots':
        return '…'
      case 'LaTeX':
        return 'LaTeX'
      case 'TeX':
        return 'TeX'
      case '%':
        return '%'
      case '&':
        return '&'
      case '$':
        return '\\$'
      case '#':
        return '#'
      case '_':
        return '\\_'
      case '{':
        return '{'
      case '}':
        return '}'
      case ',':
      case ';':
      case ' ':
      case '!':
      case '/':
        return ' '
      case '-':
        return ''
      case 'quad':
      case 'qquad':
        return ' '
      case '\\':
      case 'newline':
      case 'linebreak':
        return mode === 'title' ? ' ' : '\n'
      case 'noindent':
      case 'indent':
      case 'centering':
      case 'hfill':
      case 'hspace':
      case 'vspace':
      case 'qedhere':
      case 'small':
      case 'footnotesize':
      case 'large':
      case 'Large':
      case 'LARGE':
      case 'huge':
      case 'normalsize':
      case 'textwidth':
      case 'linewidth':
      case 'par':
      case 'smallskip':
      case 'medskip':
      case 'bigskip':
      case 'vskip':
      case 'hskip':
      case 'newpage':
      case 'clearpage':
      case 'label*':
      case 'relax':
      case 'raggedright':
        return ''
      case 'i':
        return 'i'
      case 'j':
        return 'j'
      case 'o':
        return 'ø'
      case 'ss':
        return 'ß'
      case 'ae':
        return 'æ'
      case 'oe':
        return 'œ'
      case 'sim':
        return '~'
      default:
        break
    }
    if (node.name in accents && node.args.length > 0) return this.accent(node.name, arg(0))
    const macro = this.preamble.macros.get(node.name)
    if (macro !== undefined) {
      const expanded = expandMacros(macro, this.preamble)
      this.gate(expanded, false, node.line)
      return `$${expanded}$`
    }
    this.section?.unhandled.push(`\\${node.name} (line ${node.line})`)
    const raw = `\\${node.name}${node.opt ? `[${node.opt}]` : ''}${node.args.map(a => `{${plainText(a)}}`).join('')}`
    return `\`${raw}\``
  }

  /** Replaces `\ref` placeholders once every label is known. */
  resolveRefs(sections: ConvertedSection[]): void {
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
}

function endLineOf(node: LatexNode): number {
  return node.type === 'env' ? node.endLine : node.line
}

function quote(markdown: string): string[] {
  return markdown.split('\n').map(line => (line.length > 0 ? `> ${line}` : '>'))
}

function dropLeadingBreaks(nodes: LatexNode[]): LatexNode[] {
  let i = 0
  while (i < nodes.length) {
    const node = nodes[i]
    if (node.type === 'par') i++
    else if (node.type === 'text' && node.value.trim() === '') i++
    else if (node.type === 'cmd' && (node.name === '\\' || node.name === 'newline')) i++
    else break
  }
  return nodes.slice(i)
}

/** `\begin{theorem}(Name) body` → the name, when it is short and leads the body. */
function leadingParenthetical(
  nodes: LatexNode[],
): { nodes: LatexNode[]; rest: LatexNode[] } | null {
  const body = dropLeadingBreaks(nodes)
  const first = body[0]
  if (!first || first.type !== 'text' || !first.value.trimStart().startsWith('(')) return null
  // find the closing paren across text/math/group nodes
  let depth = 0
  for (let i = 0; i < body.length; i++) {
    const node = body[i]
    if (node.type !== 'text') {
      if (node.type === 'par' || node.type === 'env') return null
      continue
    }
    for (let j = 0; j < node.value.length; j++) {
      const ch = node.value[j]
      if (ch === '(') depth++
      else if (ch === ')') {
        depth--
        if (depth === 0) {
          const head = body.slice(0, i)
          const headText = node.value.slice(0, j)
          const tail = node.value.slice(j + 1)
          const titleNodes: LatexNode[] = [
            ...head,
            { type: 'text', value: headText, line: node.line },
          ]
          const startIndex = titleNodes[0].type === 'text' ? titleNodes[0].value.indexOf('(') : 0
          if (titleNodes[0].type === 'text')
            titleNodes[0] = { ...titleNodes[0], value: titleNodes[0].value.slice(startIndex + 1) }
          const length = plainText(titleNodes).length
          if (length > 80 || length === 0) return null
          const rest: LatexNode[] = [
            { type: 'text', value: tail, line: node.line },
            ...body.slice(i + 1),
          ]
          return { nodes: titleNodes, rest }
        }
      }
    }
  }
  return null
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
  const converter = new Converter(preamble, options)
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

// ---------------------------------------------------------------------------
// aux, toc, synctex

export function parseAux(text: string): Map<string, { value: string; page: number }> {
  const out = new Map<string, { value: string; page: number }>()
  for (const match of text.matchAll(/\\newlabel\{([^}]*)\}\{\{([^}]*)\}\{(\d+)\}/g)) {
    if (match[1].startsWith('sub@')) continue
    out.set(match[1], { value: match[2].replace(/\\relax\s*/g, '').trim(), page: Number(match[3]) })
  }
  return out
}

export function parseToc(text: string): Map<number, number> {
  const out = new Map<number, number>()
  for (const match of text.matchAll(
    /\\contentsline \{section\}\{\\tocsection \{\}\{(\d+)\}\{.*\}\}\{(\d+)\}/g,
  )) {
    out.set(Number(match[1]), Number(match[2]))
  }
  if (out.size === 0) {
    for (const match of text.matchAll(
      /\\contentsline \{section\}\{\\numberline \{(\d+)\}.*\}\{(\d+)\}/g,
    )) {
      out.set(Number(match[1]), Number(match[2]))
    }
  }
  return out
}

/** Page → first source line (input 1) from a synctex file. */
export function parseSynctex(text: string): { page: number; firstLine: number }[] {
  const pages: { page: number; firstLine: number }[] = []
  let current: { page: number; firstLine: number } | null = null
  const content = text.slice(text.indexOf('\nContent:'))
  for (const line of content.split('\n')) {
    if (line.startsWith('{')) {
      current = { page: Number(line.slice(1)), firstLine: Number.MAX_SAFE_INTEGER }
      continue
    }
    if (line.startsWith('}')) {
      if (current) pages.push(current)
      current = null
      continue
    }
    if (!current) continue
    const match = /^[[(hvxkg$]1,(\d+):/.exec(line)
    if (match) current.firstLine = Math.min(current.firstLine, Number(match[1]))
  }
  return pages
    .filter(page => page.firstLine !== Number.MAX_SAFE_INTEGER)
    .sort((a, b) => a.page - b.page)
}

export function pageOfLine(
  pages: { page: number; firstLine: number }[],
  line: number,
): number | null {
  let result: number | null = null
  for (const page of pages) {
    if (page.firstLine <= line) result = page.page
    else break
  }
  return result
}

// ---------------------------------------------------------------------------
// filesystem pass

async function exists(file: string): Promise<boolean> {
  try {
    await fs.access(file)
    return true
  } catch {
    return false
  }
}

async function which(tool: string): Promise<boolean> {
  try {
    await run('which', [tool])
    return true
  } catch {
    return false
  }
}

function yamlQuote(value: string): string {
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
const problemSetNumberRe = /\b(?:pset|problem set|ps)[\s_#]*0*(\d+)\b/i

interface ZipSources {
  dir: string
  aux: Map<string, { value: string; page: number }> | null
  toc: Map<number, number> | null
  synctex: { page: number; firstLine: number }[] | null
  lineOffset: number
  base: string
}

async function extractZip(zipPath: string, notesSource: string): Promise<ZipSources | null> {
  if (!(await which('unzip'))) return null
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'ocw-latex-'))
  await run('unzip', ['-o', '-q', zipPath, '-x', '__MACOSX/*', '-d', dir])
  const texFiles = await walk(dir, name => name.endsWith('.tex'))
  const notesBody = stripLeadingComments(notesSource)
  for (const file of texFiles) {
    const text = await fs.readFile(file, 'utf8')
    if (stripLeadingComments(text) !== notesBody) continue
    const base = path.basename(file, '.tex')
    const sourceDir = path.dirname(file)
    const lineOffset = leadingCommentLines(notesSource) - leadingCommentLines(text)
    const read = async (ext: string) => {
      const candidate = path.join(sourceDir, `${base}${ext}`)
      return (await exists(candidate)) ? fs.readFile(candidate) : null
    }
    const aux = await read('.aux')
    const toc = await read('.toc')
    const synctex = await read('.synctex.gz')
    return {
      dir: sourceDir,
      base,
      lineOffset,
      aux: aux ? parseAux(aux.toString('utf8')) : null,
      toc: toc ? parseToc(toc.toString('utf8')) : null,
      synctex: synctex ? parseSynctex(gunzipSync(synctex).toString('utf8')) : null,
    }
  }
  return null
}

function leadingCommentLines(text: string): number {
  let count = 0
  for (const line of text.split('\n')) {
    if (line.startsWith('%') || line.trim() === '') count++
    else break
  }
  return count
}

function stripLeadingComments(text: string): string {
  return text.split('\n').slice(leadingCommentLines(text)).join('\n').replace(/\s+$/, '')
}

async function walk(dir: string, accept: (name: string) => boolean): Promise<string[]> {
  const out: string[] = []
  for (const entry of await fs.readdir(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name)
    if (entry.isDirectory()) out.push(...(await walk(full, accept)))
    else if (accept(entry.name)) out.push(full)
  }
  return out.sort()
}

interface FigureJob {
  name: string
  kind: 'input' | 'inline' | 'graphics'
  body: string
  line: number
  section: ConvertedSection
}

async function compileFigures(
  jobs: FigureJob[],
  zip: ZipSources | null,
  preamble: Preamble,
  outDir: string,
  report: string[],
): Promise<Map<string, string | null>> {
  const results = new Map<string, string | null>()
  const haveTools = (await which('latex')) && (await which('dvisvgm'))
  const work = await fs.mkdtemp(path.join(os.tmpdir(), 'ocw-tikz-'))
  // latex and dvisvgm run inside `work`, so every output path must be absolute
  outDir = path.resolve(outDir)
  const preambleLines = [
    '\\documentclass[border=2pt]{standalone}',
    '\\usepackage{amssymb}',
    '\\usepackage{amsmath}',
    '\\usepackage{tikz}',
    '\\usepackage{pgfplots}',
    '\\usetikzlibrary{calc}',
    '\\usetikzlibrary{patterns}',
    ...preamble.colors,
    ...Array.from(
      preamble.operators,
      ([name, body]) => `\\DeclareMathOperator{\\${name}}{${body}}`,
    ),
    ...Array.from(preamble.macros, ([name, body]) => `\\newcommand{\\${name}}{${body}}`),
  ]
  const unique = new Map<string, FigureJob>()
  for (const job of jobs) if (!unique.has(job.name)) unique.set(job.name, job)
  const queue = Array.from(unique.values())
  let cursor = 0
  const worker = async () => {
    while (cursor < queue.length) {
      const job = queue[cursor++]
      const outName = `${job.name.replace(/\s+/g, '-')}`
      if (job.kind === 'graphics') {
        const source = zip ? await findFile(zip.dir, job.name) : null
        if (!source) {
          report.push(`image ${job.name} not in the zip (line ${job.line}); placeholder written`)
          results.set(job.name, null)
          continue
        }
        const target = path.join(outDir, `${outName}`)
        await fs.mkdir(outDir, { recursive: true })
        await fs.copyFile(source, target)
        results.set(job.name, path.basename(target))
        continue
      }
      const svgName = `${outName}.svg`
      const target = path.join(outDir, svgName)
      if (!haveTools) {
        if (await exists(target)) results.set(job.name, svgName)
        else throw new LatexGateError(`latex/dvisvgm missing and ${svgName} is not committed`)
        continue
      }
      let bodyLine: string
      if (job.kind === 'input') {
        const source = zip ? await findFile(zip.dir, `${job.name}.tex`) : null
        if (!source) {
          report.push(`\\input{${job.name}} not in the zip (line ${job.line}); placeholder written`)
          results.set(job.name, null)
          continue
        }
        const body = await fs.readFile(source, 'utf8')
        if (
          !/\\(draw|fill|node|path|addplot|shade|clip|pgf|includegraphics)/.test(
            stripComments(body),
          )
        ) {
          report.push(`\\input{${job.name}} draws nothing (line ${job.line}); placeholder written`)
          results.set(job.name, null)
          continue
        }
        bodyLine = `\\input{${job.name}}`
      } else {
        if (!/\\(draw|fill|node|path|addplot|shade|clip|pgf)/.test(stripComments(job.body))) {
          report.push(`inline tikzpicture at line ${job.line} draws nothing; placeholder written`)
          results.set(job.name, null)
          continue
        }
        await fs.writeFile(path.join(work, `${job.name}.tex`), job.body)
        bodyLine = `\\input{${job.name}}`
      }
      const wrapper = `wrap-${outName}`
      await fs.writeFile(
        path.join(work, `${wrapper}.tex`),
        [...preambleLines, '\\begin{document}', bodyLine, '\\end{document}', ''].join('\n'),
      )
      const texInputs = `.:${zip ? `${zip.dir}//:` : ''}`
      try {
        await run(
          'latex',
          ['-no-shell-escape', '-interaction=nonstopmode', '-halt-on-error', `${wrapper}.tex`],
          { cwd: work, env: { ...process.env, TEXINPUTS: texInputs }, maxBuffer: 64 * 1024 * 1024 },
        )
        await run('dvisvgm', ['--no-fonts', '--exact-bbox', '-o', target, `${wrapper}.dvi`], {
          cwd: work,
          maxBuffer: 64 * 1024 * 1024,
        })
        await fs.mkdir(outDir, { recursive: true })
        const svg = await fs.readFile(target, 'utf8')
        // dvisvgm stamps nothing, but strip any comment so reruns stay byte-identical
        await fs.writeFile(target, svg.replace(/<!--[\s\S]*?-->\n?/g, ''))
        results.set(job.name, svgName)
      } catch (error) {
        const log = await fs.readFile(path.join(work, `${wrapper}.log`), 'utf8').catch(() => '')
        const reason =
          /^! (.*)$/m.exec(log)?.[1] ?? (error instanceof Error ? error.message : String(error))
        report.push(
          `figure ${job.name} failed to compile (line ${job.line}): ${reason}; placeholder written`,
        )
        results.set(job.name, null)
      }
    }
  }
  await Promise.all(Array.from({ length: 4 }, worker))
  await fs.rm(work, { recursive: true, force: true })
  return results
}

async function findFile(dir: string, name: string): Promise<string | null> {
  const direct = path.join(dir, name)
  if (await exists(direct)) return direct
  const files = await walk(dir, () => true)
  const base = path.basename(name).toLowerCase()
  return files.find(file => path.basename(file).toLowerCase() === base) ?? null
}

function pad(n: number): string {
  return String(n).padStart(2, '0')
}

function frontmatterYaml(fields: [string, string][]): string {
  return ['---', ...fields.map(([key, value]) => `${key}: ${value}`), '---'].join('\n')
}

function flowMap(entries: [string, unknown][]): string {
  return `{${entries
    .filter(([, value]) => value !== undefined && value !== null && value !== '')
    .map(([key, value]) => `${key}: ${flow(value)}`)
    .join(', ')}}`
}

function flow(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(flow).join(', ')}]`
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  return yamlQuote(String(value))
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

function lectureNumber(resource: LatexResource): number | null {
  const match =
    path.posix.basename(resource.relDir).match(lectureNumberRe) ??
    resource.title.match(lectureNumberRe)
  return match ? Number(match[1]) : null
}

function problemSetNumber(resource: LatexResource): number | null {
  const match =
    path.posix.basename(resource.relDir).match(problemSetNumberRe) ??
    resource.title.match(problemSetNumberRe)
  return match ? Number(match[1]) : null
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}
