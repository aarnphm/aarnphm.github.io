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

export const rawEnvs = new Set([
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

export class Parser {
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
