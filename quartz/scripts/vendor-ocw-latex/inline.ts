import type { Converter } from './converter-model'
import { rawEnvs, type LatexNode } from './parser'
import { checkKatex, expandMacros, plainText } from './preamble'

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

export function escapeText(value: string): string {
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

export function normalizeSpaces(value: string): string {
  return value.replace(/[ \t]+/g, ' ')
}

export function titleCaseLower(value: string): string {
  // text runs lowercased, math untouched
  return value.replace(
    /(\$[^$]*\$)|([^$]+)/g,
    (_, math: string | undefined, text: string | undefined) => math ?? (text ?? '').toLowerCase(),
  )
}

export function blocks(this: Converter, nodes: LatexNode[]): string[] {
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

export function isBlockGroup(node: Extract<LatexNode, { type: 'group' }>): boolean {
  return node.children.some(
    child => child.type === 'par' || (child.type === 'env' && !rawEnvs.has(child.name)),
  )
}

export function cmdBlocks(
  this: Converter,
  node: Extract<LatexNode, { type: 'cmd' }>,
): string[] | null {
  switch (node.name) {
    case 'section':
    case 'section*':
    case 'subsection*':
    case 'subsubsection*': {
      // unnumbered headings; a pset \section is a heading since the file is one page
      const level = node.name.startsWith('section') ? 2 : node.name.startsWith('subsection') ? 3 : 4
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

export function figureImage(this: Converter, node: Extract<LatexNode, { type: 'cmd' }>): string {
  const name = plainText(node.args[node.name === 'input' ? 0 : 0]).trim()
  const kind = node.name === 'input' ? 'input' : 'graphics'
  this.section?.figures.push({ name, kind, body: '', line: node.line })
  return `\uE001FIG:${kind}:${name}:${node.line}\uE001`
}

export function gate(this: Converter, src: string, display: boolean, line: number): void {
  const error = checkKatex(src, display)
  if (error)
    this.katexErrors.push(
      `line ${line}: ${error.replace(/^KaTeX parse error: /, '')}\n    ${src.slice(0, 160)}`,
    )
}

export function inline(this: Converter, nodes: LatexNode[], mode: 'text' | 'title'): string {
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

export function accent(name: string, letter: string): string {
  const base = letter.replace(/\\i(?![A-Za-z])/g, 'i').trim()
  const table = accents[name]
  const mapped = table?.[base]
  if (mapped) return mapped
  return base
}

export function inlineCmd(
  this: Converter,
  node: Extract<LatexNode, { type: 'cmd' }>,
  mode: 'text' | 'title',
): string {
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
