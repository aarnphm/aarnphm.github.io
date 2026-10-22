import katex from 'katex'
import type { LatexNode } from './parser'
import { customMacros, katexOptions } from '../../cfg'

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

export function plainText(nodes: LatexNode[]): string {
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
