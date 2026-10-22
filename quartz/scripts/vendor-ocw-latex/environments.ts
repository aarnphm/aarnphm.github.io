import { createHash } from 'node:crypto'
import type { Converter } from './converter-model'
import { escapeText, normalizeSpaces } from './inline'
import { Parser, type LatexNode } from './parser'
import { expandMacros, plainText, splitRows, type TheoremEnv } from './preamble'

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

export function displayMath(
  this: Converter,
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

export function envBlocks(
  this: Converter,
  node: Extract<LatexNode, { type: 'env' }>,
): string[] | null {
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

export function statement(
  this: Converter,
  node: Extract<LatexNode, { type: 'env' }>,
  theorem: TheoremEnv,
): string[] {
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

export function proof(this: Converter, node: Extract<LatexNode, { type: 'env' }>): string[] {
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

export function list(this: Converter, node: Extract<LatexNode, { type: 'env' }>): string[] {
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

export function figure(this: Converter, node: Extract<LatexNode, { type: 'env' }>): string[] {
  const outerStepped = this.figureStepped
  this.figureStepped = false
  this.subfigureLetter = 0
  const blocks = this.figureBody(node.children, true)
  this.figureStepped = outerStepped
  return blocks
}

export function subfigure(this: Converter, node: Extract<LatexNode, { type: 'env' }>): string[] {
  return this.figureBody(node.children, false)
}

export function figureBody(this: Converter, children: LatexNode[], parent: boolean): string[] {
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
