import type { Nodes } from 'mdast'
import { fromMarkdown } from 'mdast-util-from-markdown'
import { mathFromMarkdown } from 'mdast-util-math'
import { math } from 'micromark-extension-math'

export type RunStyle = 'roman' | 'italic' | 'bold' | 'bold-italic' | 'mono' | 'math'

/** A styled stretch of text. A run whose text is `\n` is a paragraph break. */
export interface Run {
  text: string
  style: RunStyle
  display?: boolean
}

export const BREAK = '\n'

// The rendered KaTeX tree is aria-hidden, so `.katex` is resolved before the generic skip.
const SKIP = [
  'script',
  'style',
  'svg',
  'button',
  'input',
  'textarea',
  'a[data-role="anchor"]',
  'sup:has(> [data-footnote-ref])',
  '[data-footnote-ref]',
  '.collapse-rail',
  '.collapsed-dots',
  '.popover',
  '.press-kit-pill',
  '[aria-hidden="true"]',
].join(', ')

const BLOCK = new Set([
  'ARTICLE',
  'BLOCKQUOTE',
  'DD',
  'DIV',
  'DT',
  'FIGCAPTION',
  'H1',
  'H2',
  'H3',
  'H4',
  'H5',
  'H6',
  'LI',
  'OL',
  'P',
  'PRE',
  'SECTION',
  'TR',
  'UL',
])

function styleFor(tag: string, inherited: RunStyle): RunStyle {
  if (tag === 'CODE' || tag === 'KBD' || tag === 'SAMP') return 'mono'
  if (inherited === 'mono' || inherited === 'math') return inherited
  if (tag === 'STRONG' || tag === 'B')
    return inherited === 'italic' || inherited === 'bold-italic' ? 'bold-italic' : 'bold'
  if (tag === 'EM' || tag === 'I' || tag === 'CITE')
    return inherited === 'bold' || inherited === 'bold-italic' ? 'bold-italic' : 'italic'
  return inherited
}

function texSource(katex: Element): string | undefined {
  return katex.querySelector('annotation[encoding="application/x-tex"]')?.textContent?.trim()
}

/** Collapses whitespace, merges neighbours of one style, and trims breaks at either end. */
export function normalizeRuns(runs: Run[]): Run[] {
  const out: Run[] = []
  for (const run of runs) {
    if (run.text === BREAK) {
      const last = out.at(-1)
      if (last && last.text !== BREAK) {
        last.text = last.text.trimEnd()
        out.push({ text: BREAK, style: 'roman' })
      }
      continue
    }
    if (run.style === 'math') {
      out.push({ ...run, text: run.text.trim() })
      continue
    }
    let text =
      run.style === 'mono' ? run.text.replace(/\r?\n/g, ' ') : run.text.replace(/\s+/g, ' ')
    const last = out.at(-1)
    if (!last || last.text === BREAK) text = text.trimStart()
    else if (last.text.endsWith(' ') && text.startsWith(' ')) text = text.slice(1)
    if (text.length === 0) continue
    if (last && last.text !== BREAK && last.style === run.style) last.text += text
    else out.push({ ...run, text })
  }
  while (out.at(-1)?.text === BREAK) out.pop()
  const tail = out.at(-1)
  if (tail) tail.text = tail.text.trimEnd()
  return out.filter(run => run.text.length > 0)
}

function inheritedStyle(node: Node | null, stop: Element | null): RunStyle {
  const chain: string[] = []
  let el: Element | null = node instanceof Element ? node : (node?.parentElement ?? null)
  while (el && el !== stop) {
    chain.push(el.tagName)
    el = el.parentElement
  }
  return chain.reduceRight<RunStyle>((style, tag) => styleFor(tag, style), 'roman')
}

export function runsFromNode(root: Node, base: RunStyle = 'roman'): Run[] {
  const out: Run[] = []
  const walk = (node: Node, style: RunStyle) => {
    if (node.nodeType === Node.TEXT_NODE) {
      out.push({ text: node.textContent ?? '', style })
      return
    }
    if (!(node instanceof Element)) {
      node.childNodes.forEach(child => walk(child, style))
      return
    }
    if (node.classList.contains('katex')) {
      const tex = texSource(node)
      if (tex) out.push({ text: tex, style: 'math', display: !!node.closest('.katex-display') })
      return
    }
    if (node.matches(SKIP)) return
    if (node.tagName === 'BR') {
      out.push({ text: BREAK, style })
      return
    }
    const block = BLOCK.has(node.tagName) || node.classList.contains('katex-display')
    if (block) out.push({ text: BREAK, style })
    const next = styleFor(node.tagName, style)
    node.childNodes.forEach(child => walk(child, next))
    if (block) out.push({ text: BREAK, style })
  }
  walk(root, base)
  return normalizeRuns(out)
}

/** A selection that cuts into rendered math widens to the whole formula. */
export function runsFromRange(range: Range, stop: Element | null): Run[] {
  const widened = range.cloneRange()
  const startEl =
    widened.startContainer instanceof Element
      ? widened.startContainer
      : widened.startContainer.parentElement
  const endEl =
    widened.endContainer instanceof Element
      ? widened.endContainer
      : widened.endContainer.parentElement
  const startMath = startEl?.closest('.katex')
  const endMath = endEl?.closest('.katex')
  if (startMath) widened.setStartBefore(startMath.closest('.katex-display') ?? startMath)
  if (endMath) widened.setEndAfter(endMath.closest('.katex-display') ?? endMath)
  return runsFromNode(
    widened.cloneContents(),
    inheritedStyle(widened.commonAncestorContainer, stop),
  )
}

const MARK: Record<RunStyle, string> = {
  roman: '',
  italic: '*',
  bold: '**',
  'bold-italic': '***',
  mono: '`',
  math: '$',
}

/** Serialises runs to the small markup that `runsFromText` reads back. */
export function runsToText(runs: Run[]): string {
  return runs
    .map(run => {
      if (run.text === BREAK) return '\n\n'
      if (run.style === 'math') return run.display ? `$$\n${run.text}\n$$` : `$${run.text}$`
      const mark =
        run.style === 'mono'
          ? '`'.repeat(Math.max(0, ...Array.from(run.text.matchAll(/`+/g), m => m[0].length)) + 1)
          : MARK[run.style]
      const inner = run.style === 'roman' ? run.text : run.text.trim()
      if (!mark || inner.length === 0) return run.text.replace(/([\\*_`$[\]])/g, '\\$1')
      const lead = run.text.match(/^\s*/)![0]
      const trail = run.text.match(/\s*$/)![0]
      const content = run.style === 'mono' ? inner : inner.replace(/([\\*_`$[\]])/g, '\\$1')
      const pad = run.style === 'mono' && inner.includes('`') ? ' ' : ''
      return `${lead}${mark}${pad}${content}${pad}${mark}${trail}`
    })
    .join('')
    .trim()
}

/** Plain text without markup, for share intents and text fragments. */
export function runsToPlain(runs: Run[]): string {
  return runs
    .map(run => (run.text === BREAK ? '\n\n' : run.style === 'math' ? `$${run.text}$` : run.text))
    .join('')
    .trim()
}

/**
 * A card's date, as formatDate (quartz/components/Date.tsx) prints a page's date in the site
 * locale. The calendar day is read from the ISO string, so a visitor's time zone cannot move it.
 */
export const cardDate = (iso: string): string =>
  new Date(`${iso.slice(0, 10)}T12:00:00`).toLocaleDateString(
    document.documentElement.lang || 'en-US',
    { year: 'numeric', month: 'short', day: '2-digit' },
  )

/** Parse the editor with the same Markdown and math syntax used by authored notes. */
export function runsFromText(text: string): Run[] {
  const runs: Run[] = []
  const tree = fromMarkdown(text, { extensions: [math()], mdastExtensions: [mathFromMarkdown()] })
  const lineBreak = () => runs.push({ text: BREAK, style: 'roman' })
  const walk = (node: Nodes, style: RunStyle = 'roman') => {
    if (node.type === 'text' || node.type === 'inlineCode') {
      runs.push({ text: node.value, style: node.type === 'inlineCode' ? 'mono' : style })
    } else if (node.type === 'inlineMath' || node.type === 'math') {
      const display = node.type === 'math'
      if (display) lineBreak()
      runs.push({ text: node.value, style: 'math', display })
      if (display) lineBreak()
    } else if (node.type === 'break') {
      lineBreak()
    } else if (node.type === 'code') {
      lineBreak()
      for (const line of node.value.split('\n')) {
        runs.push({ text: line, style: 'mono' })
        lineBreak()
      }
    } else if (node.type === 'image' || node.type === 'imageReference') {
      if (node.alt) runs.push({ text: node.alt, style })
    } else if ('children' in node) {
      const block = ['paragraph', 'heading', 'blockquote', 'listItem'].includes(node.type)
      if (block) lineBreak()
      const next =
        node.type === 'strong' || node.type === 'heading'
          ? styleFor('STRONG', style)
          : node.type === 'emphasis'
            ? styleFor('EM', style)
            : style
      for (const child of node.children) walk(child, next)
      if (block) lineBreak()
    }
  }
  walk(tree)
  return normalizeRuns(runs)
}
