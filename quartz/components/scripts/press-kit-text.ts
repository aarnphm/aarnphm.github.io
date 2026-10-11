export type RunStyle = 'roman' | 'italic' | 'bold' | 'mono' | 'math'

/** A styled stretch of text. A run whose text is `\n` is a paragraph break. */
export interface Run {
  text: string
  style: RunStyle
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
  if (tag === 'STRONG' || tag === 'B') return 'bold'
  if ((tag === 'EM' || tag === 'I' || tag === 'CITE') && inherited !== 'bold') return 'italic'
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
    let text = run.text.replace(/\s+/g, ' ')
    const last = out.at(-1)
    if (!last || last.text === BREAK) text = text.trimStart()
    else if (last.text.endsWith(' ') && text.startsWith(' ')) text = text.slice(1)
    if (text.length === 0) continue
    if (last && last.text !== BREAK && last.style === run.style) last.text += text
    else out.push({ text, style: run.style })
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
      if (tex) out.push({ text: tex, style: 'math' })
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
  if (startMath) widened.setStartBefore(startMath)
  if (endMath) widened.setEndAfter(endMath)
  return runsFromNode(
    widened.cloneContents(),
    inheritedStyle(widened.commonAncestorContainer, stop),
  )
}

const MARK: Record<RunStyle, string> = { roman: '', italic: '*', bold: '**', mono: '`', math: '$' }

/** Serialises runs to the small markup that `runsFromText` reads back. */
export function runsToText(runs: Run[]): string {
  return runs
    .map(run => {
      if (run.text === BREAK) return '\n\n'
      const mark = MARK[run.style]
      const inner = run.style === 'roman' ? run.text : run.text.trim()
      if (!mark || inner.length === 0) return run.text
      const lead = run.text.match(/^\s*/)![0]
      const trail = run.text.match(/\s*$/)![0]
      return `${lead}${mark}${inner}${mark}${trail}`
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

// `_` needs word boundaries so snake_case survives.
const INLINE = /(\*\*[^*\n]+\*\*|\*[^*\n]+\*|(?<!\w)_[^_\n]+_(?!\w)|`[^`\n]+`|\$[^$\n]+\$)/

/** Reads `*em*`, `_em_`, `**strong**`, `` `code` `` and `$tex$`; blank lines break paragraphs. */
export function runsFromText(text: string): Run[] {
  const runs: Run[] = []
  text
    .split(/\n\s*\n/)
    .map(part => part.trim())
    .filter(Boolean)
    .forEach((part, index) => {
      if (index > 0) runs.push({ text: BREAK, style: 'roman' })
      for (const token of part.split(INLINE)) {
        if (!token) continue
        if (token.startsWith('**') && token.endsWith('**') && token.length > 4)
          runs.push({ text: token.slice(2, -2), style: 'bold' })
        else if (/^([*_]).+\1$/.test(token))
          runs.push({ text: token.slice(1, -1), style: 'italic' })
        else if (/^`.+`$/.test(token)) runs.push({ text: token.slice(1, -1), style: 'mono' })
        else if (/^\$.+\$$/.test(token)) runs.push({ text: token.slice(1, -1), style: 'math' })
        else runs.push({ text: token, style: 'roman' })
      }
    })
  return normalizeRuns(runs)
}
