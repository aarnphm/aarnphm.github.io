// Figure labels often wrap plain words in \text{} so a whole label can go through KaTeX. Words
// then render in KaTeX's serif inside sans figures. This splits a label at its top-level word
// groups: words become escaped text in the figure's own font, and only the math between them
// goes to KaTeX. A label with no top-level \text group is math and stays whole.

type Part = { kind: 'text' | 'math'; value: string }

const WORD_COMMANDS = ['\\text', '\\mathrm']

// Math runs that hold only spacing, digits and punctuation read as text too.
const SPACING: [RegExp, string][] = [
  [/\\quad|\\qquad/g, ' '],
  [/\\[,;:! ]/g, ' '],
  [/~/g, ' '],
  [/\{\\cdot\}|\\cdot/g, '·'],
  [/\{\\times\}|\\times/g, '×'],
]
const OPERAND_BEFORE = /(?:[=([{+]|\\(?:mapsto|to|perp|in|le|ge|approx|ll|gg))$/
const TRAILING_SPACE = /(?:\s|\\[,;:! ]|\\quad)+$/
const LEADING_SPACE = /^(?:\s|\\[,;:! ]|\\quad)+/
const PLAIN_MATH = /^[\s\d.,:;/()+\-·×]*$/

const unescapeText = (value: string): string => value.replace(/\\([%&$#_{}])/g, '$1')

const escapeHtml = (value: string): string =>
  value.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;')

const readGroup = (tex: string, open: number): number => {
  let depth = 0
  for (let i = open; i < tex.length; i++) {
    if (tex[i] === '\\') {
      i++
      continue
    }
    if (tex[i] === '{') depth++
    else if (tex[i] === '}' && --depth === 0) return i
  }
  return -1
}

const splitLabel = (tex: string): Part[] | null => {
  const parts: Part[] = []
  let math = ''
  let depth = 0
  let words = false
  for (let i = 0; i < tex.length; i++) {
    const command = depth === 0 && WORD_COMMANDS.find(c => tex.startsWith(`${c}{`, i))
    if (command) {
      const open = i + command.length
      const close = readGroup(tex, open)
      if (close < 0) return null
      // A word group carrying a sub- or superscript is a math symbol, such as \text{head}_i.
      if (/[_^]/.test(tex[close + 1] ?? '') || /[_^]\s*$/.test(math)) {
        math += tex.slice(i, close + 1)
        i = close
        continue
      }
      // A word right after an operator or an open bracket is an operand: p\mapsto\text{emotion}.
      if (OPERAND_BEFORE.test(math.replace(TRAILING_SPACE, ''))) return null
      if (math) parts.push({ kind: 'math', value: math })
      math = ''
      parts.push({ kind: 'text', value: unescapeText(tex.slice(open + 1, close)) })
      words ||= command === '\\text'
      i = close
      continue
    }
    const c = tex[i]
    if (c === '\\') {
      math += tex.slice(i, i + 2)
      i++
      continue
    }
    if (c === '{') depth++
    else if (c === '}') depth--
    math += c
  }
  if (math) parts.push({ kind: 'math', value: math })
  // \mathrm next to math is an operator name; a label made only of upright words is text.
  if (!words && parts.some(part => part.kind === 'math')) return null
  const merged: Part[] = []
  const push = (next: Part) => {
    const last = merged.at(-1)
    if (last?.kind === 'text' && next.kind === 'text') last.value += next.value
    else merged.push({ ...next })
  }
  for (const part of parts) {
    if (part.kind === 'text') {
      push(part)
      continue
    }
    const plain = SPACING.reduce(
      (value, [pattern, replace]) => value.replace(pattern, replace),
      part.value,
    )
    if (PLAIN_MATH.test(plain)) {
      push({ kind: 'text', value: plain })
      continue
    }
    // Spacing at the edge of a math run becomes a text space, so gaps beside words match.
    const core = part.value.replace(LEADING_SPACE, '').replace(TRAILING_SPACE, '')
    if (core !== part.value.replace(TRAILING_SPACE, '')) push({ kind: 'text', value: ' ' })
    push({ kind: 'math', value: core })
    if (core !== part.value.replace(LEADING_SPACE, '')) push({ kind: 'text', value: ' ' })
  }
  return merged
}

/**
 * Render a figure label as HTML: words as text, math through `renderMath`. A lone "-" (the
 * missing-value marker) becomes an en dash.
 */
export const renderFigureLabel = (
  tex: string,
  renderMath: (math: string) => string,
  display = false,
): string => {
  const parts = display ? null : splitLabel(tex)
  if (!parts) return renderMath(tex)
  return parts
    .map((part, index) => {
      if (part.kind === 'math') return renderMath(part.value)
      let text = part.value.replace(/\s+/g, ' ')
      if (text.trim() === '-') return '–'
      // Flex label boxes trim a space at the edge of a text run, so a space beside math is
      // a no-break space; one at the label's edge is dropped.
      text = index === 0 ? text.trimStart() : text.replace(/^ /, ' ')
      text = index === parts.length - 1 ? text.trimEnd() : text.replace(/ $/, ' ')
      return escapeHtml(text)
    })
    .join('')
}
