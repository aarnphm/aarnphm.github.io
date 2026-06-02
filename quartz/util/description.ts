import type { KatexOptions } from 'katex'
import katex from 'katex'
import { isFullSlug, type FullSlug } from './path'
import { extractWikilinks, resolveWikilinkTarget, type WikilinkData } from './wikilinks'

const defaultKatexOptions: Omit<KatexOptions, 'output'> = { strict: true, throwOnError: true }
const latexCommandText: Record<string, string> = {
  cdot: '*',
  ge: '>=',
  geq: '>=',
  infty: 'infinity',
  le: '<=',
  leq: '<=',
  left: '',
  right: '',
  mathcal: '',
  mathbb: '',
  mathbf: '',
  mathit: '',
  mathrm: '',
  operatorname: '',
  text: '',
  times: 'x',
  top: '^T',
}

export function renderLatexInString(
  text: string,
  options: Omit<KatexOptions, 'output'> = defaultKatexOptions,
): string {
  let result = text

  const blockMathRegex = /\$\$([\s\S]*?)\$\$/g
  result = result.replace(blockMathRegex, (match, math) => {
    try {
      return katex.renderToString(math.trim(), { ...options, displayMode: true })
    } catch {
      return match
    }
  })

  const inlineMathRegex = /(?<!\$)\$([^$\n]+?)\$(?!\$)/g
  result = result.replace(inlineMathRegex, (match, math) => {
    try {
      return katex.renderToString(math.trim(), { ...options, displayMode: false })
    } catch {
      return match
    }
  })

  return result
}

function wikilinkDisplayText(link: WikilinkData, fallback = ''): string {
  return link.alias || link.anchorText || link.target || fallback
}

export function processWikilinksToHtml(text: string, currentSlug: FullSlug): string {
  const wikilinks = extractWikilinks(text)
  let result = text

  for (const link of wikilinks) {
    const resolved = resolveWikilinkTarget(link, currentSlug)
    if (resolved) {
      const displayText = wikilinkDisplayText(link, resolved.slug)
      const href = `/${resolved.slug}${resolved.anchor || ''}`
      const htmlLink = `<a href="${href}" class="internal">${displayText}</a>`
      result = result.replace(link.raw, htmlLink)
    }
  }

  return result
}

function processWikilinksToText(text: string, currentSlug?: FullSlug | string): string {
  const wikilinks = extractWikilinks(text)
  let result = text
  const resolvedSlug =
    typeof currentSlug === 'string' && isFullSlug(currentSlug) ? currentSlug : undefined

  for (const link of wikilinks) {
    const resolved = resolvedSlug ? resolveWikilinkTarget(link, resolvedSlug) : undefined
    const displayText = wikilinkDisplayText(link, resolved?.slug)
    result = result.replace(link.raw, displayText)
  }

  return result
}

function stripHtml(text: string): string {
  let output = ''
  let inTag = false
  let quote: string | undefined
  for (const char of text) {
    if (inTag) {
      if (quote) {
        if (char === quote) quote = undefined
      } else if (char === '"' || char === "'") {
        quote = char
      } else if (char === '>') {
        inTag = false
      }
      continue
    }
    if (char === '<') {
      inTag = true
      continue
    }
    output += char
  }
  return output
}

function htmlEntitiesToText(text: string): string {
  return text
    .replaceAll('&amp;', '&')
    .replaceAll('&lt;', '<')
    .replaceAll('&gt;', '>')
    .replaceAll('&quot;', '"')
    .replaceAll('&#39;', "'")
}

function latexCommandToText(command: string): string {
  return latexCommandText[command] ?? command
}

function latexToText(math: string): string {
  let text = math.trim()
  let previous = ''
  while (text !== previous) {
    previous = text
    text = text.replace(
      /\\(mathcal|mathbb|mathbf|mathit|mathrm|operatorname|text)\s*\{([^{}]*)\}/g,
      '$2',
    )
    text = text.replace(/\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}/g, '($1)/($2)')
    text = text.replace(/\\sqrt\s*\{([^{}]*)\}/g, 'sqrt($1)')
  }
  return text
    .replace(/\\([A-Za-z]+)/g, (_match, command: string) => latexCommandToText(command))
    .replace(/\\([{}])/g, '$1')
    .replace(/[{}]/g, '')
    .replace(/\s+/g, ' ')
    .trim()
}

export function descriptionToPlainText(
  description: string | undefined,
  currentSlug?: FullSlug | string,
) {
  if (!description) return ''
  const text = htmlEntitiesToText(stripHtml(processWikilinksToText(description, currentSlug)))
  return text
    .replace(/\$\$([\s\S]*?)\$\$/g, (_match, math: string) => latexToText(math))
    .replace(/(?<!\$)\$([^$\n]+?)\$(?!\$)/g, (_match, math: string) => latexToText(math))
    .replace(/\s+/g, ' ')
    .trim()
}
