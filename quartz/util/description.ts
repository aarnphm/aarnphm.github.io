import type { KatexOptions } from 'katex'
import { fromHtmlIsomorphic } from 'hast-util-from-html-isomorphic'
import { toHtml } from 'hast-util-to-html'
import { toText } from 'hast-util-to-text'
import isAbsoluteUrl from 'is-absolute-url'
import katex from 'katex'
import { fromMarkdown } from 'mdast-util-from-markdown'
import { mathFromMarkdown } from 'mdast-util-math'
import { toHast } from 'mdast-util-to-hast'
import { math } from 'micromark-extension-math'
import { visit } from 'unist-util-visit'
import { escapeHTML } from './escape'
import {
  isFullSlug,
  simplifySlug,
  slugifyFilePath,
  stripSlashes,
  type FilePath,
  type FullSlug,
} from './path'
import { isRecord } from './type-guards'
import {
  extractWikilinks,
  parseWikilink,
  resolveWikilinkTarget,
  type WikilinkData,
} from './wikilinks'

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
  if (!text.includes('[[')) return text
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

function processMarkdownLinksToHtml(text: string): string {
  if (!text.includes('[') && !/<(?:https?:|mailto:)/i.test(text)) return text
  const tree = fromMarkdown(text, { extensions: [math()], mdastExtensions: [mathFromMarkdown()] })
  const html = toHast(tree, {
    allowDangerousHtml: true,
    handlers: {
      // The description math pass needs the source before HTML entity encoding.
      inlineMath: (_state, node) => ({ type: 'raw', value: `$${node.value}$` }),
      math: (_state, node) => ({ type: 'raw', value: `$$${node.value}$$` }),
    },
  })
  const replacements: { start: number; end: number; html: string }[] = []
  visit(html, 'element', node => {
    if (node.tagName !== 'a') return
    const start = node.position?.start.offset
    const end = node.position?.end.offset
    const href = node.properties.href
    if (start === undefined || end === undefined || typeof href !== 'string') return
    node.properties.className = [isAbsoluteUrl(href, { httpOnly: false }) ? 'external' : 'internal']
    replacements.push({ start, end, html: toHtml(node, { allowDangerousHtml: true }) })
  })
  // Replace source ranges from the end, preserving authored HTML, math, and surrounding text.
  let result = text
  for (const replacement of replacements.sort((a, b) => b.start - a.start)) {
    result = result.slice(0, replacement.start) + replacement.html + result.slice(replacement.end)
  }
  return result
}

export function renderDescriptionHtml(text: string, currentSlug: FullSlug): string {
  return renderLatexInString(processWikilinksToHtml(processMarkdownLinksToHtml(text), currentSlug))
}

function processWikilinksToText(text: string, currentSlug?: FullSlug | string): string {
  if (!text.includes('[[')) return text
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
  if (
    !description.includes('[') &&
    !description.includes('$') &&
    !description.includes('<') &&
    !description.includes('&')
  ) {
    return description.replace(/\s+/g, ' ').trim()
  }
  // Escape math before parsing HTML so comparisons remain text rather than becoming tags.
  const text = processWikilinksToText(processMarkdownLinksToHtml(description), currentSlug)
    .replace(/\$\$([\s\S]*?)\$\$/g, (_match, math: string) => escapeHTML(latexToText(math)))
    .replace(/(?<!\$)\$([^$\n]+?)\$(?!\$)/g, (_match, math: string) =>
      escapeHTML(latexToText(math)),
    )
  return toText(fromHtmlIsomorphic(text, { fragment: true }))
    .replace(/\s+/g, ' ')
    .trim()
}

export function parseDescriptionManifest(raw: unknown): Map<string, string> {
  const entries = new Map<string, string>()
  const records = Array.isArray(raw) ? raw.filter(isRecord) : isRecord(raw) ? [raw] : []
  const rootSlug = 'index' as FullSlug

  for (const record of records) {
    for (const [rawKey, rawValue] of Object.entries(record)) {
      if (typeof rawValue !== 'string') continue
      const description = rawValue.trim()
      const trimmedKey = rawKey.trim()
      if (!trimmedKey || !description) continue

      const wikilink =
        parseWikilink(trimmedKey) ??
        parseWikilink(`[[${trimmedKey.replace(/^!?\[\[|\]\]$/g, '')}]]`)
      const resolved = wikilink ? resolveWikilinkTarget(wikilink, rootSlug) : null
      if (resolved) {
        entries.set(resolved.slug, description)
        entries.set(simplifySlug(resolved.slug), description)
      }

      if (!wikilink?.anchor) {
        const rawTarget = stripSlashes((wikilink?.target ?? trimmedKey).trim())
        if (rawTarget) {
          const slugWithExt = slugifyFilePath(rawTarget as FilePath)
          const slugWithoutExt = slugifyFilePath(rawTarget as FilePath, true)
          entries.set(slugWithExt, description)
          entries.set(simplifySlug(slugWithExt), description)
          entries.set(slugWithoutExt, description)
          entries.set(simplifySlug(slugWithoutExt), description)
        }
      }
    }
  }

  return entries
}

export function resolveManifestDescription(
  descriptions: ReadonlyMap<string, string>,
  slug: FullSlug,
  relativePath?: FilePath,
): string | undefined {
  return (
    descriptions.get(slug) ??
    descriptions.get(simplifySlug(slug)) ??
    (relativePath
      ? (descriptions.get(slugifyFilePath(relativePath)) ??
        descriptions.get(slugifyFilePath(relativePath, true)))
      : undefined)
  )
}
