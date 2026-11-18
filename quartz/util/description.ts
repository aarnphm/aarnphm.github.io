import type { ElementContent } from 'hast'
import type { KatexOptions } from 'katex'
import type { Extension } from 'mdast-util-from-markdown'
import { fromHtmlIsomorphic } from 'hast-util-from-html-isomorphic'
import { toHtml } from 'hast-util-to-html'
import { toText } from 'hast-util-to-text'
import isAbsoluteUrl from 'is-absolute-url'
import katex from 'katex'
import { fromMarkdown } from 'mdast-util-from-markdown'
import { mathFromMarkdown } from 'mdast-util-math'
import { toHast, type Handler } from 'mdast-util-to-hast'
import { math } from 'micromark-extension-math'
import { visit } from 'unist-util-visit'
import type { ReferenceSyntax } from './references/model'
import { wikilink } from '../extensions/micromark-extension-ofm-wikilinks/syntax'
import { escapeHTML } from './escape'
import { isFullSlug, simplifySlug, type FilePath, type FullSlug } from './path'
import { labelToHtml, labelToText, parseReferenceLabel } from './references/label'
import {
  extractLiteralReferences,
  parseLiteralReference,
  resolveLiteralReference,
  referenceLabelSource,
  type LiteralReferenceResolutionOptions,
} from './references/literal'
import { referenceFields } from './references/request'
import { isRecord } from './type-guards'

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

function wikilinkLabel(link: ReferenceSyntax, fallback = '') {
  return parseReferenceLabel(referenceLabelSource(link, fallback))
}

export function processWikilinksToHtml(
  text: string,
  currentSlug: FullSlug,
  options?: LiteralReferenceResolutionOptions,
): string {
  if (!text.includes('[[')) return text
  const wikilinks = extractLiteralReferences(text)
  let result = text

  for (const link of wikilinks) {
    const resolved = resolveLiteralReference(link, currentSlug, options)
    if (resolved) {
      const href = `/${resolved.slug}${resolved.anchor || ''}`
      const htmlLink = `<a href="${escapeHTML(href)}" class="internal">${labelToHtml(wikilinkLabel(link, resolved.slug))}</a>`
      result = result.replace(link.raw, htmlLink)
    }
  }

  return result
}

function descriptionWikilinks(
  currentSlug: FullSlug | undefined,
  plainText: boolean,
  options?: LiteralReferenceResolutionOptions,
): Extension {
  return {
    enter: {
      wikilink() {
        this.buffer()
      },
    },
    exit: {
      wikilink(token) {
        this.resume()
        const source = this.sliceSerialize(token)
        const link = parseLiteralReference(source)
        let value = escapeHTML(source)
        if (link) {
          const resolved = currentSlug
            ? resolveLiteralReference(link, currentSlug, options)
            : undefined
          const label = wikilinkLabel(link, resolved?.slug)
          const content = plainText ? escapeHTML(labelToText(label)) : labelToHtml(label)
          value =
            resolved && !plainText
              ? `<a href="${escapeHTML(`/${resolved.slug}${resolved.anchor || ''}`)}" class="internal">${content}</a>`
              : content
        }
        this.enter({ type: 'html', value, data: { referenceProjection: true } }, token)
        this.exit(token)
      },
    },
  }
}

function renderDescriptionMarkup(
  text: string,
  currentSlug: FullSlug | undefined,
  plainText: boolean,
  options?: LiteralReferenceResolutionOptions,
  literalHtml = false,
): string {
  const tree = fromMarkdown(text, {
    extensions: [math(), wikilink()],
    mdastExtensions: [mathFromMarkdown(), descriptionWikilinks(currentSlug, plainText, options)],
  })
  const renderMath = (source: string, displayMode: boolean): ElementContent => {
    if (plainText) return { type: 'text', value: latexToText(source) }
    return {
      type: 'raw',
      value: renderLatexInString(displayMode ? `$$${source}$$` : `$${source}$`),
    }
  }
  const literalHtmlHandler: Handler = (_state, node) => {
    if (node.type !== 'html') return
    return node.data?.referenceProjection === true
      ? { type: 'raw', value: node.value }
      : { type: 'text', value: node.value }
  }
  const html = toHast(tree, {
    allowDangerousHtml: true,
    handlers: {
      ...(literalHtml ? { html: literalHtmlHandler } : {}),
      inlineMath: (_state, node) => renderMath(node.value, false),
      math: (_state, node) => renderMath(node.value, true),
    },
  })
  if (!html || html.type !== 'root') return ''
  visit(html, 'element', node => {
    if (node.tagName !== 'a' || typeof node.properties.href !== 'string') return
    node.properties.className = [
      isAbsoluteUrl(node.properties.href, { httpOnly: false }) ? 'external' : 'internal',
    ]
  })
  html.children = html.children.flatMap(node =>
    node.type === 'element' && node.tagName === 'p' ? node.children : [node],
  )
  return toHtml(html, { allowDangerousHtml: true })
}

export function renderDescriptionHtml(
  text: string,
  currentSlug: FullSlug,
  options?: LiteralReferenceResolutionOptions,
): string {
  return renderDescriptionMarkup(text, currentSlug, false, options)
}

export function renderInlineReferenceHtml(
  text: string,
  currentSlug: FullSlug,
  options?: LiteralReferenceResolutionOptions,
): string {
  if (!/[[\]$<>&*_`\\]/.test(text)) return escapeHTML(text)
  return renderDescriptionMarkup(text, currentSlug, false, options, true)
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
  options?: LiteralReferenceResolutionOptions,
) {
  if (!description) return ''
  if (!/[[\]$<>&*_`\\]/.test(description)) return description.replace(/\s+/g, ' ').trim()
  const slug = typeof currentSlug === 'string' && isFullSlug(currentSlug) ? currentSlug : undefined
  const text = renderDescriptionMarkup(description, slug, true, options)
  return toText(fromHtmlIsomorphic(text, { fragment: true }))
    .replace(/\s+/g, ' ')
    .trim()
}

export function parseDescriptionManifest(
  raw: unknown,
  options?: LiteralReferenceResolutionOptions,
): Map<string, string> {
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
        parseLiteralReference(trimmedKey) ??
        parseLiteralReference(`[[${trimmedKey.replace(/^!?\[\[|\]\]$/g, '')}]]`)
      const resolved = wikilink ? resolveLiteralReference(wikilink, rootSlug, options) : null
      if (resolved) {
        entries.set(resolved.slug, description)
        entries.set(simplifySlug(resolved.slug), description)
      }

      const fields = wikilink ? referenceFields(wikilink) : undefined
      if (!fields?.fragments.length) {
        const source = (fields?.target ?? trimmedKey).trim().replace(/^\/+/, '')
        if (source) entries.set(source, description)
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
    (relativePath ? descriptions.get(relativePath) : undefined)
  )
}

declare module 'mdast' {
  interface HtmlData {
    referenceProjection?: boolean
  }
}
