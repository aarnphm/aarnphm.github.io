import DOMPurify from 'dompurify'
import katex from 'katex'
import { Marked } from 'marked'
import { customMacros } from '../cfg'
import { escapeHTML } from './escape'
import { resolveRelative, type FullSlug } from './path'
import { labelToHtml, parseReferenceLabel } from './references/label'
import { referenceLabelSource } from './references/literal'
import { referenceRequest } from './references/request'
import { missingReferenceRoute } from './references/route'
import { readReferenceAt } from './references/scan'

// Comments take the notes' math syntax, rendered with the notes' macros. Inline math follows
// pandoc's rule: no space inside either dollar and no digit after the closing one, so "$5 and $10"
// stays text.
const inlineMath = /^\$(?![\s$])((?:\\.|[^\\$\n])+?)(?<!\s)\$(?!\d)/
const displayMath = /^\$\$([\s\S]+?)\$\$/

const renderMath = (tex: string, displayMode: boolean) =>
  katex.renderToString(tex.trim(), {
    displayMode,
    throwOnError: false,
    macros: { ...customMacros },
  })

function markdownParser(currentSlug?: FullSlug): Marked {
  return new Marked({
    gfm: true,
    breaks: true,
    extensions: [
      {
        name: 'wikilink',
        level: 'inline',
        start: (src: string) => src.indexOf('[['),
        tokenizer(src: string) {
          if (!currentSlug) return
          const parsed = readReferenceAt(src, 0, { context: 'literal' })
          if (parsed.kind !== 'complete') return
          const reference = parsed.reference
          const request = referenceRequest(reference, currentSlug)
          // Comments have a page route, while the server owns the source catalog.
          const route = missingReferenceRoute(request.target, currentSlug, request.fragments)
          if (!route) return
          const href = `${resolveRelative(currentSlug, route.slug)}${(route.anchor ?? '').replace(/^#\^/, '#')}`
          const label = labelToHtml(
            parseReferenceLabel(referenceLabelSource(reference, route.slug)),
          )
          return {
            type: 'wikilink',
            raw: reference.raw,
            html: `<a href="${escapeHTML(href)}" class="internal-link">${label}</a>`,
          }
        },
        renderer(token) {
          return 'html' in token && typeof token.html === 'string' ? token.html : token.raw
        },
      },
      {
        name: 'displayMath',
        level: 'block',
        start: (src: string) => src.indexOf('$$'),
        tokenizer(src: string) {
          const match = src.match(displayMath)
          if (match) return { type: 'displayMath', raw: match[0], tex: match[1] }
        },
        renderer: token =>
          'tex' in token && typeof token.tex === 'string' ? renderMath(token.tex, true) : token.raw,
      },
      {
        name: 'inlineMath',
        level: 'inline',
        start: (src: string) => src.indexOf('$'),
        tokenizer(src: string) {
          const match = src.match(displayMath) ?? src.match(inlineMath)
          if (match) {
            return {
              type: 'inlineMath',
              raw: match[0],
              tex: match[1],
              display: match[0][1] === '$',
            }
          }
        },
        renderer(token) {
          return 'tex' in token && typeof token.tex === 'string'
            ? renderMath(token.tex, 'display' in token && token.display === true)
            : token.raw
        },
      },
      {
        name: 'mention',
        level: 'inline',
        start: (src: string) => src.indexOf('@'),
        tokenizer(src: string) {
          const match = src.match(/^@([\w-]+)/)
          if (match) {
            return { type: 'mention', raw: match[0], username: match[1] }
          }
        },
        renderer(token) {
          return 'username' in token && typeof token.username === 'string'
            ? `<a class="mention" href="https://github.com/${token.username}">@${token.username}</a>`
            : token.raw
        },
      },
    ],
  })
}

function renderMarkdown(markdown: string, currentSlug?: FullSlug): string {
  if (!markdown || !markdown.trim()) {
    return ''
  }

  const html = markdownParser(currentSlug).parse(markdown, { async: false })
  return DOMPurify.sanitize(html)
}

/** Writes sanitized comment markdown into `target`; code blocks gain syntax colour once the lazy highlighter loads. */
export function renderMarkdownInto(target: HTMLElement, markdown: string, currentSlug?: FullSlug) {
  target.innerHTML = renderMarkdown(markdown, currentSlug)
  const blocks = Array.from(target.querySelectorAll<HTMLElement>('pre > code[class^="language-"]'))
  if (blocks.length === 0) return
  // Until the chunk loads, or if it fails, the blocks stay as sanitized plain text.
  void import('./highlight-source').then(
    ({ codeAttributes, highlightCode }) => {
      for (const code of blocks) {
        const highlighted = highlightCode(
          (code.textContent ?? '').replace(/\n$/, ''),
          code.className.slice('language-'.length),
        )
        for (const [name, value] of Object.entries(codeAttributes(highlighted))) {
          code.setAttribute(name, value)
        }
        // Comment snippets are short; line numbers would crowd the bubble.
        code.dataset.disableLineNumber = ''
        code.innerHTML = highlighted.html
      }
    },
    () => {},
  )
}
