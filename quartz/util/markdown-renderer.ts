import DOMPurify from 'dompurify'
import katex from 'katex'
import { marked } from 'marked'
import { customMacros } from '../cfg'
import { stripSlashes, splitAnchor, resolveRelative, type FullSlug } from './path'
import { extractWikilinks, resolveWikilinkTarget } from './wikilinks'

const renderer = new marked.Renderer()

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

marked.use({
  gfm: true,
  breaks: true,
  renderer,
  extensions: [
    {
      name: 'displayMath',
      level: 'block',
      start: (src: string) => src.indexOf('$$'),
      tokenizer(src: string) {
        const match = src.match(displayMath)
        if (match) return { type: 'displayMath', raw: match[0], tex: match[1] }
      },
      renderer: token => renderMath((token as unknown as { tex: string }).tex, true),
    },
    {
      name: 'inlineMath',
      level: 'inline',
      start: (src: string) => src.indexOf('$'),
      tokenizer(src: string) {
        const match = src.match(displayMath) ?? src.match(inlineMath)
        if (match) {
          return { type: 'inlineMath', raw: match[0], tex: match[1], display: match[0][1] === '$' }
        }
      },
      renderer(token) {
        const t = token as unknown as { tex: string; display: boolean }
        return renderMath(t.tex, t.display)
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
        const t = token as unknown as { username: string }
        return `<a class="mention" href="https://github.com/${t.username}">@${t.username}</a>`
      },
    },
  ],
})

function processWikilinks(text: string, currentSlug: FullSlug): string {
  const wikilinks = extractWikilinks(text)
  if (wikilinks.length === 0) return text

  let processed = text
  for (const link of wikilinks) {
    const resolved = resolveWikilinkTarget(link, currentSlug)
    if (!resolved) continue

    let dest = resolved.slug
    if (!dest.startsWith('/')) dest = `/${dest}` as FullSlug

    const url = new URL(dest, `https://base.com/${stripSlashes(currentSlug, true)}`)
    let canonicalDest = url.pathname
    let [destCanonical, _destAnchor] = splitAnchor(canonicalDest)

    if (destCanonical.endsWith('/')) {
      destCanonical += 'index'
    }

    const finalSlug = decodeURIComponent(stripSlashes(destCanonical, true)) as FullSlug
    const relativeHref = resolveRelative(currentSlug, finalSlug)
    const href = resolved.anchor ? `${relativeHref}${resolved.anchor}` : relativeHref

    const displayText = link.alias || link.target || resolved.slug
    const anchorTag = `<a href="${href}" class="internal-link">${displayText}</a>`

    processed = processed.replace(link.raw, anchorTag)
  }
  return processed
}

function renderMarkdown(markdown: string, currentSlug?: FullSlug): string {
  if (!markdown || !markdown.trim()) {
    return ''
  }

  let processedMarkdown = markdown
  if (currentSlug) {
    processedMarkdown = processWikilinks(markdown, currentSlug)
  }

  const html = marked.parse(processedMarkdown, { async: false }) as string
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
