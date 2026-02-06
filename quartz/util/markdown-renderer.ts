import DOMPurify from 'dompurify'
import hljs from 'highlight.js/lib/core'
import bash from 'highlight.js/lib/languages/bash'
import go from 'highlight.js/lib/languages/go'
import javascript from 'highlight.js/lib/languages/javascript'
import python from 'highlight.js/lib/languages/python'
import rust from 'highlight.js/lib/languages/rust'
import typescript from 'highlight.js/lib/languages/typescript'
import { marked } from 'marked'
import { stripSlashes, splitAnchor, resolveRelative, type FullSlug } from './path'
import { extractWikilinks, resolveWikilinkTarget } from './wikilinks'

hljs.registerLanguage('javascript', javascript)
hljs.registerLanguage('typescript', typescript)
hljs.registerLanguage('python', python)
hljs.registerLanguage('rust', rust)
hljs.registerLanguage('go', go)
hljs.registerLanguage('bash', bash)

const renderer = new marked.Renderer()

marked.use({
  gfm: true,
  breaks: true,
  renderer,
  extensions: [
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

export function renderMarkdown(markdown: string, currentSlug?: FullSlug): string {
  if (!markdown || !markdown.trim()) {
    return ''
  }

  let processedMarkdown = markdown
  if (currentSlug) {
    processedMarkdown = processWikilinks(markdown, currentSlug)
  }

  let html = marked.parse(processedMarkdown, { async: false }) as string
  html = html.replace(
    /<pre><code class="language-(\w+)">([\s\S]*?)<\/code><\/pre>/g,
    (_match, lang: string, code: string) => {
      const decodedCode = code
        .replace(/&lt;/g, '<')
        .replace(/&gt;/g, '>')
        .replace(/&amp;/g, '&')
        .replace(/&quot;/g, '"')
      if (lang && hljs.getLanguage(lang)) {
        try {
          const highlighted = hljs.highlight(decodedCode, { language: lang }).value
          return `<pre><code class="hljs language-${lang}">${highlighted}</code></pre>`
        } catch {
          return `<pre><code class="hljs">${decodedCode}</code></pre>`
        }
      }
      return `<pre><code class="hljs">${decodedCode}</code></pre>`
    },
  )
  return DOMPurify.sanitize(html)
}
