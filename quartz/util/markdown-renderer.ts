import { marked } from "marked"
import DOMPurify from "dompurify"
import hljs from "highlight.js/lib/core"
import javascript from "highlight.js/lib/languages/javascript"
import typescript from "highlight.js/lib/languages/typescript"
import python from "highlight.js/lib/languages/python"
import rust from "highlight.js/lib/languages/rust"
import go from "highlight.js/lib/languages/go"
import bash from "highlight.js/lib/languages/bash"

hljs.registerLanguage("javascript", javascript)
hljs.registerLanguage("typescript", typescript)
hljs.registerLanguage("python", python)
hljs.registerLanguage("rust", rust)
hljs.registerLanguage("go", go)
hljs.registerLanguage("bash", bash)

const renderer = new marked.Renderer()

marked.use({
  gfm: true,
  breaks: true,
  renderer,
})

export function renderMarkdown(markdown: string): string {
  if (!markdown || !markdown.trim()) {
    return ""
  }
  let html = marked.parse(markdown, { async: false }) as string
  html = html.replace(
    /<pre><code class="language-(\w+)">([\s\S]*?)<\/code><\/pre>/g,
    (_match, lang: string, code: string) => {
      const decodedCode = code
        .replace(/&lt;/g, "<")
        .replace(/&gt;/g, ">")
        .replace(/&amp;/g, "&")
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
