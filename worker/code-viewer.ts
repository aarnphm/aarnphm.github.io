import { escapeHTML } from '../quartz/util/escape'
import { codeAttributes, highlightSourceCode } from '../quartz/util/highlight-source'
import { mergeVary, withHeaders } from './index/http-policy'
import { wantsCodeViewerHtml } from './request-utils'

const PAGE_STYLE = `
@font-face {
  font-family: 'Berkeley Mono';
  src: url('/fonts/BerkeleyMono-Regular.woff2') format('woff2');
  font-weight: 400;
  font-style: normal;
  font-display: swap;
}
:root {
  --light: #fffcf0;
  --lightgray: #e6e4d9;
  --gray: #b7b5ac;
  --darkgray: #6f6e69;
  --dark: #100f0f;
  --secondary: #205ea6;
  --highlight: rgba(143, 159, 169, 0.15);
  --code-font: 'Berkeley Mono', ui-monospace, 'SF Mono', Menlo, Monaco, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([saved-theme='light']) {
    --light: #100f0f;
    --lightgray: #282726;
    --gray: #575653;
    --darkgray: #878580;
    --dark: #cecdc3;
    --secondary: #4385be;
  }
}
:root[saved-theme='dark'] {
  --light: #100f0f;
  --lightgray: #282726;
  --gray: #575653;
  --darkgray: #878580;
  --dark: #cecdc3;
  --secondary: #4385be;
}
* {
  box-sizing: border-box;
}
body {
  margin: 0;
  background: var(--light);
  color: var(--dark);
  font: 13px/1.55 var(--code-font);
}
.code-viewer {
  max-width: 112ch;
  margin: 2rem auto 4rem;
  padding: 0 1.25rem;
}
.code-viewer-header {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  padding: 0.5rem 0.75rem;
  border: 1px solid var(--lightgray);
  border-bottom: 0;
  background: var(--light);
}
.code-viewer-path {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 0.15rem;
  min-width: 0;
  color: var(--darkgray);
  font-size: 13px;
}
.code-viewer-path a {
  color: var(--darkgray);
  text-decoration: none;
}
.code-viewer-path a:hover,
.code-viewer-path a:focus-visible {
  color: var(--dark);
  text-decoration: underline;
}
.code-viewer-filename {
  color: var(--dark);
  font-weight: 600;
}
.code-viewer-actions {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-left: auto;
}
.code-viewer-meta {
  color: var(--darkgray);
  font-size: 12px;
}
.code-viewer-action {
  appearance: none;
  display: inline-flex;
  align-items: center;
  padding: 0.15rem 0.5rem;
  border: 1px solid var(--lightgray);
  background: var(--light);
  color: var(--darkgray);
  font: 12px/1.3 var(--code-font);
  text-decoration: none;
  cursor: pointer;
}
.code-viewer-action:hover,
.code-viewer-action:focus-visible {
  border-color: var(--gray);
  color: var(--dark);
  outline: none;
}
.code-viewer figure[data-rehype-pretty-code-figure] {
  margin: 0;
}
.code-viewer pre {
  margin: 0;
  padding: 0;
  border: 1px solid var(--lightgray);
  overflow-x: auto;
  background: var(--light);
}
.code-viewer pre:focus-visible {
  outline: 1px solid var(--secondary);
  outline-offset: -1px;
}
.code-viewer pre > code {
  display: grid;
  padding: 0.5rem 0;
  font: 13px/1.55 var(--code-font);
  counter-reset: line;
  counter-increment: line 0;
  overflow-x: auto;
}
.code-viewer pre > code > [data-line] {
  padding: 0 0.75rem 0 0.5rem;
  white-space: pre;
}
.code-viewer pre > code > [data-line]:hover:not([data-highlighted-line]) {
  background-color: var(--lightgray);
}
.code-viewer pre > code > [data-line][data-highlighted-line] {
  background-color: var(--highlight);
}
.code-viewer pre > code > [data-line]::before {
  content: counter(line);
  counter-increment: line;
  width: calc(var(--line-digits, 2) * 1ch);
  margin-right: 1.25rem;
  display: inline-block;
  text-align: right;
  color: var(--darkgray);
  opacity: 0.65;
  user-select: none;
}
code[data-theme*=' '] {
  color: var(--syntax-light);
  background-color: var(--syntax-light-bg);
}
code[data-theme*=' '] span {
  color: var(--syntax-light);
  font-style: var(--syntax-light-font-style, normal);
  font-weight: var(--syntax-light-font-weight, normal);
}
@media (prefers-color-scheme: dark) {
  :root:not([saved-theme='light']) code[data-theme*=' '] {
    color: var(--syntax-dark);
    background-color: var(--syntax-dark-bg);
  }
  :root:not([saved-theme='light']) code[data-theme*=' '] span {
    color: var(--syntax-dark);
    font-style: var(--syntax-dark-font-style, normal);
    font-weight: var(--syntax-dark-font-weight, normal);
  }
}
[saved-theme='dark'] code[data-theme*=' '] {
  color: var(--syntax-dark);
  background-color: var(--syntax-dark-bg);
}
[saved-theme='dark'] code[data-theme*=' '] span {
  color: var(--syntax-dark);
  font-style: var(--syntax-dark-font-style, normal);
  font-weight: var(--syntax-dark-font-weight, normal);
}
`

const PAGE_SCRIPT = `
(() => {
  const controller = new AbortController();
  document.addEventListener('prenav', () => controller.abort(), { once: true, signal: controller.signal });
  // Reloading a viewer can retain same-document history entries from the SPA session.
  if (!window.spaNavigate) {
    const pathname = window.location.pathname;
    window.addEventListener('popstate', () => {
      if (window.location.pathname !== pathname) window.location.reload();
    }, { signal: controller.signal });
  }
  try {
    const saved = localStorage.getItem('theme');
    if (saved === 'dark' || saved === 'light') {
      document.documentElement.setAttribute('saved-theme', saved);
    }
  } catch {}
  const applyLineHash = () => {
    const match = window.location.hash.match(/^#L(\\d+)(?:-L?(\\d+))?$/i);
    const lines = document.querySelectorAll('.code-viewer [data-line]');
    lines.forEach(line => line.removeAttribute('data-highlighted-line'));
    if (!match) return;
    const start = Math.max(1, Number(match[1]));
    const end = match[2] ? Math.max(start, Number(match[2])) : start;
    let first = null;
    lines.forEach((line, idx) => {
      const lineNo = idx + 1;
      if (lineNo >= start && lineNo <= end) {
        line.setAttribute('data-highlighted-line', '');
        if (!first) first = line;
      }
    });
    if (first) first.scrollIntoView({ block: 'center' });
  };
  window.addEventListener('hashchange', applyLineHash, { signal: controller.signal });
  if (document.readyState === 'loading') {
    window.addEventListener('DOMContentLoaded', applyLineHash, { once: true, signal: controller.signal });
  } else {
    requestAnimationFrame(applyLineHash);
  }
  document.addEventListener('click', event => {
    const target = event.target instanceof Element ? event.target.closest('[data-copy-code]') : null;
    if (!(target instanceof HTMLButtonElement)) return;
    const viewer = target.closest('.code-viewer');
    if (!viewer) return;
    const text = Array.from(viewer.querySelectorAll('[data-line]'))
      .map(line => line.textContent ?? '')
      .join('\\n');
    navigator.clipboard?.writeText(text).then(() => {
      const prev = target.textContent;
      target.textContent = 'copied';
      setTimeout(() => {
        target.textContent = prev;
      }, 1500);
    });
  }, { signal: controller.signal });
})();
`

function formatByteSize(bytes: number): string {
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${bytes} B`
}

export function codeViewerHtml(pathname: string, source: string): string {
  const normalizedPath = pathname.replace(/^\/+/, '')
  const segments = normalizedPath.split('/').filter(Boolean)
  const fileName = segments.at(-1) ?? normalizedPath
  const dirSegments = segments.slice(0, -1)
  const dirPath = dirSegments.join('/')
  const dirHref = dirPath ? `/${dirPath}/` : '/'
  const title = dirPath ? `${fileName} · ${dirPath}` : fileName

  const highlighted = highlightSourceCode(source, fileName)
  const attrs = codeAttributes(highlighted)
  const byteSize = formatByteSize(new TextEncoder().encode(source).byteLength)
  const lineLabel = `${highlighted.lines} ${highlighted.lines === 1 ? 'line' : 'lines'}`
  const summary = `${highlighted.language} · ${lineLabel} · ${byteSize}`

  const breadcrumbHtml = dirPath
    ? `<a href="${escapeHTML(dirHref)}">${escapeHTML(dirPath)}/</a><span class="code-viewer-filename">${escapeHTML(fileName)}</span>`
    : `<span class="code-viewer-filename">${escapeHTML(fileName)}</span>`

  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${escapeHTML(title)}</title>
<style>${PAGE_STYLE}</style>
<script data-spa-init>${PAGE_SCRIPT}</script>
</head>
<body data-page="code-viewer" data-slug="${escapeHTML(normalizedPath)}">
<main class="code-viewer popover-hint">
<header class="code-viewer-header">
<div class="code-viewer-path">${breadcrumbHtml}</div>
<div class="code-viewer-actions">
<span class="code-viewer-meta">${escapeHTML(summary)}</span>
<a class="code-viewer-action" href="${escapeHTML(pathname)}?raw=1" data-router-ignore>raw</a>
<button type="button" class="code-viewer-action" data-copy-code>copy</button>
</div>
</header>
<figure data-rehype-pretty-code-figure>
<pre tabindex="0"><code data-language="${escapeHTML(attrs['data-language'])}" data-theme="${escapeHTML(attrs['data-theme'])}" style="${escapeHTML(attrs.style)}">${highlighted.html}</code></pre>
</figure>
</main>
</body>
</html>`
}

export async function codeViewerResponse(
  request: Request,
  url: URL,
  originResp: Response,
): Promise<Response> {
  if (originResp.ok && wantsCodeViewerHtml(request, url)) {
    const source = request.method === 'HEAD' ? '' : await originResp.text()
    return new Response(request.method === 'HEAD' ? null : codeViewerHtml(url.pathname, source), {
      status: originResp.status,
      headers: {
        'Content-Type': 'text/html; charset=utf-8',
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'Cache-Control': 'no-store, no-cache, must-revalidate',
        Vary: mergeVary(originResp, 'Accept', 'User-Agent'),
        Link: `<${url.pathname}?raw=1>; rel="alternate"; type="text/plain"`,
      },
    })
  }
  return withHeaders(originResp, {
    'Content-Type': 'text/plain; charset=utf-8',
    'X-Content-Type-Options': 'nosniff',
    'X-Frame-Options': 'DENY',
    'Cache-Control': 'no-store, no-cache, must-revalidate',
    Vary: mergeVary(originResp, 'Accept', 'User-Agent'),
  })
}
