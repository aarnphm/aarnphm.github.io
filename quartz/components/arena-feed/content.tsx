import type { RefObject } from 'preact'
import DOMPurify from 'dompurify'
import katex from 'katex'
import { useEffect, useMemo, useState } from 'preact/hooks'
import type { ArenaReaderArtifact, ArenaReaderRenderResult } from '../../util/arena-reader'
import { customMacros, katexOptions } from '../../cfg'
import { arenaFeedSourceNames, type ArenaFeedEntry } from '../../util/arena-feed'
import { parseWikipediaTarget } from '../../util/wikipedia'
import { CodeContent } from './code'
import { mathVariantText } from './math'

export function safeHref(raw: string): string | undefined {
  try {
    const url = new URL(raw, location.origin)
    return ['http:', 'https:'].includes(url.protocol) && !url.username && !url.password
      ? url.href
      : undefined
  } catch {
    return undefined
  }
}

export function sanitizeReaderHtml(html: string): string {
  const clean = DOMPurify.sanitize(html, {
    USE_PROFILES: { html: true, mathMl: true },
    FORBID_TAGS: [
      'style',
      'form',
      'input',
      'button',
      'textarea',
      'iframe',
      'object',
      'embed',
      'link',
      'meta',
    ],
    FORBID_ATTR: ['style', 'srcset', 'autofocus'],
    ALLOW_DATA_ATTR: false,
    ADD_ATTR: ['data-lang', 'data-latex', 'data-callout', 'data-arena-figure-width'],
  })
  const document = new DOMParser().parseFromString(clean, 'text/html')
  for (const figure of document.querySelectorAll<HTMLElement>('[data-arena-figure-width]')) {
    const width = Number(figure.getAttribute('data-arena-figure-width'))
    if (figure.tagName === 'FIGURE' && Number.isInteger(width) && width > 1 && width <= 1600) {
      figure.style.setProperty('--arena-figure-width', `${width}px`)
    } else {
      figure.removeAttribute('data-arena-figure-width')
    }
  }
  // MathML Core uses Unicode alphabets instead of MathJax's legacy mathvariant values.
  for (const token of document.querySelectorAll('math mi, math mn, math mo, math mtext, math ms')) {
    const variant = token.closest('[mathvariant]')?.getAttribute('mathvariant')
    if (!variant || token.children.length) continue
    const original = token.textContent ?? ''
    const text = mathVariantText(original, variant)
    if (text !== original) {
      token.textContent = text
      token.setAttribute('mathvariant', 'normal')
    }
  }
  for (const math of document.querySelectorAll('math[data-latex]')) {
    const latex = math.getAttribute('data-latex')?.trim()
    if (!latex || latex.length > 16_384) continue
    try {
      const rendered = katex.renderToString(latex, {
        ...katexOptions,
        displayMode: math.getAttribute('display') === 'block',
        output: 'htmlAndMathml',
        macros: { ...customMacros },
        trust: false,
      })
      const replacement = document.createElement('span')
      replacement.innerHTML = rendered
      math.replaceWith(...Array.from(replacement.childNodes))
    } catch {
      // Keep the captured MathML when KaTeX cannot parse publisher TeX or macros.
    }
  }
  for (const element of document.querySelectorAll('[src], [poster]')) {
    for (const attribute of ['src', 'poster']) {
      const raw = element.getAttribute(attribute)
      if (!raw) continue
      const safe = safeHref(raw)
      const url = safe ? new URL(safe) : null
      if (!url || url.origin !== location.origin || !url.pathname.startsWith('/api/arena/'))
        element.removeAttribute(attribute)
    }
  }
  for (const link of document.querySelectorAll('a')) {
    if (link.classList.contains('arena-footnote-backref')) {
      link.textContent = (link.textContent ?? '').replace(/↩[\uFE0E\uFE0F]?/gu, '↩\uFE0E')
    }
    const raw = link.getAttribute('href')
    if (!raw || raw.startsWith('#')) continue
    const href = safeHref(raw)
    if (!href) link.removeAttribute('href')
    else {
      link.href = href
      link.target = '_blank'
      link.rel = 'noopener noreferrer'
      const wikipedia = parseWikipediaTarget(href)
      if (wikipedia) {
        link.classList.add('internal')
        link.dataset.wikipediaLang = wikipedia.lang
        link.dataset.wikipediaTitle = wikipedia.title
      }
    }
  }
  for (const image of document.querySelectorAll('img')) {
    image.loading = 'lazy'
    image.decoding = 'async'
    image.referrerPolicy = 'no-referrer'
  }
  return document.body.innerHTML
}

function PdfContent({
  artifact,
  contentRef,
}: {
  artifact: Extract<ArenaReaderArtifact, { kind: 'pdf' }>
  contentRef: RefObject<HTMLDivElement>
}) {
  const resource = artifact.resources.find(item => item.id === artifact.resourceId)
  const src = resource && safeHref(resource.url)
  useEffect(() => {
    const root = contentRef.current
    if (!root || !src) return
    window.quartzPdfEmbeds?.mount(root)
    return () => {
      window.quartzPdfEmbeds?.cleanup(root)
    }
  }, [src, contentRef])
  return (
    <div ref={contentRef} class="arena-reader-pdf">
      {src ? (
        <div
          class="pdf-embed"
          data-pdf-src={src}
          data-pdf-title={artifact.title}
          data-pdf-fit="width"
        >
          <a href={src} target="_blank" rel="noopener noreferrer">
            Open PDF
          </a>
        </div>
      ) : (
        <p>
          The saved PDF is unavailable.{' '}
          <a href={safeHref(artifact.sourceUrl)} target="_blank" rel="noopener noreferrer">
            Open the original PDF
          </a>
          .
        </p>
      )}
    </div>
  )
}

function HtmlContent({
  artifact,
  contentRef,
}: {
  artifact: Extract<ArenaReaderArtifact, { kind: 'html' }>
  contentRef: RefObject<HTMLDivElement>
}) {
  const html = useMemo(
    () => (artifact.readerHtml ? sanitizeReaderHtml(artifact.readerHtml) : ''),
    [artifact.snapshotId, artifact.readerHtml],
  )
  if (!html)
    return (
      <p class="arena-reader-status">
        reader content is unavailable.{' '}
        <a href={safeHref(artifact.sourceUrl)} target="_blank" rel="noopener noreferrer">
          open original ↗
        </a>
      </p>
    )
  return (
    <>
      {artifact.quality === 'partial' && (
        <p class="arena-reader-quality" role="status" title={artifact.diagnostics.join(' ')}>
          incomplete copy
        </p>
      )}
      <div ref={contentRef} class="arena-reader-prose" dangerouslySetInnerHTML={{ __html: html }} />
    </>
  )
}

export function ArticleContent({
  entry,
  result,
  loading,
  contentRef,
  onRetry,
}: {
  entry: ArenaFeedEntry
  result: ArenaReaderRenderResult | null
  loading: boolean
  contentRef: RefObject<HTMLDivElement>
  onRetry: () => void
}) {
  const artifact = result?.status === 'ready' ? result.artifact : null
  const rawTitle = artifact?.title || entry.title
  const title = parseWikipediaTarget(entry.sourceUrl)
    ? rawTitle.replace(/\s+[-–—|]\s+Wikipedia\s*$/i, '')
    : rawTitle
  const [cooldown, setCooldown] = useState(0)
  useEffect(() => {
    const seconds = result?.status === 'unavailable' ? (result.retryAfter ?? 0) : 0
    const deadline = Date.now() + seconds * 1000
    setCooldown(Math.ceil(seconds))
    if (seconds <= 0) return
    const timer = window.setInterval(() => {
      const remaining = Math.max(0, Math.ceil((deadline - Date.now()) / 1000))
      setCooldown(remaining)
      if (remaining === 0) clearInterval(timer)
    }, 1000)
    return () => clearInterval(timer)
  }, [result])
  return (
    <>
      <header class="arena-reader-article-header">
        <p class="arena-reader-eyebrow">
          {entry.later ? 'later' : 'from your links'} · {arenaFeedSourceNames(entry).join(' / ')}
        </p>
        <h1>{title}</h1>
        {artifact && (
          <div class="arena-reader-source">
            {result?.status === 'ready' && result.cached ? 'saved copy' : 'captured'} ·{' '}
            {new Date(artifact.capturedAt)
              .toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })
              .toLocaleLowerCase()}
          </div>
        )}
      </header>
      {loading && !artifact && (
        <p role="status" class="arena-reader-status arena-reader-loading">
          {result?.status === 'pending'
            ? 'An article copy is being prepared…'
            : 'Opening your saved article…'}
        </p>
      )}
      {result?.status === 'ready' && result.warning && (
        <p class="arena-reader-notice">{result.warning}</p>
      )}
      {result?.status === 'unavailable' && (
        <div class="arena-reader-empty">
          <h2>Source unavailable</h2>
          <p>{result.message}</p>
          {cooldown > 0 ? <p>Retry in {cooldown} seconds.</p> : null}
          <button type="button" disabled={loading || cooldown > 0} onClick={onRetry}>
            Try again
          </button>
        </div>
      )}
      {artifact?.kind === 'html' && (
        <HtmlContent key={artifact.snapshotId} artifact={artifact} contentRef={contentRef} />
      )}
      {artifact?.kind === 'pdf' && <PdfContent artifact={artifact} contentRef={contentRef} />}
      {artifact?.kind === 'code' && <CodeContent artifact={artifact} contentRef={contentRef} />}
      {artifact?.kind === 'video' && (
        <div class="arena-reader-media">
          {artifact.embedUrl &&
          /^https:\/\/(www\.)?(youtube-nocookie\.com|youtube\.com|player\.vimeo\.com)\//.test(
            artifact.embedUrl,
          ) ? (
            <>
              <p class="arena-reader-status">
                Video provided by {new URL(artifact.embedUrl).hostname}.
              </p>
              <iframe
                title={artifact.title}
                src={artifact.embedUrl}
                referrerPolicy="no-referrer"
                sandbox="allow-scripts allow-same-origin allow-presentation"
                allow="fullscreen; picture-in-picture"
                allowFullScreen
              />
            </>
          ) : (
            <p>Open the original to watch this video.</p>
          )}
          {artifact.description && <p>{artifact.description}</p>}
        </div>
      )}
      {artifact?.kind === 'internal' && (
        <p>
          <a href={safeHref(artifact.internalUrl)} class="internal">
            Read this Garden note ↗
          </a>
        </p>
      )}
      {artifact?.kind === 'external' && <p class="arena-reader-notice">{artifact.message}</p>}
      {entry.occurrences.some(occurrence => occurrence.notesHtml) && (
        <details class="arena-reader-saved-context">
          <summary>Saved with this link</summary>
          {entry.occurrences
            .filter(occurrence => occurrence.notesHtml)
            .map(occurrence => (
              <section key={`${occurrence.channelSlug}:${occurrence.blockId}`}>
                <a
                  class="internal"
                  href={`/arena/${encodeURIComponent(occurrence.channelSlug)}#${encodeURIComponent(occurrence.blockId)}`}
                >
                  {occurrence.channelName}
                </a>
                <div
                  dangerouslySetInnerHTML={{
                    __html: sanitizeReaderHtml(occurrence.notesHtml ?? ''),
                  }}
                />
              </section>
            ))}
        </details>
      )}
    </>
  )
}
