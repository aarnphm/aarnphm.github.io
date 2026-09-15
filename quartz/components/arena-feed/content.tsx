import type { RefObject } from 'preact'
import DOMPurify from 'dompurify'
import { useEffect, useMemo, useRef, useState } from 'preact/hooks'
import type { ArenaFeedEntry } from '../../util/arena-feed'
import type { ArenaReaderArtifact, ArenaReaderRenderResult } from '../../util/arena-reader'

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
  })
  const document = new DOMParser().parseFromString(clean, 'text/html')
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
    const raw = link.getAttribute('href')
    if (!raw || raw.startsWith('#')) continue
    const href = safeHref(raw)
    if (!href) link.removeAttribute('href')
    else {
      link.href = href
      link.target = '_blank'
      link.rel = 'noopener noreferrer'
    }
  }
  for (const image of document.querySelectorAll('img')) {
    image.loading = 'lazy'
    image.decoding = 'async'
    image.referrerPolicy = 'no-referrer'
  }
  return document.body.innerHTML
}

function isolatedDocument(html: string): string {
  const body = sanitizeReaderHtml(html)
  return `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src ${location.origin}; style-src 'unsafe-inline'; font-src 'none'; base-uri 'none'; form-action 'none'"><style>:root{color-scheme:light dark}body{font:18px/1.65 Georgia,serif;max-width:70ch;margin:24px auto;padding:0 18px;overflow-wrap:anywhere}img,video{max-width:100%;height:auto}pre{overflow:auto;padding:1rem;background:light-dark(#f4f4f2,#222)}table{display:block;overflow:auto;border-collapse:collapse}td,th{border:1px solid #888;padding:.4rem}a{color:light-dark(#246344,#9cd8b4)}blockquote{margin-inline:1rem}h1,h2,h3{line-height:1.2}</style></head><body>${body}</body></html>`
}

function PdfContent({ artifact }: { artifact: Extract<ArenaReaderArtifact, { kind: 'pdf' }> }) {
  const ref = useRef<HTMLDivElement>(null)
  const resource = artifact.resources.find(item => item.id === artifact.resourceId)
  const src = resource && safeHref(resource.url)
  useEffect(() => {
    const root = ref.current
    if (!root || !src) return
    window.quartzPdfEmbeds?.mount(root)
    return () => {
      window.quartzPdfEmbeds?.cleanup(root)
    }
  }, [src])
  return (
    <div ref={ref} class="arena-reader-pdf">
      {src ? (
        <>
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
          <p>
            <a href={src} target="_blank" rel="noopener noreferrer">
              Open PDF in a separate tab
            </a>
          </p>
        </>
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
  const [full, setFull] = useState(!artifact.readerHtml)
  const html = useMemo(
    () => sanitizeReaderHtml(artifact.readerHtml ?? ''),
    [artifact.snapshotId, artifact.readerHtml],
  )
  const document = useMemo(
    () => isolatedDocument(artifact.documentHtml),
    [artifact.snapshotId, artifact.documentHtml],
  )
  return (
    <>
      <div class="arena-reader-representation" role="group" aria-label="Article representation">
        <button
          type="button"
          aria-pressed={!full}
          disabled={!artifact.readerHtml}
          onClick={() => setFull(false)}
        >
          Reader
        </button>
        <button type="button" aria-pressed={full} onClick={() => setFull(true)}>
          Full document
        </button>
      </div>
      {artifact.quality === 'partial' && (
        <p class="arena-reader-notice">
          This copy may be incomplete. {artifact.diagnostics.join(' ')}
        </p>
      )}
      {full ? (
        <iframe
          class="arena-reader-document"
          title={`${artifact.title}, full document`}
          sandbox="allow-popups allow-popups-to-escape-sandbox"
          referrerPolicy="no-referrer"
          srcDoc={document}
        />
      ) : (
        <div
          ref={contentRef}
          class="arena-reader-prose"
          dangerouslySetInnerHTML={{ __html: html }}
        />
      )}
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
          {entry.later ? 'Later' : 'From your channels'} ·{' '}
          {entry.occurrences
            .map(item => item.channelName)
            .filter((name, index, names) => names.indexOf(name) === index)
            .join(' / ')}
        </p>
        <h1>{artifact?.title || entry.title}</h1>
        <div class="arena-reader-source">
          <a href={safeHref(entry.sourceUrl)} target="_blank" rel="noopener noreferrer">
            Open original ↗
          </a>
          {artifact && (
            <span>
              {result?.status === 'ready' && result.cached ? 'Saved copy' : 'Captured'} ·{' '}
              {new Date(artifact.capturedAt).toLocaleDateString(undefined, {
                month: 'short',
                day: 'numeric',
                year: 'numeric',
              })}
            </span>
          )}
        </div>
      </header>
      {loading && (
        <p role="status" class="arena-reader-status">
          {artifact
            ? 'Refreshing from the source…'
            : result?.status === 'pending'
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
      {artifact?.kind === 'pdf' && <PdfContent artifact={artifact} />}
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
