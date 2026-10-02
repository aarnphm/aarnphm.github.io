import type { ComponentChildren } from 'preact'
import { autoUpdate, computePosition, flip, offset, shift } from '@floating-ui/dom'
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'preact/hooks'
import type { PdfCitation, PdfMark, PdfMarkKind } from '../../util/pdf-marks'
import { simplifySlug, type FullSlug } from '../../util/path'
import { renderBody } from './body'
import { saveStatusLabel, type MarkEntry } from './book'

export const kindMeta: Record<PdfMarkKind, { glyph: string; label: string; key: string }> = {
  mark: { glyph: '●', label: 'mark', key: 'a' },
  question: { glyph: '?', label: 'question', key: 'q' },
  contra: { glyph: '≠', label: 'disagree', key: 'x' },
}

export const kinds = Object.keys(kindMeta) as PdfMarkKind[]

export function notePath(slug: string): string {
  const simple = simplifySlug(slug as FullSlug)
  return `/${encodeURI(simple === '/' ? '' : simple)}`
}

function KindGlyph({ kind }: { kind: PdfMarkKind }) {
  return (
    <span class="pdf-kind-glyph" data-kind={kind} aria-hidden="true">
      {kindMeta[kind].glyph}
    </span>
  )
}

function CopyButton({ label, onCopy }: { label: 'link' | 'quote'; onCopy(): Promise<boolean> }) {
  const [copied, setCopied] = useState(false)
  const attempt = useRef(0)
  const timeout = useRef<number>()
  const name = label === 'link' ? 'Link' : 'Quote'

  useEffect(
    () => () => {
      attempt.current++
      window.clearTimeout(timeout.current)
    },
    [],
  )

  const copy = async () => {
    const current = ++attempt.current
    window.clearTimeout(timeout.current)
    setCopied(false)
    const success = await onCopy()
    if (!success || current !== attempt.current) return
    setCopied(true)
    timeout.current = window.setTimeout(() => setCopied(false), 1800)
  }

  return (
    <button
      type="button"
      class="pdf-mark-copy-button"
      data-copied={copied}
      aria-label={copied ? `${name} copied` : `Copy ${label}`}
      title={copied ? `${name} copied` : `Copy ${label}`}
      onClick={() => void copy()}
    >
      <span aria-hidden="true">{label}</span>
      <svg
        width="14"
        height="14"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="1.75"
        stroke-linecap="round"
        stroke-linejoin="round"
        aria-hidden="true"
        focusable="false"
      >
        <path d="m5 12 4 4L19 6" />
      </svg>
    </button>
  )
}

export interface MarkCardProps {
  entry: MarkEntry
  active: boolean
  canWrite: boolean
  pageLabel: string
  autoFocus: boolean
  onActivate(): void
  onBody(body: string): void
  onKind(kind: PdfMarkKind): void
  onVisibility(visibility: PdfMark['visibility']): void
  onDelete(): void
  onResolve(choice: 'mine' | 'theirs'): void
  onCopyLink(): Promise<boolean>
  onCopyQuote(): Promise<boolean>
}

export function MarkCard(props: MarkCardProps) {
  const { entry, active, canWrite } = props
  const { mark, status } = entry
  const textarea = useRef<HTMLTextAreaElement>(null)
  const quote = mark.target.type === 'text' ? mark.target.quote.exact : null
  const what =
    mark.target.type === 'region' ? 'region' : mark.target.type === 'page' ? 'page note' : null

  useEffect(() => {
    if (active && props.autoFocus) textarea.current?.focus({ preventScroll: true })
  }, [active, props.autoFocus])

  return (
    <article
      class="pdf-mark-card"
      data-kind={mark.kind}
      data-active={active}
      data-status={status}
      aria-label={`${kindMeta[mark.kind].label} on page ${props.pageLabel}`}
      onClick={event => {
        if (!(event.target as Element).closest('button, a, textarea, input, label'))
          props.onActivate()
      }}
    >
      <header class="pdf-mark-card-meta">
        <button
          type="button"
          class="pdf-mark-card-jump"
          aria-pressed={active}
          onClick={props.onActivate}
        >
          <KindGlyph kind={mark.kind} />
          <span>{kindMeta[mark.kind].label}</span>
          <span class="pdf-mark-card-page">p. {props.pageLabel}</span>
          {what && <span class="pdf-mark-card-what">{what}</span>}
        </button>
        {mark.visibility === 'private' && <span class="pdf-mark-card-private">private</span>}
        {canWrite && status !== 'saved' && (
          <span class="pdf-mark-card-status" role="status">
            {saveStatusLabel[status]}
          </span>
        )}
      </header>
      {quote && <blockquote class="pdf-mark-card-quote">{quote}</blockquote>}
      {active && canWrite && status !== 'conflict' ? (
        <textarea
          ref={textarea}
          class="pdf-mark-card-input"
          value={mark.body}
          aria-label="Note"
          placeholder="Write a note. Markdown and $math$ work."
          maxLength={16384}
          onInput={event => props.onBody(event.currentTarget.value)}
          onKeyDown={event => {
            if (event.key === 'Escape') {
              event.preventDefault()
              event.stopPropagation()
              event.currentTarget.blur()
            }
          }}
        />
      ) : (
        mark.body.trim() && (
          <div
            class="pdf-mark-card-body"
            dangerouslySetInnerHTML={{ __html: renderBody(mark.body) }}
          />
        )
      )}
      {entry.error && <p class="pdf-mark-card-error">{entry.error}</p>}
      {status === 'conflict' && (
        <section class="pdf-mark-card-conflict" aria-label="Conflicting versions">
          <p>
            {entry.conflict
              ? 'Another device saved a newer version.'
              : 'Another device deleted this mark.'}
          </p>
          {entry.conflict?.body && (
            <details>
              <summary>their note</summary>
              <p>{entry.conflict.body}</p>
            </details>
          )}
          <div class="pdf-reader-row">
            <button type="button" onClick={() => props.onResolve('mine')}>
              keep mine
            </button>
            <button type="button" onClick={() => props.onResolve('theirs')}>
              {entry.conflict ? 'take theirs' : 'let it go'}
            </button>
          </div>
        </section>
      )}
      {active && (
        <footer class="pdf-mark-card-actions">
          {canWrite && (
            <div class="pdf-mark-card-kinds" role="group" aria-label="Kind">
              {kinds.map(kind => (
                <button
                  type="button"
                  aria-pressed={mark.kind === kind}
                  aria-label={kindMeta[kind].label}
                  title={`${kindMeta[kind].label} (${kindMeta[kind].key})`}
                  onClick={() => props.onKind(kind)}
                >
                  <KindGlyph kind={kind} />
                </button>
              ))}
            </div>
          )}
          {canWrite && (
            <button
              type="button"
              aria-pressed={mark.visibility === 'private'}
              title="Private marks are only visible to you"
              onClick={() =>
                props.onVisibility(mark.visibility === 'private' ? 'public' : 'private')
              }
            >
              private
            </button>
          )}
          <CopyButton label="link" onCopy={props.onCopyLink} />
          <CopyButton label="quote" onCopy={props.onCopyQuote} />
          {canWrite && (
            <button
              type="button"
              class="pdf-reader-danger pdf-mark-card-delete"
              aria-label="Delete mark"
              title="Delete mark"
              onClick={props.onDelete}
            >
              <svg
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="1.5"
                stroke-linecap="round"
                stroke-linejoin="round"
                aria-hidden="true"
                focusable="false"
              >
                <path d="M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7m4-7v7" />
              </svg>
            </button>
          )}
        </footer>
      )}
    </article>
  )
}

export function CitationCard({
  citation,
  pageLabel,
  onJump,
}: {
  citation: PdfCitation
  pageLabel?: string
  onJump?: () => void
}) {
  return (
    <article class="pdf-cite-card">
      <header class="pdf-mark-card-meta">
        <a href={notePath(citation.from)} class="internal" data-slug={citation.from}>
          ← {citation.title}
        </a>
        {onJump && pageLabel && (
          <button type="button" class="pdf-mark-card-page" onClick={onJump}>
            p. {pageLabel}
          </button>
        )}
      </header>
      {citation.excerpt && <p class="pdf-cite-card-excerpt">{citation.excerpt}</p>}
    </article>
  )
}

const MARGIN_GAP = 8

/**
 * Stacks margin cards at their anchors' heights. The active card keeps its anchor; neighbours move
 * out of its way, so the card being read never jumps.
 */
export function layoutMargin(container: HTMLElement, activeKey: string | null) {
  const items = [...container.querySelectorAll<HTMLElement>(':scope > [data-margin-top]')]
    .map(element => ({
      element,
      key: element.dataset.marginKey ?? '',
      desired: Number(element.dataset.marginTop) || 0,
      height: element.offsetHeight,
    }))
    .sort((a, b) => a.desired - b.desired)
  const tops = items.map(item => item.desired)
  const pivot = activeKey ? items.findIndex(item => item.key === activeKey) : -1
  if (pivot >= 0) {
    for (let index = pivot + 1; index < items.length; index++) {
      tops[index] = Math.max(tops[index], tops[index - 1] + items[index - 1].height + MARGIN_GAP)
    }
    for (let index = pivot - 1; index >= 0; index--) {
      tops[index] = Math.min(tops[index], tops[index + 1] - items[index].height - MARGIN_GAP)
    }
  }
  for (let index = 0; index < items.length; index++) {
    const floor = index === 0 ? 0 : tops[index - 1] + items[index - 1].height + MARGIN_GAP
    tops[index] = Math.max(tops[index], floor)
  }
  items.forEach((item, index) => {
    item.element.style.transform = `translateY(${Math.round(tops[index])}px)`
  })
  const last = items.length - 1
  container.style.minHeight = last >= 0 ? `${tops[last] + items[last].height}px` : ''
}

export interface MarginItem {
  key: string
  top: number
  node: ComponentChildren
}

export function Margin({
  items,
  activeKey,
  layoutKey,
  hidden,
}: {
  items: MarginItem[]
  activeKey: string | null
  layoutKey: string
  hidden: boolean
}) {
  const container = useRef<HTMLDivElement>(null)

  useLayoutEffect(() => {
    const element = container.current
    if (!element || hidden) return
    layoutMargin(element, activeKey)
  })

  useEffect(() => {
    const element = container.current
    if (!element) return
    let frame = 0
    const observer = new ResizeObserver(() => {
      window.cancelAnimationFrame(frame)
      frame = window.requestAnimationFrame(() =>
        layoutMargin(element, element.dataset.active || null),
      )
    })
    for (const child of element.children) observer.observe(child)
    return () => {
      observer.disconnect()
      window.cancelAnimationFrame(frame)
    }
  }, [layoutKey, hidden])

  return (
    <div
      ref={container}
      class="pdf-reader-margin"
      data-active={activeKey ?? ''}
      hidden={hidden}
      aria-label="Margin notes"
      role="region"
    >
      {items.map(item => (
        <div
          class="pdf-reader-margin-item"
          key={item.key}
          data-margin-key={item.key}
          data-margin-top={item.top}
        >
          {item.node}
        </div>
      ))}
    </div>
  )
}

export interface OutlineEntry {
  title: string
  dest: unknown
  url: string | null
  depth: number
}

export function IndexPanel(props: {
  entries: MarkEntry[]
  citations: PdfCitation[]
  outline: OutlineEntry[]
  orphans: PdfMark[]
  canWrite: boolean
  label(page: number): string
  activeId: string | null
  renderCard(entry: MarkEntry): ComponentChildren
  onActivate(id: string): void
  onOutline(entry: OutlineEntry): void
  onJumpPage(page: number): void
  onDropOrphan(mark: PdfMark): void
}) {
  const [filter, setFilter] = useState('')
  const [kind, setKind] = useState<PdfMarkKind | 'all'>('all')
  const query = filter.trim().toLowerCase()
  const visible = useMemo(
    () =>
      props.entries.filter(({ mark }) => {
        if (kind !== 'all' && mark.kind !== kind) return false
        if (!query) return true
        const quote = mark.target.type === 'text' ? mark.target.quote.exact : ''
        return `${quote}\n${mark.body}`.toLowerCase().includes(query)
      }),
    [props.entries, kind, query],
  )
  const activeRow = useRef<HTMLLIElement>(null)
  useEffect(() => {
    activeRow.current?.scrollIntoView({ block: 'nearest' })
  }, [props.activeId])

  return (
    <div class="pdf-reader-index">
      <section aria-labelledby="pdf-index-marks">
        <h2 id="pdf-index-marks">marks</h2>
        <div class="pdf-reader-row">
          <input
            type="search"
            value={filter}
            placeholder="filter quotes and notes"
            aria-label="Filter marks"
            onInput={event => setFilter(event.currentTarget.value)}
          />
        </div>
        <div class="pdf-reader-segmented" role="group" aria-label="Show kinds">
          <button type="button" aria-pressed={kind === 'all'} onClick={() => setKind('all')}>
            all
          </button>
          {kinds.map(value => (
            <button type="button" aria-pressed={kind === value} onClick={() => setKind(value)}>
              <KindGlyph kind={value} /> {kindMeta[value].label}
            </button>
          ))}
        </div>
        {visible.length === 0 ? (
          <p class="pdf-reader-empty">
            {props.entries.length === 0
              ? props.canWrite
                ? 'Select text and press a, q or x to mark it.'
                : 'No public marks on this document yet.'
              : 'No marks match.'}
          </p>
        ) : (
          <ol class="pdf-reader-mark-list">
            {visible.map(entry => {
              const active = entry.mark.id === props.activeId
              const quote =
                entry.mark.target.type === 'text' ? entry.mark.target.quote.exact : entry.mark.body
              return (
                <li key={entry.mark.id} ref={active ? activeRow : undefined}>
                  {active ? (
                    props.renderCard(entry)
                  ) : (
                    <button
                      type="button"
                      class="pdf-reader-mark-row"
                      onClick={() => props.onActivate(entry.mark.id)}
                    >
                      <KindGlyph kind={entry.mark.kind} />
                      <span class="pdf-mark-card-page">
                        p. {props.label(entry.mark.target.page)}
                      </span>
                      <span class="pdf-reader-mark-row-text">{quote || 'page note'}</span>
                    </button>
                  )}
                </li>
              )
            })}
          </ol>
        )}
      </section>
      {props.outline.length > 0 && (
        <section aria-labelledby="pdf-index-contents">
          <h2 id="pdf-index-contents">contents</h2>
          <ol class="pdf-reader-outline">
            {props.outline.map(entry => (
              <li style={{ '--depth': entry.depth }}>
                <button type="button" onClick={() => props.onOutline(entry)}>
                  {entry.title}
                </button>
              </li>
            ))}
          </ol>
        </section>
      )}
      {props.citations.length > 0 && (
        <section aria-labelledby="pdf-index-cited">
          <h2 id="pdf-index-cited">cited by</h2>
          <ol class="pdf-reader-mark-list">
            {props.citations.map(citation => (
              <li>
                <CitationCard
                  citation={citation}
                  pageLabel={citation.page ? props.label(citation.page) : undefined}
                  onJump={citation.page ? () => props.onJumpPage(citation.page!) : undefined}
                />
              </li>
            ))}
          </ol>
        </section>
      )}
      {props.canWrite && props.orphans.length > 0 && (
        <section aria-labelledby="pdf-index-orphans">
          <h2 id="pdf-index-orphans">lost in the new copy</h2>
          <p class="pdf-reader-empty">
            The PDF changed and these quotes no longer appear near their old pages.
          </p>
          <ol class="pdf-reader-mark-list">
            {props.orphans.map(mark => (
              <li>
                <article class="pdf-mark-card" data-kind={mark.kind}>
                  <header class="pdf-mark-card-meta">
                    <KindGlyph kind={mark.kind} />
                    <span class="pdf-mark-card-page">was p. {mark.target.page}</span>
                  </header>
                  {mark.target.type === 'text' && (
                    <blockquote class="pdf-mark-card-quote">{mark.target.quote.exact}</blockquote>
                  )}
                  {mark.body && <p>{mark.body}</p>}
                  <div class="pdf-reader-row">
                    <button
                      type="button"
                      class="pdf-reader-danger"
                      onClick={() => props.onDropOrphan(mark)}
                    >
                      drop
                    </button>
                  </div>
                </article>
              </li>
            ))}
          </ol>
        </section>
      )}
    </div>
  )
}

export interface StripTick {
  key: string
  top: number
  kind: PdfMarkKind | 'cite'
}

/** Overview ruler: one tick per mark or citation, plus a bracket for the visible span. */
export function Strip({
  ticks,
  bracket,
  onScrub,
}: {
  ticks: StripTick[]
  bracket: { current: HTMLDivElement | null }
  onScrub(fraction: number): void
}) {
  const track = useRef<HTMLDivElement>(null)
  const scrub = (event: PointerEvent) => {
    const rect = track.current?.getBoundingClientRect()
    if (!rect || rect.height <= 0) return
    onScrub(Math.min(1, Math.max(0, (event.clientY - rect.top) / rect.height)))
  }
  return (
    <div
      ref={track}
      class="pdf-reader-strip"
      aria-hidden="true"
      onPointerDown={event => {
        event.currentTarget.setPointerCapture(event.pointerId)
        scrub(event)
      }}
      onPointerMove={event => {
        if (event.currentTarget.hasPointerCapture(event.pointerId)) scrub(event)
      }}
    >
      {ticks.map(tick => (
        <span
          key={tick.key}
          class="pdf-reader-strip-tick"
          data-kind={tick.kind}
          style={{ top: `${tick.top * 100}%` }}
        />
      ))}
      <div ref={bracket} class="pdf-reader-strip-bracket" />
    </div>
  )
}

export function SelectionBubble({
  range,
  canWrite,
  onAction,
}: {
  range: Range
  canWrite: boolean
  onAction(action: PdfMarkKind | 'note' | 'quote'): void
}) {
  const bubble = useRef<HTMLDivElement>(null)

  useLayoutEffect(() => {
    const floating = bubble.current
    if (!floating) return
    const reference = {
      getBoundingClientRect: () => {
        const rects = range.getClientRects()
        return rects.length > 0 ? rects[rects.length - 1] : range.getBoundingClientRect()
      },
      contextElement:
        range.endContainer instanceof Element
          ? range.endContainer
          : (range.endContainer.parentElement ?? undefined),
    }
    return autoUpdate(reference, floating, async () => {
      const { x, y } = await computePosition(reference, floating, {
        placement: 'bottom-end',
        strategy: 'fixed',
        middleware: [offset(6), flip({ padding: 12 }), shift({ padding: 12 })],
      })
      Object.assign(floating.style, { left: `${x}px`, top: `${y}px`, visibility: 'visible' })
    })
  }, [range])

  return (
    <div
      ref={bubble}
      class="pdf-reader-bubble"
      role="toolbar"
      aria-label="Mark the selection"
      onPointerDown={event => event.preventDefault()}
    >
      {canWrite &&
        kinds.map(kind => (
          <button
            type="button"
            title={`${kindMeta[kind].label} (${kindMeta[kind].key})`}
            onClick={() => onAction(kind)}
          >
            <KindGlyph kind={kind} /> {kindMeta[kind].label}
          </button>
        ))}
      {canWrite && (
        <button type="button" title="mark with a note (n)" onClick={() => onAction('note')}>
          note
        </button>
      )}
      <button type="button" title="copy as a garden quote (c)" onClick={() => onAction('quote')}>
        quote
      </button>
    </div>
  )
}
