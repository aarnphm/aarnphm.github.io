// Intelligence mode for the search palette: the semantic results stay, and Gemma 4 E4B reads the
// top passages and writes a short overview above them, cited by attention mass (the share of the
// 7 global layers' attention each passage received while the overview was written) and annotated
// with the transcoder features that fired on each token. Gemma never runs in the browser: on
// localhost the stream comes from plt_serve.py (POST /read, quartz/scripts/latent/read.py); on the
// public site a baked overview is fetched for the normalised query, or the mode says it is local.
// Spec: /mnt/project-files/edge-aaron-llm/palette-intelligence.md and mac-read-spec.md.

export const INTELLIGENCE_PREFIX = '?'
const PORT = 8093
const MAX_PASSAGES = 10

/** A 3×3 grid of dots with the centre one lit, on the same 20-unit grid as the mode icons.
 *  Loading: an `.is-loading` ancestor runs a wave of opacity across the grid, top-left to
 *  bottom-right, by nth-child delays (scss in palette.scss; glyph/intelligence-mode.svg carries
 *  the same rules inline for the standalone file). */
export function intelligenceGlyph(): string {
  return `<svg viewBox="0 0 20 20" role="img" aria-hidden="true"><g class="intel-grid" fill="currentColor"><circle cx="4.5" cy="4.5" r="1.3"/><circle cx="10" cy="4.5" r="1.3"/><circle cx="15.5" cy="4.5" r="1.3"/><circle cx="4.5" cy="10" r="1.3"/><circle class="intel-dot" cx="10" cy="10" r="1.3" fill="var(--tertiary)"/><circle cx="15.5" cy="10" r="1.3"/><circle cx="4.5" cy="15.5" r="1.3"/><circle cx="10" cy="15.5" r="1.3"/><circle cx="15.5" cy="15.5" r="1.3"/></g></svg>`
}

export const INTELLIGENCE_GLYPH = intelligenceGlyph()

export interface ReadPassage {
  row: number
  slug: string
  text: string
}

export interface ReadRequest {
  query: string
  passages: ReadPassage[]
  think?: boolean
  max_tokens?: number
  readout?: number[]
  steer?: { layer: number; id: number; s: number }[]
  mask?: { row: number; bias: number }[]
  lora_scale?: number
}

/** One generated token: text, index, attention-mass cites [row, mass], features [layer, id, act]. */
export interface ReadToken {
  t: string
  i: number
  cites?: [number, number][]
  feat?: [number, number, number][]
}

export interface ReadDone {
  done: true
  hops?: string[]
  cites?: [number, number][]
  prefill_ms?: number
  decode_tps?: number
}

export type ReadEvent = ReadToken | ReadDone

export function isReadDone(event: ReadEvent): event is ReadDone {
  return 'done' in event && event.done === true
}

/** `?` leads an intelligence query; the rest is the search term the semantic ranker sees. */
export function parseIntelligenceQuery(raw: string): string | null {
  const value = raw.trimStart()
  if (!value.startsWith(INTELLIGENCE_PREFIX)) return null
  return value.slice(INTELLIGENCE_PREFIX.length).trim().replace(/\s+/g, ' ')
}

/** The key a baked overview is filed under: lowercase, single spaces, no trailing punctuation. */
export function normaliseQuery(query: string): string {
  return query
    .toLowerCase()
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/[?!.。]+$/u, '')
    .trim()
}

export async function queryHash(query: string): Promise<string> {
  const bytes = new TextEncoder().encode(normaliseQuery(query))
  const digest = await crypto.subtle.digest('SHA-256', bytes)
  return Array.from(new Uint8Array(digest).slice(0, 8), b => b.toString(16).padStart(2, '0')).join(
    '',
  )
}

/** Parse a newline-delimited JSON stream into read events. Partial trailing lines wait for more. */
export async function* parseReadStream(
  body: ReadableStream<Uint8Array>,
  signal?: AbortSignal,
): AsyncGenerator<ReadEvent> {
  const reader = body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  const abort = () => void reader.cancel().catch(() => {})
  signal?.addEventListener('abort', abort, { once: true })
  try {
    for (;;) {
      const { value, done } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      let at: number
      while ((at = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, at).trim()
        buffer = buffer.slice(at + 1)
        if (line) yield JSON.parse(line) as ReadEvent
      }
    }
    const tail = buffer.trim()
    if (tail) yield JSON.parse(tail) as ReadEvent
  } finally {
    signal?.removeEventListener('abort', abort)
  }
}

/** Sum per-token cites into one normalised list, strongest first. */
export function aggregateCites(tokens: readonly ReadToken[]): [number, number][] {
  const mass = new Map<number, number>()
  for (const token of tokens) {
    for (const [row, m] of token.cites ?? []) mass.set(row, (mass.get(row) ?? 0) + m)
  }
  const total = [...mass.values()].reduce((a, b) => a + b, 0)
  if (!total) return []
  return [...mass.entries()]
    .map(([row, m]) => [row, m / total] as [number, number])
    .sort((a, b) => b[1] - a[1])
}

/** `search: …` lines at the end of an overview are hops the client runs as new queries. */
export function splitHops(text: string): { body: string; hops: string[] } {
  const lines = text.trimEnd().split('\n')
  const hops: string[] = []
  while (lines.length) {
    const m = /^\s*search:\s*(.+?)\s*$/i.exec(lines[lines.length - 1])
    if (!m) break
    hops.unshift(m[1])
    lines.pop()
  }
  return { body: lines.join('\n').trimEnd(), hops }
}

export type ReadSource =
  | { kind: 'live'; url: string }
  | { kind: 'baked'; url: string }
  | { kind: 'none' }

const isLocal = () => ['localhost', '127.0.0.1'].includes(window.location.hostname)

const WORKER_READ = '/api/gemma/read'

/** The site's Worker route (worker/gemma.ts → Modal) answers HEAD with 204 once its secrets exist
 *  and 503 otherwise. Probing first keeps the caller from loading passages nobody will read. */
async function workerReads(signal?: AbortSignal): Promise<boolean> {
  try {
    const response = await fetch(WORKER_READ, { method: 'HEAD', cache: 'no-store', signal })
    return response.ok
  } catch {
    return false
  }
}

/** Where an overview can come from on this page, in order of preference: the Mac reader on
 *  localhost, then the Worker route when it answers, then a baked overview. */
export async function readSources(
  query: string,
  sidecar8: string | null,
  signal?: AbortSignal,
): Promise<ReadSource[]> {
  const sources: ReadSource[] = []
  if (isLocal()) sources.push({ kind: 'live', url: `http://localhost:${PORT}/read` })
  if (await workerReads(signal)) sources.push({ kind: 'live', url: WORKER_READ })
  if (sidecar8) {
    const hash = await queryHash(query)
    sources.push({ kind: 'baked', url: `/models/latent-plt/bake/${sidecar8}/reads/${hash}.json` })
  }
  sources.push({ kind: 'none' })
  return sources
}

/** Stream an overview from the first source that answers. A baked file is a whole event list. */
export async function* readOverview(
  request: ReadRequest,
  sources: readonly ReadSource[],
  signal?: AbortSignal,
): AsyncGenerator<ReadEvent> {
  for (const source of sources) {
    if (source.kind === 'none') return
    try {
      if (source.kind === 'live') {
        const response = await fetch(source.url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ...request, passages: request.passages.slice(0, MAX_PASSAGES) }),
          signal,
        })
        if (!response.ok || !response.body) continue
        yield* parseReadStream(response.body, signal)
        return
      }
      const response = await fetch(source.url, { signal })
      if (!response.ok) continue
      const events = (await response.json()) as ReadEvent[]
      for (const event of events) yield event
      return
    } catch (error) {
      if (signal?.aborted) return
      console.warn('[intelligence] source failed:', source.url, error)
    }
  }
}

/** The block above the results. One DOM write per animation frame: the site's :has() rules make
 *  small writes restyle the whole page, so text is batched and nodes are reused. */
export interface OverviewBlock {
  element: HTMLElement
  push(event: ReadEvent): void
  fail(message: string): void
  dispose(): void
}

export function mountOverviewBlock(
  output: HTMLElement,
  passages: readonly ReadPassage[],
  hrefOf: (slug: string) => string,
  onHop: (query: string) => void,
): OverviewBlock {
  const slugOf = new Map(passages.map(p => [p.row, p.slug]))
  const section = document.createElement('section')
  section.className = 'intelligence'
  section.setAttribute('aria-live', 'polite')
  section.setAttribute('aria-busy', 'true')
  section.classList.add('is-loading')
  // A div, not <header>: the site's header rules push a <header>'s children to the right.
  section.innerHTML = `<div class="intelligence-head"><span class="intelligence-glyph">${INTELLIGENCE_GLYPH}</span><span class="intelligence-status">reading…</span></div><p class="intelligence-text"></p><ol class="intelligence-cites" aria-label="sources"></ol><p class="intelligence-hops"></p>`
  const status = section.querySelector<HTMLElement>('.intelligence-status')!
  const text = section.querySelector<HTMLElement>('.intelligence-text')!
  const cites = section.querySelector<HTMLElement>('.intelligence-cites')!
  const hopsEl = section.querySelector<HTMLElement>('.intelligence-hops')!
  output.prepend(section)

  const tokens: ReadToken[] = []
  let raw = ''
  let frame = 0
  let finished = false
  let finalCites: [number, number][] | null = null
  let hops: string[] = []

  const flush = () => {
    frame = 0
    const { body } = splitHops(raw)
    if (text.textContent !== body) text.textContent = body
    const list = finalCites ?? aggregateCites(tokens)
    const top = list.slice(0, 5)
    const want = top.map(([row, mass]) => `${row}:${mass.toFixed(2)}`).join(' ')
    if (cites.dataset.want !== want) {
      cites.dataset.want = want
      cites.replaceChildren(
        ...top.flatMap(([row, mass]) => {
          const slug = slugOf.get(row)
          if (!slug) return []
          const li = document.createElement('li')
          li.style.setProperty('--mass', mass.toFixed(3))
          const a = document.createElement('a')
          a.href = hrefOf(slug)
          a.textContent = slug.split('/').at(-1) ?? slug
          a.title = `${Math.round(mass * 100)}% of attention`
          li.append(a)
          return [li]
        }),
      )
    }
    if (finished) {
      section.removeAttribute('aria-busy')
      section.classList.remove('is-loading')
      status.textContent = hops.length ? 'read · hops below' : 'read'
      hopsEl.replaceChildren(
        ...hops.map(hop => {
          const button = document.createElement('button')
          button.type = 'button'
          button.textContent = `search: ${hop}`
          button.addEventListener('click', () => onHop(hop))
          return button
        }),
      )
    }
  }
  const schedule = () => {
    if (!frame) frame = requestAnimationFrame(flush)
  }

  return {
    element: section,
    push(event) {
      if (isReadDone(event)) {
        finished = true
        finalCites = event.cites ?? null
        hops = event.hops ?? splitHops(raw).hops
      } else {
        tokens.push(event)
        raw += event.t
      }
      schedule()
    },
    fail(message) {
      finished = true
      section.removeAttribute('aria-busy')
      section.classList.remove('is-loading')
      section.classList.add('is-unavailable')
      status.textContent = message
      if (frame) cancelAnimationFrame(frame)
      frame = 0
    },
    dispose() {
      if (frame) cancelAnimationFrame(frame)
      section.remove()
    },
  }
}
