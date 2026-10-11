import type { RoughAnnotation } from 'rough-notation/lib/model'
import { annotate } from 'rough-notation'
import { groupCues, parseVtt, type VttCue } from '../../util/vtt'
import { currentNavSignal } from './nav-lifecycle'

/** Clipboard write with the legacy fallback for panes that deny the async clipboard. */
async function copyText(text: string): Promise<void> {
  try {
    await navigator.clipboard.writeText(text)
    return
  } catch {
    // fall through
  }
  const area = document.createElement('textarea')
  area.value = text
  area.setAttribute('readonly', '')
  area.style.position = 'fixed'
  area.style.opacity = '0'
  document.body.appendChild(area)
  area.select()
  const ok = document.execCommand('copy')
  area.remove()
  if (!ok) throw new Error('copy rejected')
}

// Watch page: the youtube-nocookie player mounts on load (no autoplay), timestamps seek through
// the widget postMessage protocol (no iframe_api script), the transcript follows the player with a
// rough-notation box on the current paragraph and an underline that sweeps the current cue at the
// speed of its timing (cues come from the served captions file), and the search paints hits with
// the CSS Custom Highlight API.
// docs/plans/course-resources.md §3.

const YOUTUBE_ORIGINS = new Set(['https://www.youtube-nocookie.com', 'https://www.youtube.com'])
const READY_TIMEOUT = 5000
const USER_SCROLL_HOLD = 4000
const MAX_HIGHLIGHTS = 1000
// A cue shorter than this sweeps at the floor.
const CUE_SWEEP_MIN = 300

interface Cue {
  start: number
  end: number
  span: HTMLElement
  paragraph: number
}

/** Typographic marks the markdown pipeline applies, undone so cue text matches the rendered text. */
const plainChar: Record<string, string> = {
  '\u2018': "'",
  '\u2019': "'",
  '\u201c': '"',
  '\u201d': '"',
  '\u2026': '...',
  '\u2014': '--',
  '\u2013': '--',
}

/** `text` with typographic marks undone; `at[i]` is the source index of normalized char i. */
function plain(text: string): { text: string; at: number[] } {
  let out = ''
  const at: number[] = []
  for (let i = 0; i < text.length; i++) {
    const piece = plainChar[text[i]] ?? text[i]
    out += piece
    for (let k = 0; k < piece.length; k++) at.push(i)
  }
  return { text: out, at }
}

/** Index of the last cue starting at or before `seconds`. */
function cueAt(cues: Cue[], seconds: number): number {
  let lo = 0
  let hi = cues.length - 1
  let found = -1
  while (lo <= hi) {
    const mid = (lo + hi) >> 1
    if (cues[mid].start <= seconds) {
      found = mid
      lo = mid + 1
    } else hi = mid - 1
  }
  return found
}

interface Stamp {
  seconds: number
  paragraph: HTMLElement
  link: HTMLAnchorElement
}

function parseStamp(href: string): number | null {
  try {
    const url = new URL(href)
    if (!/(^|\.)youtu(\.be|be\.com)$/.test(url.hostname)) return null
    const t = url.searchParams.get('t')
    if (!t) return null
    const seconds = Number.parseInt(t, 10)
    return Number.isFinite(seconds) ? seconds : null
  } catch {
    return null
  }
}

function formatStamp(total: number, hours: boolean): string {
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const s = Math.floor(total % 60)
  const mm = hours || h > 0 ? String(m).padStart(2, '0') : String(m)
  return h > 0 || hours
    ? `${h}:${mm}:${String(s).padStart(2, '0')}`
    : `${mm}:${String(s).padStart(2, '0')}`
}

/** Index of the last stamp at or before `seconds`. */
function stampAt(stamps: Stamp[], seconds: number): number {
  let lo = 0
  let hi = stamps.length - 1
  let found = -1
  while (lo <= hi) {
    const mid = (lo + hi) >> 1
    if (stamps[mid].seconds <= seconds) {
      found = mid
      lo = mid + 1
    } else hi = mid - 1
  }
  return found
}

document.addEventListener('nav', () => {
  const root = document.querySelector<HTMLElement>('[data-course-watch]')
  if (!root) return
  const signal = currentNavSignal()
  const youtube = root.dataset.courseWatch ?? ''
  const frame = root.querySelector<HTMLElement>('.course-player')
  const status = root.querySelector<HTMLElement>('.course-player-status')
  const list = root.querySelector<HTMLElement>('.course-transcript-list')
  const pane = root.querySelector<HTMLElement>('.course-transcript-pane') ?? list
  const search = root.querySelector<HTMLInputElement>('.course-transcript-search')
  const hits = root.querySelector<HTMLElement>('.course-transcript-hits')
  const followButton = root.querySelector<HTMLButtonElement>('[data-transcript-follow]')
  const prevButton = root.querySelector<HTMLButtonElement>('[data-transcript-prev]')
  const nextButton = root.querySelector<HTMLButtonElement>('[data-transcript-next]')
  const copyButton = root.querySelector<HTMLButtonElement>('[data-transcript-copy]')
  if (!frame || !list) return

  const stamps: Stamp[] = []
  for (const paragraph of list.querySelectorAll<HTMLElement>('p')) {
    const link = paragraph.querySelector<HTMLAnchorElement>('a[href]')
    if (!link || link !== paragraph.firstElementChild) continue
    const seconds = parseStamp(link.href)
    if (seconds === null) continue
    link.classList.add('course-stamp')
    stamps.push({ seconds, paragraph, link })
  }
  stamps.sort((a, b) => a.seconds - b.seconds)
  const hours = (stamps[stamps.length - 1]?.seconds ?? 0) >= 3600

  let iframe: HTMLIFrameElement | null = null
  let ready = false
  let failed = false
  let readyTimer = 0
  let current = -1
  let follow = followButton?.getAttribute('aria-pressed') !== 'false'
  let lastUserScroll = 0
  let box: RoughAnnotation | null = null
  const inkColor = getComputedStyle(root).getPropertyValue('--secondary').trim() || 'currentColor'
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
  const cues: Cue[] = []
  const cueMarks = new Map<number, RoughAnnotation>()
  let currentCue = -1
  let playing = false
  let lastTime = 0
  signal.addEventListener(
    'abort',
    () => {
      box?.remove()
      for (const mark of cueMarks.values()) mark.remove()
    },
    { once: true },
  )

  // The box is a sibling svg after the paragraph (rough-notation), so it scrolls with the pane.
  const drawBox = (paragraph: HTMLElement) => {
    box?.remove()
    box = annotate(paragraph, {
      type: 'box',
      color: inkColor,
      strokeWidth: 1.6,
      padding: [2, 6, 2, 4],
      iterations: 2,
      animate: !reducedMotion,
      animationDuration: 450,
    })
    box.show()
  }

  // Wraps each cue's words of a paragraph in a span. The rendered text is the cues joined by one
  // space (plus typographic marks), so each cue is found in order from a cursor.
  const wrapCues = (paragraph: HTMLElement, group: VttCue[], index: number): boolean => {
    const after: Text[] = []
    for (const node of Array.from(paragraph.childNodes)) {
      if (node instanceof HTMLAnchorElement && node.classList.contains('course-stamp')) continue
      if (node.nodeType !== Node.TEXT_NODE) return false
      after.push(node as Text)
    }
    const source = after.map(node => node.data).join('')
    const { text, at } = plain(source)
    let cursor = 0
    const pieces: { cue?: VttCue; text: string }[] = []
    for (const cue of group) {
      const needle = plain(cue.text).text
      const hit = text.indexOf(needle, cursor)
      if (hit === -1) return false
      const from = at[hit]
      const to = at[hit + needle.length - 1] + 1
      const gap = source.slice(at[cursor] ?? source.length, from)
      if (gap.length > 0) pieces.push({ text: gap })
      pieces.push({ cue, text: source.slice(from, to) })
      cursor = hit + needle.length
    }
    const tailFrom = cursor < text.length ? at[cursor] : source.length
    if (tailFrom < source.length) pieces.push({ text: source.slice(tailFrom) })
    for (const node of after) node.remove()
    for (const piece of pieces) {
      if (!piece.cue) {
        paragraph.append(piece.text)
        continue
      }
      const span = document.createElement('span')
      span.className = 'course-cue'
      span.textContent = piece.text
      paragraph.append(span)
      cues.push({ start: piece.cue.start, end: piece.cue.end, span, paragraph: index })
    }
    return true
  }

  const markPaths = (cue: Cue): HTMLElement[] => {
    const svg = cue.span.nextElementSibling
    return svg instanceof SVGElement && svg.classList.contains('rough-annotation')
      ? Array.from(svg.querySelectorAll<HTMLElement>('path'))
      : []
  }

  // Retimes the sweep so its progress matches the player: delay = base - elapsed in the cue.
  const syncMark = (cue: Cue, seconds: number) => {
    const offset = Math.max(0, (seconds - cue.start) * 1000)
    for (const path of markPaths(cue)) {
      const base = Number(path.dataset.delay ?? 0)
      path.style.animationDelay = `${base - offset}ms`
      path.style.animationPlayState = playing ? 'running' : 'paused'
    }
  }

  // Underline across the cue text; animated at the cue's own duration, so the ink reaches each
  // word as it is spoken. Earlier cues of the paragraph (after a seek) are inked whole.
  const inkCue = (cue: Cue, sweep: boolean, seconds: number): RoughAnnotation => {
    const mark = annotate(cue.span, {
      type: 'underline',
      color: inkColor,
      strokeWidth: 1.6,
      padding: 1,
      iterations: 1,
      multiline: true,
      animate: sweep && !reducedMotion,
      animationDuration: Math.max(CUE_SWEEP_MIN, (cue.end - cue.start) * 1000),
    })
    mark.show()
    if (sweep && !reducedMotion) {
      for (const path of markPaths(cue)) {
        path.style.animationTimingFunction = 'linear'
        path.dataset.delay = String(parseFloat(path.style.animationDelay) || 0)
      }
      syncMark(cue, seconds)
    }
    return mark
  }

  const setCue = (seconds: number) => {
    const index = cueAt(cues, seconds)
    const cue = cues[index]
    const backward = seconds < lastTime - 0.5
    lastTime = seconds
    if (index === currentCue) {
      if (cue && backward) {
        cueMarks.get(index)?.remove()
        cueMarks.set(index, inkCue(cue, true, seconds))
      } else if (cue && playing) {
        syncMark(cue, seconds)
      }
      return
    }
    for (const [key, mark] of cueMarks) {
      if (!cue || key > index || cues[key].paragraph !== cue.paragraph) {
        mark.remove()
        cueMarks.delete(key)
      }
    }
    currentCue = index
    if (!cue) return
    for (let k = index - 1; k >= 0 && cues[k].paragraph === cue.paragraph; k--) {
      if (!cueMarks.has(k)) cueMarks.set(k, inkCue(cues[k], false, seconds))
    }
    cueMarks.set(index, inkCue(cue, true, seconds))
  }

  const setPlaying = (state: boolean) => {
    if (state === playing) return
    playing = state
    const cue = cues[currentCue]
    if (cue) syncMark(cue, lastTime)
  }

  const captionsUrl = root.dataset.courseWatchCaptions
  if (captionsUrl && stamps.length > 0) {
    fetch(captionsUrl, { signal })
      .then(response => (response.ok ? response.text() : ''))
      .then(text => {
        if (signal.aborted || !text) return
        const groups = groupCues(parseVtt(text).cues)
        // same grouping as the vendor script: paragraphs must line up with the stamps
        if (groups.length !== stamps.length) return
        if (groups.some((group, i) => Math.floor(group.start) !== stamps[i].seconds)) return
        for (const [i, group] of groups.entries()) wrapCues(stamps[i].paragraph, group.cues, i)
        if (currentCue === -1 && lastTime > 0) setCue(lastTime)
      })
      .catch(() => {})
  }

  const say = (text: string) => {
    if (status) status.textContent = text
  }

  const post = (payload: Record<string, unknown>) => {
    iframe?.contentWindow?.postMessage(JSON.stringify(payload), '*')
  }

  const command = (func: string, args: unknown[] = []) =>
    post({ event: 'command', func, args, id: 1, channel: 'widget' })

  const setCurrent = (index: number) => {
    if (index === current) return
    if (current >= 0) stamps[current]?.paragraph.removeAttribute('aria-current')
    current = index
    const stamp = stamps[index]
    if (!stamp) return
    stamp.paragraph.setAttribute('aria-current', 'time')
    if (follow && Date.now() - lastUserScroll > USER_SCROLL_HOLD) {
      stamp.paragraph.scrollIntoView({ block: 'center', behavior: 'smooth' })
    }
    drawBox(stamp.paragraph)
  }

  const mount = (start: number, autoplay: boolean) => {
    if (iframe) return
    const params = new URLSearchParams({
      enablejsapi: '1',
      origin: location.origin,
      rel: '0',
      playsinline: '1',
      start: String(Math.max(0, Math.floor(start))),
      autoplay: autoplay ? '1' : '0',
    })
    iframe = document.createElement('iframe')
    iframe.className = 'course-player-frame'
    iframe.src = `https://www.youtube-nocookie.com/embed/${encodeURIComponent(youtube)}?${params}`
    iframe.title = root.dataset.courseWatchTitle ?? 'lecture video'
    iframe.allow = 'autoplay; encrypted-media; picture-in-picture; fullscreen'
    iframe.setAttribute('allowfullscreen', '')
    iframe.addEventListener(
      'load',
      () => {
        post({ event: 'listening', id: 1, channel: 'widget' })
      },
      { signal },
    )
    readyTimer = window.setTimeout(() => {
      if (ready || signal.aborted) return
      failed = true
      say('the player did not answer; timestamps open youtube')
    }, READY_TIMEOUT)
    frame.replaceChildren(iframe)
    frame.dataset.state = 'mounted'
  }

  const seek = (seconds: number) => {
    if (!iframe) {
      mount(seconds, true)
      return
    }
    command('seekTo', [seconds, true])
    command('playVideo')
    setCurrent(stampAt(stamps, seconds))
    setCue(seconds)
  }

  const onMessage = (event: MessageEvent) => {
    if (!iframe || event.source !== iframe.contentWindow || !YOUTUBE_ORIGINS.has(event.origin))
      return
    let data: unknown = event.data
    if (typeof data === 'string') {
      try {
        data = JSON.parse(data)
      } catch {
        return
      }
    }
    if (typeof data !== 'object' || data === null) return
    const message = data as { event?: string; info?: unknown }
    if (message.event === 'onReady') {
      ready = true
      window.clearTimeout(readyTimer)
      say('')
      return
    }
    if (message.event === 'onError') {
      failed = true
      const code = typeof message.info === 'number' ? message.info : 0
      say(
        code === 101 || code === 150
          ? 'the owner does not allow this video to play here; timestamps open youtube'
          : `youtube refused to play this video (error ${code || 'unknown'}); timestamps open youtube`,
      )
      return
    }
    if (message.event === 'infoDelivery' && typeof message.info === 'object' && message.info) {
      const info = message.info as { currentTime?: unknown; playerState?: unknown }
      if (typeof info.playerState === 'number') setPlaying(info.playerState === 1)
      if (typeof info.currentTime === 'number') {
        ready = true
        window.clearTimeout(readyTimer)
        setCurrent(stampAt(stamps, info.currentTime))
        setCue(info.currentTime)
      }
    }
  }
  window.addEventListener('message', onMessage, { signal })

  mount(0, false)

  list.addEventListener(
    'click',
    event => {
      if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey)
        return
      const link = (event.target as HTMLElement).closest<HTMLAnchorElement>('a.course-stamp')
      if (!link || failed) return
      const seconds = parseStamp(link.href)
      if (seconds === null) return
      event.preventDefault()
      seek(seconds)
    },
    { signal },
  )

  pane?.addEventListener(
    'scroll',
    () => {
      lastUserScroll = Date.now()
    },
    { passive: true, signal },
  )

  followButton?.addEventListener(
    'click',
    () => {
      follow = !follow
      followButton.setAttribute('aria-pressed', String(follow))
      if (follow && current >= 0) stamps[current].paragraph.scrollIntoView({ block: 'center' })
    },
    { signal },
  )

  const stepStamp = (direction: 1 | -1) => {
    if (stamps.length === 0) return
    const next = Math.min(stamps.length - 1, Math.max(0, (current < 0 ? 0 : current) + direction))
    seek(stamps[next].seconds)
    if (!ready) setCurrent(next)
  }
  prevButton?.addEventListener('click', () => stepStamp(-1), { signal })
  nextButton?.addEventListener('click', () => stepStamp(1), { signal })

  copyButton?.addEventListener(
    'click',
    async () => {
      const title = root.dataset.courseWatchTitle ?? document.title
      const source = root.dataset.courseWatchSource ?? location.href
      const lines = stamps.map(stamp => {
        const text = stamp.paragraph.textContent?.replace(stamp.link.textContent ?? '', '').trim()
        return `[${formatStamp(stamp.seconds, hours)}] ${text}`
      })
      const body = `${title}\n${source}\nmit opencourseware, cc by-nc-sa 4.0\n\n${lines.join('\n')}\n`
      try {
        await copyText(body)
        copyButton.textContent = 'copied'
        window.setTimeout(() => {
          if (!signal.aborted) copyButton.textContent = 'copy'
        }, 1500)
      } catch {
        copyButton.textContent = 'copy failed'
      }
    },
    { signal },
  )

  // Search: ranges over the paragraph text nodes, painted by ::highlight(course-transcript).
  const highlightApi = typeof CSS !== 'undefined' && 'highlights' in CSS
  const highlightName = 'course-transcript'
  let matches: { paragraph: HTMLElement; range: Range }[] = []
  let active = -1

  const clearMatches = () => {
    if (highlightApi) CSS.highlights.delete(highlightName)
    for (const { paragraph } of matches) delete paragraph.dataset.match
    matches = []
    active = -1
    if (hits) hits.textContent = ''
  }

  const showActive = () => {
    if (hits)
      hits.textContent = matches.length > 0 ? `${active + 1} of ${matches.length}` : 'no matches'
    for (const [index, match] of matches.entries()) {
      if (index === active) match.paragraph.dataset.match = 'active'
      else if (!highlightApi) match.paragraph.dataset.match = 'hit'
      else delete match.paragraph.dataset.match
    }
    const target = matches[active]
    if (target) {
      lastUserScroll = Date.now()
      target.paragraph.scrollIntoView({ block: 'center' })
    }
  }

  const runSearch = () => {
    clearMatches()
    const query = search?.value.trim().toLowerCase() ?? ''
    if (query.length < 2) return
    const walker = document.createTreeWalker(list, NodeFilter.SHOW_TEXT)
    let node: Text | null
    while ((node = walker.nextNode() as Text | null) && matches.length < MAX_HIGHLIGHTS) {
      const haystack = node.data.toLowerCase()
      const paragraph = node.parentElement?.closest<HTMLElement>('p')
      if (!paragraph || node.parentElement?.closest('a.course-stamp')) continue
      let at = haystack.indexOf(query)
      while (at >= 0 && matches.length < MAX_HIGHLIGHTS) {
        const range = new Range()
        range.setStart(node, at)
        range.setEnd(node, at + query.length)
        matches.push({ paragraph, range })
        at = haystack.indexOf(query, at + query.length)
      }
    }
    if (highlightApi) CSS.highlights.set(highlightName, new Highlight(...matches.map(m => m.range)))
    active = matches.length > 0 ? 0 : -1
    showActive()
  }

  let searchTimer = 0
  search?.addEventListener(
    'input',
    () => {
      window.clearTimeout(searchTimer)
      searchTimer = window.setTimeout(runSearch, 120)
    },
    { signal },
  )
  search?.addEventListener(
    'keydown',
    event => {
      if (event.key === 'Enter') {
        event.preventDefault()
        if (matches.length === 0) return
        active = (active + (event.shiftKey ? -1 : 1) + matches.length) % matches.length
        showActive()
      } else if (event.key === 'Escape') {
        event.preventDefault()
        search.value = ''
        clearMatches()
        search.blur()
      }
      event.stopPropagation()
    },
    { signal },
  )

  window.addCleanup(() => {
    window.clearTimeout(readyTimer)
    window.clearTimeout(searchTimer)
    clearMatches()
    iframe?.remove()
    iframe = null
  })
})
