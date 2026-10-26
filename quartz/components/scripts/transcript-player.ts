import type { RoughAnnotation } from 'rough-notation/lib/model'
import { annotate } from 'rough-notation'
import { groupCues, parseVtt, type VttCue } from '../../util/vtt'
import { createCueHighlighter, type Cue } from './transcript-player/cue-highlighter'
import { copyText, plain, parseStamp, formatStamp } from './transcript-text'

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
const rootsWithPlayerCleanup = new WeakSet<HTMLElement>()

interface Stamp {
  seconds: number
  paragraph: HTMLElement
  link: HTMLAnchorElement
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

interface TranscriptPlayerOptions {
  reusePlayer?: boolean
}

export function mountTranscriptPlayer(
  root: HTMLElement,
  signal: AbortSignal,
  { reusePlayer = false }: TranscriptPlayerOptions = {},
) {
  if (signal.aborted) return
  const youtube = root.dataset.courseWatch ?? ''
  const frame = root.querySelector<HTMLElement>('.course-player')
  const status = root.querySelector<HTMLElement>('.course-player-status')
  const list =
    root.querySelector<HTMLElement>('.course-transcript-list') ?? document.createElement('div')
  const pane = root.querySelector<HTMLElement>('.course-transcript-pane') ?? list
  const search = root.querySelector<HTMLInputElement>('.course-transcript-search')
  const hits = root.querySelector<HTMLElement>('.course-transcript-hits')
  const followButton = root.querySelector<HTMLButtonElement>('[data-transcript-follow]')
  const prevButton = root.querySelector<HTMLButtonElement>('[data-transcript-prev]')
  const nextButton = root.querySelector<HTMLButtonElement>('[data-transcript-next]')
  const copyButton = root.querySelector<HTMLButtonElement>('[data-transcript-copy]')
  const findButton = root.querySelector<HTMLButtonElement>('[data-transcript-find]')
  const findField = search?.closest<HTMLElement>('.course-transcript-find')
  if (!frame) return

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

  const adoptedIframe = reusePlayer
    ? frame.querySelector<HTMLIFrameElement>('.course-player-frame')
    : null
  let iframe = adoptedIframe
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
  const cueHighlighter = createCueHighlighter(cues, inkColor, reducedMotion)
  let lastTime = 0
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
        if (lastTime > 0) cueHighlighter.setCue(lastTime)
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

  const reveal = (paragraph: HTMLElement, smooth = false) => {
    if (!pane) return
    const bounds = paragraph.getBoundingClientRect()
    pane.scrollBy({
      top: bounds.top - pane.getBoundingClientRect().top - (pane.clientHeight - bounds.height) / 2,
      behavior: smooth && !reducedMotion ? 'smooth' : 'instant',
    })
  }

  const setCurrent = (index: number) => {
    if (index === current) return
    if (current >= 0) stamps[current]?.paragraph.removeAttribute('aria-current')
    current = index
    const stamp = stamps[index]
    if (!stamp) return
    stamp.paragraph.setAttribute('aria-current', 'time')
    if (follow && Date.now() - lastUserScroll > USER_SCROLL_HOLD) {
      reveal(stamp.paragraph, true)
    }
    drawBox(stamp.paragraph)
  }

  const mount = (start: number, autoplay: boolean) => {
    if (iframe) return
    const params = new URLSearchParams(root.dataset.courseWatchParams)
    for (const [key, value] of Object.entries({
      enablejsapi: '1',
      origin: location.origin,
      rel: '0',
      playsinline: '1',
      start: String(Math.max(0, Math.floor(start))),
      autoplay: autoplay ? '1' : '0',
    }))
      params.set(key, value)
    if (root.dataset.transcriptLanguage) {
      params.set('cc_lang_pref', root.dataset.transcriptLanguage)
      params.set('cc_load_policy', '1')
    }
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
    lastTime = seconds
    cueHighlighter.setCue(seconds)
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
      failed = false
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
      if (typeof info.playerState === 'number') cueHighlighter.setPlaying(info.playerState === 1)
      if (typeof info.currentTime === 'number') {
        ready = true
        failed = false
        window.clearTimeout(readyTimer)
        setCurrent(stampAt(stamps, info.currentTime))
        lastTime = info.currentTime
        cueHighlighter.setCue(info.currentTime)
      }
    }
  }
  window.addEventListener('message', onMessage, { signal })

  mount(Number(root.dataset.courseWatchStart ?? 0), false)
  if (adoptedIframe) {
    adoptedIframe.addEventListener(
      'load',
      () => post({ event: 'listening', id: 1, channel: 'widget' }),
      { signal },
    )
    frame.dataset.state = 'mounted'
    post({ event: 'listening', id: 1, channel: 'widget' })
    readyTimer = window.setTimeout(() => {
      if (ready || signal.aborted) return
      failed = true
      say('the player did not answer; timestamps open youtube')
    }, READY_TIMEOUT)
  }

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
      if (follow && current >= 0) reveal(stamps[current].paragraph)
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

  let copyTimer = 0
  copyButton?.addEventListener(
    'click',
    async () => {
      const title = root.dataset.courseWatchTitle ?? document.title
      const source = root.dataset.courseWatchSource ?? location.href
      const lines = stamps.map(stamp => {
        const text = stamp.paragraph.textContent?.replace(stamp.link.textContent ?? '', '').trim()
        return `[${formatStamp(stamp.seconds, hours)}] ${text}`
      })
      const attribution = root.dataset.transcriptAttribution ?? ''
      const body = `${title}\n${source}\n${attribution}\n\n${lines.join('\n')}\n`
      try {
        await copyText(body)
        if (signal.aborted) return
        copyButton.dataset.copied = 'done'
        copyButton.setAttribute('aria-label', 'copied')
        window.clearTimeout(copyTimer)
        copyTimer = window.setTimeout(() => {
          if (signal.aborted) return
          delete copyButton.dataset.copied
          copyButton.setAttribute('aria-label', 'copy the transcript')
        }, 1500)
      } catch {
        if (!signal.aborted) copyButton.setAttribute('aria-label', 'copy failed')
      }
    },
    { signal },
  )

  // Search: ranges over the paragraph text nodes, painted by ::highlight(course-transcript).
  const highlightApi = typeof CSS !== 'undefined' && 'highlights' in CSS
  const highlightName = `course-transcript-${root.dataset.transcriptInstance ?? 'course'}`
  const highlightStyle = document.createElement('style')
  highlightStyle.textContent = `::highlight(${highlightName}) { background: color-mix(in srgb, var(--tertiary) 25%, var(--light)); color: var(--dark); }`
  root.append(highlightStyle)
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
      reveal(target.paragraph)
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
        setFindOpen(false)
        findButton?.focus()
      }
      event.stopPropagation()
    },
    { signal },
  )

  // The magnifier opens the field beside it; an empty field closes again when focus leaves it.
  const setFindOpen = (open: boolean) => {
    if (!findField || !findButton) return
    findField.hidden = !open
    findButton.setAttribute('aria-expanded', String(open))
    if (open) search?.focus()
  }
  findButton?.addEventListener(
    'click',
    () => setFindOpen(findField?.hidden !== false || search?.value.trim() !== ''),
    { signal },
  )
  search?.addEventListener(
    'blur',
    event => {
      if (search.value.trim() === '' && event.relatedTarget !== findButton) setFindOpen(false)
    },
    { signal },
  )

  const cleanup = () => {
    window.clearTimeout(readyTimer)
    window.clearTimeout(searchTimer)
    window.clearTimeout(copyTimer)
    clearMatches()
    highlightStyle.remove()
    box?.remove()
    box = null
    cueHighlighter.dispose()
  }
  signal.addEventListener('abort', cleanup, { once: true })
  if (!rootsWithPlayerCleanup.has(root)) {
    rootsWithPlayerCleanup.add(root)
    window.addCleanup(() => frame.querySelector('.course-player-frame')?.remove())
  }
}
