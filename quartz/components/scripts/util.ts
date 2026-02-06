import { FullSlug, getFullSlug, resolveRelative } from '../../util/path'

export function registerEscapeHandler(outsideContainer: HTMLElement | null, cb: () => void) {
  if (!outsideContainer) return
  function click(this: HTMLElement, e: HTMLElementEventMap['click']) {
    if (e.target !== this) return
    e.preventDefault()
    e.stopPropagation()
    cb()
  }

  function esc(e: HTMLElementEventMap['keydown']) {
    if (!e.key.startsWith('Esc')) return
    e.preventDefault()
    cb()
  }

  outsideContainer?.addEventListener('click', click)
  window.addCleanup(() => outsideContainer?.removeEventListener('click', click))
  document.addEventListener('keydown', esc)
  window.addCleanup(() => document.removeEventListener('keydown', esc))
}

export function removeAllChildren(node: HTMLElement) {
  while (node.firstChild) {
    node.removeChild(node.firstChild)
  }
}

export function registerMouseHover(el: HTMLElement, ...classList: string[]) {
  const onMouseEnter = () => el.classList.add(...classList)
  const onMouseLeave = () => el.classList.remove(...classList)

  registerEvents(el, ['mouseenter', onMouseEnter], ['mouseleave', onMouseLeave])
}

type EventType = HTMLElementEventMap[keyof HTMLElementEventMap]
type EventHandlers<E extends EventType> = (evt: E) => any | void

export function registerEvents<
  T extends Document | HTMLElement | null,
  E extends keyof HTMLElementEventMap,
>(element: T, ...events: [E, EventHandlers<HTMLElementEventMap[E]>][]) {
  if (!element) return

  events.forEach(([event, cb]) => {
    const listener: EventListener = evt => cb(evt as HTMLElementEventMap[E])
    element.addEventListener(event, listener)
    window.addCleanup(() => element.removeEventListener(event, listener))
  })
}

export function decodeString(el: HTMLSpanElement, targetString: string, duration: number = 1000) {
  const start = performance.now()
  const end = start + duration

  function update() {
    const current = performance.now()
    const progress = (current - start) / duration
    const currentIndex = Math.floor(progress * targetString.length)

    if (current < end) {
      let decodingString =
        targetString.substring(0, currentIndex) +
        getRandomString(targetString.length - currentIndex)
      el.textContent = decodingString
      requestAnimationFrame(update)
    } else {
      el.textContent = targetString
    }
  }

  requestAnimationFrame(update)
}

export function getRandomString(length: number) {
  const characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
  let result = ''
  for (let i = 0; i < length; i++) {
    result += characters.charAt(Math.floor(Math.random() * characters.length))
  }
  return result
}

export function isInViewport(element: HTMLElement, buffer: number = 100) {
  const rect = element.getBoundingClientRect()
  return (
    rect.top >= -buffer &&
    rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) + buffer
  )
}

// Computes an offset such that setting `top` on elemToAlign will put it
// in vertical alignment with targetAlignment.
function computeOffsetForAlignment(elemToAlign: HTMLElement, targetAlignment: HTMLElement) {
  const elemRect = elemToAlign.getBoundingClientRect()
  const targetRect = targetAlignment.getBoundingClientRect()
  const parentRect = elemToAlign.parentElement?.getBoundingClientRect() || elemRect
  return targetRect.top - parentRect.top
}

// Get bounds for the sidenote positioning
function getBounds(parent: HTMLElement, child: HTMLElement): { min: number; max: number } {
  const containerRect = parent.getBoundingClientRect()
  const sidenoteRect = child.getBoundingClientRect()

  return { min: 0, max: containerRect.height - sidenoteRect.height }
}

export function updatePosition(ref: HTMLElement, child: HTMLElement, parent: HTMLElement) {
  // Calculate ideal position
  let referencePosition = computeOffsetForAlignment(child, ref)

  // Get bounds for this sidenote
  const bounds = getBounds(parent, child)

  // Clamp the position within bounds
  referencePosition = Math.max(referencePosition, Math.min(bounds.min, bounds.max))

  // Apply position
  child.style.top = `${referencePosition}px`
}

export function debounce(fn: Function, delay: number) {
  let timeoutId: ReturnType<typeof setTimeout>
  return (...args: any[]) => {
    clearTimeout(timeoutId)
    timeoutId = setTimeout(() => fn(...args), delay)
  }
}

export interface DagNode {
  slug: string
  title: string
  contents: HTMLElement[]
  note: HTMLElement
  anchor?: HTMLElement | null
  hash?: string
}

export class Dag {
  private nodes: Map<string, DagNode>
  private order: string[] // Maintain order of nodes

  constructor() {
    this.nodes = new Map()
    this.order = []
  }

  addNode(node: DagNode) {
    const { slug } = node
    if (!this.nodes.has(slug)) {
      this.nodes.set(slug, node)
      this.order.push(slug)
    }
    return this.nodes.get(slug)!
  }

  getOrderedNodes(): DagNode[] {
    return this.order.map(slug => this.nodes.get(slug)!).filter(Boolean)
  }

  truncateAfter(slug: string) {
    const idx = this.order.indexOf(slug)
    if (idx === -1) return

    // Remove all nodes after idx from both order and nodes map
    const removed = this.order.splice(idx + 1)
    removed.forEach(slug => this.nodes.delete(slug))
  }

  clear() {
    this.nodes.clear()
    this.order = []
  }

  has(slug: string): boolean {
    return this.nodes.has(slug)
  }

  get(slug: string): DagNode | undefined {
    return this.nodes.get(slug)
  }

  getTail(): DagNode | undefined {
    const lastSlug = this.order[this.order.length - 1]
    return lastSlug ? this.nodes.get(lastSlug) : undefined
  }
}

// AliasRedirect emits HTML redirects which also have the link[rel="canonical"]
// containing the URL it's redirecting to.
// Extracting it here with regex is _probably_ faster than parsing the entire HTML
// with a DOMParser effectively twice (here and later in the SPA code), even if
// way less robust - we only care about our own generated redirects after all.
const canonicalRegex = /<link rel="canonical" href="([^"]*)">/
export async function fetchCanonical(url: URL, init?: RequestInit): Promise<Response> {
  const res = await fetch(`${url}`, init)
  if (!res.headers.get('content-type')?.startsWith('text/html')) {
    return res
  }
  // reading the body can only be done once, so we need to clone the response
  // to allow the caller to read it if it's was not a redirect
  const text = await res.clone().text()
  const [_, redirect] = text.match(canonicalRegex) ?? []
  return redirect ? fetch(`${new URL(redirect, url)}`) : res
}

export function isBrowser() {
  return typeof window !== 'undefined'
}

export function isStreamHost(): boolean {
  return window.location.hostname === 'stream.aarnphm.xyz'
}

const contextWindowWords = 30
export const tokenizeTerm = (term: string) => {
  const tokens = term.split(/\s+/).filter(t => t.trim() !== '')
  const tokenLen = tokens.length
  if (tokenLen > 1) {
    for (let i = 1; i < tokenLen; i++) {
      tokens.push(tokens.slice(0, i + 1).join(' '))
    }
  }

  return tokens.sort((a, b) => b.length - a.length) // always highlight longest terms first
}

export function highlight(searchTerm: string, text: string, trim?: boolean) {
  const tokenizedTerms = tokenizeTerm(searchTerm)
  let tokenizedText = text.split(/\s+/).filter(t => t !== '')

  let startIndex = 0
  let endIndex = tokenizedText.length - 1
  if (trim) {
    const includesCheck = (tok: string) =>
      tokenizedTerms.some(term => tok.toLowerCase().startsWith(term.toLowerCase()))
    const occurrencesIndices = tokenizedText.map(includesCheck)

    let bestSum = 0
    let bestIndex = 0
    for (let i = 0; i < Math.max(tokenizedText.length - contextWindowWords, 0); i++) {
      const windowIndices = occurrencesIndices.slice(i, i + contextWindowWords)
      const windowSum = windowIndices.reduce((total, cur) => total + (cur ? 1 : 0), 0)
      if (windowSum >= bestSum) {
        bestSum = windowSum
        bestIndex = i
      }
    }

    startIndex = Math.max(bestIndex - contextWindowWords, 0)
    endIndex = Math.min(startIndex + 2 * contextWindowWords, tokenizedText.length - 1)
    tokenizedText = tokenizedText.slice(startIndex, endIndex)
  }

  const slice = tokenizedText
    .map(tok => {
      // see if this tok is prefixed by any search terms
      for (const searchTok of tokenizedTerms) {
        if (tok.toLowerCase().includes(searchTok.toLowerCase())) {
          const regex = new RegExp(searchTok.toLowerCase(), 'gi')
          return tok.replace(regex, `<span class="highlight">$&</span>`)
        }
      }
      return tok
    })
    .join(' ')

  return `${startIndex === 0 ? '' : '...'}${slice}${
    endIndex === tokenizedText.length - 1 ? '' : '...'
  }`
}

// To be used with search and everything else with flexsearch
export const encode = (str: string): string[] => {
  const tokens: string[] = []
  let bufferStart = -1
  let bufferEnd = -1
  const lower = str.toLowerCase()

  let i = 0
  for (const char of lower) {
    const code = char.codePointAt(0)!

    const isCJK =
      (code >= 0x3040 && code <= 0x309f) ||
      (code >= 0x30a0 && code <= 0x30ff) ||
      (code >= 0x4e00 && code <= 0x9fff) ||
      (code >= 0xac00 && code <= 0xd7af) ||
      (code >= 0x20000 && code <= 0x2a6df)

    const isWhitespace = code === 32 || code === 9 || code === 10 || code === 13

    if (isCJK) {
      if (bufferStart !== -1) {
        tokens.push(lower.slice(bufferStart, bufferEnd))
        bufferStart = -1
      }
      tokens.push(char)
    } else if (isWhitespace) {
      if (bufferStart !== -1) {
        tokens.push(lower.slice(bufferStart, bufferEnd))
        bufferStart = -1
      }
    } else {
      if (bufferStart === -1) bufferStart = i
      bufferEnd = i + char.length
    }

    i += char.length
  }

  if (bufferStart !== -1) {
    tokens.push(lower.slice(bufferStart))
  }

  return tokens
}

export function getOrCreateSidePanel(): HTMLDivElement {
  let asidePanel = document.querySelector<HTMLDivElement>(
    "main > * > aside[class~='sidepanel-container']",
  )

  if (!asidePanel) {
    const pageContent = document.querySelector<HTMLDivElement>(
      'main > div[class~="page-body-grid"]',
    )
    if (!pageContent) {
      throw new Error('page-content section not found')
    }

    asidePanel = document.createElement('aside') as HTMLDivElement
    asidePanel.classList.add('sidepanel-container')
    pageContent.appendChild(asidePanel)
  }

  return asidePanel
}

export function createSidePanel(asidePanel: HTMLDivElement, ...inner: HTMLElement[]) {
  const pageHeader = document.querySelector<HTMLDivElement>(
    "main > * > section[class~='page-header']",
  )
  if (!asidePanel || !pageHeader) console.error('asidePanel must not be null')

  asidePanel.classList.add('active')
  removeAllChildren(asidePanel)

  const updateSidepanelOffset = () => {
    const headerSection = document.querySelector<HTMLElement>('main > section.header')
    if (!headerSection) {
      asidePanel.style.setProperty('--sidepanel-top-offset', '0px')
      return
    }

    const headerRect = headerSection.getBoundingClientRect()
    const stickyTop = parseFloat(getComputedStyle(headerSection).top || '0') || 0
    const offset = Math.max(0, headerRect.height + stickyTop)
    asidePanel.style.setProperty('--sidepanel-top-offset', `${offset}px`)
  }

  updateSidepanelOffset()

  const handleResize = () => updateSidepanelOffset()
  window.addEventListener('resize', handleResize)
  window.addCleanup(() => window.removeEventListener('resize', handleResize))

  let resizeObserver: ResizeObserver | null = null
  if (typeof ResizeObserver !== 'undefined') {
    const headerSection = document.querySelector<HTMLElement>('main > * > section[class~="header"')
    if (headerSection) {
      resizeObserver = new ResizeObserver(() => updateSidepanelOffset())
      resizeObserver.observe(headerSection)
      window.addCleanup(() => resizeObserver?.disconnect())
    }
  }

  const header = document.createElement('div')
  header.classList.add('sidepanel-header', 'all-col')

  const closeButton = document.createElement('button')
  closeButton.classList.add('close-button')
  closeButton.ariaLabel = 'close button'
  closeButton.title = 'close button'
  closeButton.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width=16 height=16 viewbox="0 0 24 24" fill="currentColor" stroke="currentColor"><use href="#close-button"></svg>`
  function onCloseClick() {
    removeAllChildren(asidePanel)
    asidePanel.classList.remove('active')
    asidePanel.style.removeProperty('--sidepanel-top-offset')
    window.removeEventListener('resize', handleResize)
    resizeObserver?.disconnect()
    resizeObserver = null
  }
  closeButton.addEventListener('click', onCloseClick)
  window.addCleanup(() => closeButton.removeEventListener('click', onCloseClick))

  const redirectButton = document.createElement('button')
  redirectButton.classList.add('redirect-button')
  redirectButton.ariaLabel = 'redirect to page'
  redirectButton.title = 'redirect to page'
  redirectButton.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width=16 height=16 viewbox="0 0 24 24" fill="var(--gray)" stroke="none"><use href="#triple-dots"></svg>`
  function onRedirectClick() {
    window.spaNavigate(
      new URL(
        resolveRelative(getFullSlug(window), asidePanel.dataset.slug as FullSlug),
        window.location.toString(),
      ),
    )
  }
  redirectButton.addEventListener('click', onRedirectClick)
  window.addCleanup(() => redirectButton.removeEventListener('click', onRedirectClick))

  header.appendChild(redirectButton)
  header.appendChild(closeButton)

  const sideInner = document.createElement('div')
  sideInner.classList.add('sidepanel-inner')
  sideInner.append(...inner, header)
  asidePanel.appendChild(sideInner)

  return sideInner
}

/**
 * Wraps a DOM update in a View Transition if supported by the browser.
 * Falls back to immediate execution if the API is unavailable.
 * @param callback - The function containing DOM updates to animate
 */
export function startViewTransition(callback: () => void): void {
  callback()
}
