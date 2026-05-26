import micromorph from 'micromorph'
import {
  FullSlug,
  RelativeURL,
  getFullSlug,
  normalizeRelativeURLs,
  sluggify,
} from '../../util/path'
import { decodeStackedNoteHash, hashStackedNoteSlug } from '../../util/stacked-notes'
import { Toast } from './toast'
import {
  cacheStackedNotePayload,
  Dag,
  DagNode,
  getCachedStackedNotePayload,
  NoteDocument,
  readStackedNotePayload,
  removeAllChildren,
  stackedNotePayloadUrl,
  StackedNotePayload,
  StackedNoteState,
  VirtualRange,
} from './util'
import { fetchCanonical, startViewTransition } from './util'

// adapted from `micromorph`
// https://github.com/natemoo-re/micromorph
const NODE_TYPE_ELEMENT = 1
let announcer = document.createElement('route-announcer')
const isElement = (target: EventTarget | null): target is Element =>
  (target as Node)?.nodeType === NODE_TYPE_ELEMENT
const isLocalUrl = (href: string) => {
  try {
    const url = new URL(href)
    if (window.location.origin === url.origin) return true
  } catch {}
  return false
}

const isSamePage = (url: URL): boolean => {
  const sameOrigin = url.origin === window.location.origin
  const samePath = url.pathname === window.location.pathname
  return sameOrigin && samePath
}

const getOpts = ({ target }: Event): { url: URL; scroll?: boolean } | undefined => {
  if (!isElement(target)) return
  if (target.attributes.getNamedItem('target')?.value === '_blank') return
  const a = target.closest('a')
  if (!a) return
  if ('routerIgnore' in a.dataset) return
  const { href } = a
  if (!isLocalUrl(href)) return
  return { url: new URL(href), scroll: 'routerNoscroll' in a.dataset ? false : undefined }
}

function notifyNav(url: FullSlug) {
  const event: CustomEventMap['nav'] = new CustomEvent('nav', { detail: { url } })
  document.dispatchEvent(event)
}

const cleanupFns: Set<(...args: any[]) => void> = new Set()
window.addCleanup = fn => cleanupFns.add(fn)

if (!window.quartzToast) {
  const toast = new Toast()
  window.quartzToast = toast

  const handleToast = (event: CustomEventMap['toast']) => {
    const detail = event.detail
    if (!detail?.message) return
    toast.show(detail.message, {
      durationMs: detail.durationMs,
      styles: detail.styles,
      containerId: detail.containerId,
      containerStyles: detail.containerStyles,
    })
  }

  document.addEventListener('toast', handleToast)
  document.addEventListener('prenav', () => toast.destroy())
}

let navProgressBar: HTMLDivElement | null = null
let navProgressStart: number | null = null
let navProgressReset: number | null = null

function getNavProgressBar() {
  if (navProgressBar && document.body.contains(navProgressBar)) {
    return navProgressBar
  }

  const existing = document.querySelector<HTMLDivElement>('.navigation-progress')
  if (existing) {
    navProgressBar = existing
    return navProgressBar
  }

  navProgressBar = document.createElement('div')
  navProgressBar.className = 'navigation-progress'
  document.body.appendChild(navProgressBar)
  return navProgressBar
}

function startLoading() {
  const loadingBar = getNavProgressBar()
  if (navProgressStart) {
    window.clearTimeout(navProgressStart)
  }
  if (navProgressReset) {
    window.clearTimeout(navProgressReset)
  }

  loadingBar.style.opacity = '1'
  loadingBar.style.width = '0'
  loadingBar.style.animation = 'none'
  loadingBar.style.backgroundPosition = '-100% 0'

  navProgressStart = window.setTimeout(() => {
    loadingBar.style.width = '100%'
    loadingBar.style.animation = 'navigation-progress-sweep 2.5s linear infinite'
  }, 10)
}

function stopLoading() {
  const loadingBar = getNavProgressBar()
  if (navProgressStart) {
    window.clearTimeout(navProgressStart)
  }
  loadingBar.style.animation = 'none'
  loadingBar.style.opacity = '0'
  navProgressReset = window.setTimeout(() => {
    loadingBar.style.width = '0'
    loadingBar.style.backgroundPosition = '-100% 0'
  }, 300)
}

let p: DOMParser
const STACKED_OVERSCAN = 2

class StackedNoteManager {
  private dag: Dag = new Dag()
  private documentCache: Map<string, NoteDocument> = new Map()
  private inflight: Map<string, Promise<NoteDocument>> = new Map()

  container!: HTMLElement
  column!: HTMLElement
  main!: HTMLElement

  private styled!: CSSStyleDeclaration

  private events: AbortController | null = null
  private layoutFrame: number | null = null
  private pendingTailScroll = false

  private isActive: boolean = false

  constructor() {
    this.ensureElements()
    queueMicrotask(() => {
      void this.hydrateInitialStack()
    })
  }

  private mobile() {
    return window.innerWidth <= 800
  }

  private ensureElements() {
    const container = document.getElementById('stacked-notes-container')
    const main = container?.querySelector<HTMLElement>('#stacked-notes-main')
    const column = main?.querySelector<HTMLElement>('.stacked-notes-column')
    if (!container || !main || !column) {
      throw new Error('stacked notes container not found')
    }
    if (this.container === container && this.main === main && this.column === column) return
    this.events?.abort()
    this.container = container
    this.main = main
    this.column = column
    this.styled = getComputedStyle(this.main)
    this.setupHandlers()
  }

  private setupHandlers() {
    const events = new AbortController()
    this.events = events
    const signal = events.signal
    this.main.addEventListener('scroll', () => this.scheduleLayout(), { signal })
    window.addEventListener('resize', () => this.scheduleLayout(), { signal })
    this.column.addEventListener('click', event => this.onStackClick(event), { signal })
    this.column.addEventListener('mouseover', event => this.onStackHover(event, true), { signal })
    this.column.addEventListener('mouseout', event => this.onStackHover(event, false), { signal })
    this.column.addEventListener('keydown', event => this.onStackKey(event, true), { signal })
    this.column.addEventListener('keyup', event => this.onStackKey(event, false), { signal })
  }

  private dimensions() {
    return {
      contentWidth: parseInt(this.styled.getPropertyValue('--note-content-width')) || 620,
      titleWidth: parseInt(this.styled.getPropertyValue('--note-title-width')) || 40,
    }
  }

  private motion(): ScrollBehavior {
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth'
  }

  private notifyContentMounted(article: HTMLElement, content: HTMLElement, slug: string) {
    const event: CustomEventMap['contentdecrypted'] = new CustomEvent('contentdecrypted', {
      detail: { article, content, slug },
    })
    document.dispatchEvent(event)
  }

  private closestStackLink(target: EventTarget | null): HTMLAnchorElement | null {
    if (!isElement(target)) return null
    const link = target.closest('a.internal')
    if (!(link instanceof HTMLAnchorElement)) return null
    if (!this.column.contains(link)) return null
    if ('routerIgnore' in link.dataset) return null
    return link
  }

  private onStackClick(event: MouseEvent) {
    if (event.button !== 0) return
    const target = event.target
    if (isElement(target)) {
      const retry = target.closest('[data-stacked-retry]')
      if (retry && this.column.contains(retry)) {
        event.preventDefault()
        event.stopPropagation()
        const shell = retry.closest<HTMLElement>('.stacked-note')
        const slug = shell?.dataset.slug
        if (!slug) return
        this.documentCache.delete(slug)
        void this.loadAndApply(new URL(`/${slug}`, window.location.toString()), slug)
        return
      }
      const title = target.closest('.stacked-title')
      if (title && this.column.contains(title)) {
        event.preventDefault()
        event.stopPropagation()
        const shell = title.closest<HTMLElement>('.stacked-note')
        if (shell) this.scrollToShell(shell)
        return
      }
    }
    const link = this.closestStackLink(target)
    if (!link || event.ctrlKey || event.metaKey || event.shiftKey) return
    event.preventDefault()
    event.stopPropagation()
    const href = new URL(link.href, window.location.toString())
    if (event.altKey) {
      void this.addToTail(href, link)
      return
    }
    void this.add(href, link)
  }

  private onStackHover(event: MouseEvent, active: boolean) {
    const link = this.closestStackLink(event.target)
    const slug = link?.dataset.slug
    if (!slug) return
    const node = this.dag.get(slug)
    if (!node) return
    node.mounted.titleRail.classList.toggle('dag', active)
    node.mounted.bodyHost.querySelector<HTMLElement>('h1')?.classList.toggle('dag', active)
  }

  private onStackKey(event: KeyboardEvent, active: boolean) {
    const link = this.closestStackLink(event.target)
    if (!link) return
    if (active && event.altKey && !link.title) {
      link.title = 'pour ajouter à la fin de la pile'
    } else if (!active && link.title === 'pour ajouter à la fin de la pile') {
      link.title = ''
    }
  }

  private async hydrateInitialStack() {
    this.ensureElements()
    const url = new URL(window.location.toString())
    if (!this.container.classList.contains('active') && !url.searchParams.has('stackedNotes'))
      return
    this.isActive = true
    await this.initFromParams()
    this.updateURL()
    this.render({ scrollToTail: url.searchParams.getAll('stackedNotes').length > 1 })
  }

  private async initFromParams() {
    this.ensureElements()
    const url = new URL(window.location.toString())
    const stackedNotes = this.dedupe(url.searchParams.getAll('stackedNotes'))

    if (stackedNotes.length === 0) return

    const existingNotes = [...this.column.querySelectorAll('.stacked-note')] as HTMLElement[]

    if (existingNotes.length > 0) {
      for (const [index, noteElement] of existingNotes.entries()) {
        const slug = noteElement.dataset.slug
        if (!slug || this.dag.has(slug)) continue
        const noteHash = stackedNotes[index]
        const decodedSlug = this.decodeHash(noteHash)

        if (slug !== decodedSlug) {
          console.warn(`slug mismatch at index ${index}: ${slug} vs ${decodedSlug}`)
          continue
        }

        const noteTitle = noteElement.querySelector<HTMLElement>('.stacked-title')
        const noteContent = noteElement.querySelector<HTMLElement>('.stacked-content')
        const title = noteTitle?.textContent || slug
        const state = noteElement.dataset.state
        let noteDocument: NoteDocument
        if (state === 'pending') {
          noteDocument = this.pendingDocument(slug, new URL(`/${slug}`, window.location.toString()))
          noteDocument.title = title
        } else {
          const noteState: StackedNoteState =
            state === 'failed' || state === 'protected' || state === 'ready' ? state : 'ready'
          noteDocument = {
            slug,
            title,
            hash: '',
            bodyHtml: noteContent?.innerHTML ?? '',
            metadataHtml: '',
            state: noteState,
          }
        }
        const bodyHost = noteContent ?? window.document.createElement('div')
        const titleRail = noteTitle ?? window.document.createElement('div')
        bodyHost.classList.add('stacked-content')
        titleRail.classList.add('stacked-title')
        if (!noteContent) noteElement.prepend(bodyHost)
        if (!noteTitle) noteElement.append(titleRail)
        if (noteDocument.state === 'pending') {
          removeAllChildren(bodyHost)
          bodyHost.dataset.virtualized = 'unmounted'
        } else {
          bodyHost.dataset.virtualized = 'mounted'
        }
        noteElement.classList.add(noteDocument.state)
        noteElement.dataset.state = noteDocument.state
        transformHostInternalLinks(noteElement)
        this.documentCache.set(slug, noteDocument)
        this.dag.addNode({
          slug,
          title,
          document: noteDocument,
          anchor: null,
          mounted: {
            shell: noteElement,
            bodyHost,
            titleRail,
            mounted: noteDocument.state !== 'pending',
          },
        })

        if (noteDocument.state !== 'pending') {
          this.notifyContentMounted(noteElement, bodyHost, slug)
        }

        notifyNav(slug as FullSlug)
      }

      this.updateURL()
      return
    }

    stackedNotes.forEach(noteHash => {
      const slug = this.decodeHash(noteHash)
      if (!slug) return

      const href = new URL(`/${slug}`, window.location.toString())

      if (this.dag.has(slug)) {
        notifyNav(href.pathname as FullSlug)
        return
      }

      this.addPendingNode(slug, href, null)
    })
  }

  private dedupe(values: string[]): string[] {
    const seen = new Set<string>()
    const result: string[] = []
    for (const value of values) {
      if (seen.has(value)) continue
      seen.add(value)
      result.push(value)
    }
    return result
  }

  private updateURL() {
    this.ensureElements()
    const url = new URL(window.location.toString())

    url.searchParams.delete('stackedNotes')

    this.dag.getOrderedNodes().forEach((node, index, nodes) => {
      url.searchParams.append('stackedNotes', this.hashSlug(node.slug))
      this.applyGeometry(node, index, nodes.length)
    })

    window.history.replaceState({}, '', url)
    this.dag.getOrderedNodes().forEach(node => this.updateDagClasses(node.mounted.bodyHost))
  }

  getChain() {
    return this.dag
      .getOrderedNodes()
      .map(el => `stackedNotes=${this.hashSlug(el.slug)}`)
      .join('&')
  }

  private generateHash(slug: string): string {
    return hashStackedNoteSlug(slug)
  }

  private decodeHash(hash: string): string {
    return decodeStackedNoteHash(hash) ?? ''
  }

  private hashes: Map<string, string> = new Map()
  private slugs: Map<string, string> = new Map()

  private hashSlug(slug: string): string {
    if (this.slugs.has(slug)) {
      return this.slugs.get(slug)!
    }

    const hash = this.generateHash(slug)
    this.hashes.set(hash, slug)
    this.slugs.set(slug, hash)
    return hash
  }

  private payloadDocument(payload: StackedNotePayload, target: URL, slug: string, hash: string) {
    p = p || new DOMParser()
    const html = p.parseFromString(payload.content, 'text/html')
    normalizeRelativeURLs(html, target)
    transformHostInternalLinks(html)

    const elements = Array.from(html.body.children).flatMap(el =>
      el instanceof HTMLElement ? [el] : [],
    )

    return {
      slug,
      title: payload.title,
      hash,
      bodyHtml: this.serializeElements(elements),
      metadataHtml: payload.metadata ?? '',
      state: payload.state,
    }
  }

  private async fetchContent(url: URL, slug: string): Promise<NoteDocument> {
    p = p || new DOMParser()

    const target = new URL(url.toString())
    const hash = decodeURIComponent(target.hash)
    target.hash = ''
    target.search = ''

    const cachedPayload = getCachedStackedNotePayload(slug)
    if (cachedPayload) return this.payloadDocument(cachedPayload, target, slug, hash)

    const response = await fetch(stackedNotePayloadUrl(slug), {
      headers: { Accept: 'application/json' },
    }).catch(error => {
      console.error(error)
      return null
    })
    if (!response || !response.ok) return this.failedDocument(slug, hash)

    const json = await response.json().catch(error => {
      console.error(error)
      return null
    })
    const payload = readStackedNotePayload(json)
    if (!payload) return this.failedDocument(slug, hash)
    cacheStackedNotePayload(payload)

    return this.payloadDocument(payload, target, slug, hash)
  }

  private serializeElements(elements: HTMLElement[]): string {
    return elements.map(el => el.outerHTML).join('\n')
  }

  private failedDocument(slug: string, hash?: string, title?: string): NoteDocument {
    return {
      slug,
      title: title || slug,
      hash,
      bodyHtml: `<div class="stacked-note-status stacked-note-status-failed"><p>Impossible de charger cette note.</p><button type="button" data-stacked-retry>retry</button></div>`,
      metadataHtml: '',
      state: 'failed',
    }
  }

  private pendingDocument(slug: string, href: URL): NoteDocument {
    return {
      slug,
      title: slug,
      hash: decodeURIComponent(href.hash),
      bodyHtml: `<div class="stacked-note-status" role="status">chargement...</div>`,
      metadataHtml: '',
      state: 'pending',
    }
  }

  private createMountedNote(noteDocument: NoteDocument) {
    const note = window.document.createElement('div')
    note.className = 'stacked-note'
    note.id = this.hashSlug(noteDocument.slug)
    note.dataset.slug = noteDocument.slug
    note.dataset.entering = 'true'

    const noteContent = window.document.createElement('div')
    noteContent.className = 'stacked-content'
    noteContent.dataset.virtualized = 'unmounted'

    const noteTitle = window.document.createElement('div')
    noteTitle.classList.add('stacked-title')
    noteTitle.textContent = noteDocument.title

    note.append(noteContent, noteTitle)
    return { shell: note, bodyHost: noteContent, titleRail: noteTitle, mounted: false }
  }

  private addPendingNode(slug: string, href: URL, anchor: HTMLElement | null): DagNode {
    const cached = this.documentCache.get(slug)
    const noteDocument = cached ?? this.pendingDocument(slug, href)
    const mounted = this.createMountedNote(noteDocument)
    return this.dag.addNode({
      slug,
      title: noteDocument.title,
      document: noteDocument,
      anchor,
      mounted,
    })
  }

  private applyGeometry(node: DagNode, index: number, total: number) {
    const { contentWidth, titleWidth } = this.dimensions()
    const right = contentWidth - titleWidth
    const shell = node.mounted.shell
    shell.style.left = `${index * titleWidth}px`
    shell.style.right = `${-right + (total - index - 1) * titleWidth}px`
  }

  private syncShell(node: DagNode, index: number, total: number) {
    const { shell, titleRail } = node.mounted
    shell.id = this.hashSlug(node.slug)
    shell.dataset.slug = node.slug
    shell.dataset.state = node.document.state
    shell.classList.toggle('pending', node.document.state === 'pending')
    shell.classList.toggle('ready', node.document.state === 'ready')
    shell.classList.toggle('protected', node.document.state === 'protected')
    shell.classList.toggle('failed', node.document.state === 'failed')
    if (shell.dataset.entering === 'true') {
      requestAnimationFrame(() => {
        delete shell.dataset.entering
      })
    }
    titleRail.textContent = node.document.title
    this.applyGeometry(node, index, total)
  }

  private mountBody(node: DagNode) {
    if (node.mounted.mounted) return
    const { bodyHost } = node.mounted
    bodyHost.innerHTML = `${node.document.bodyHtml}${node.document.metadataHtml ?? ''}`
    bodyHost.dataset.virtualized = 'mounted'
    transformHostInternalLinks(bodyHost)
    this.updateDagClasses(bodyHost)
    node.mounted.mounted = true
    this.notifyContentMounted(node.mounted.shell, bodyHost, node.slug)
    this.scrollToHash(node)
  }

  private unmountBody(node: DagNode) {
    if (!node.mounted.mounted) return
    removeAllChildren(node.mounted.bodyHost)
    node.mounted.bodyHost.dataset.virtualized = 'unmounted'
    node.mounted.mounted = false
  }

  private updateDagClasses(root: HTMLElement) {
    root.querySelectorAll<HTMLAnchorElement>('a.internal').forEach(link => {
      const slug = link.dataset.slug
      link.classList.toggle('dag', slug !== undefined && this.dag.has(slug))
    })
  }

  private scrollToHash(node: DagNode) {
    const hash = node.document.hash
    if (!hash) return
    requestAnimationFrame(() => {
      const heading = node.mounted.bodyHost.querySelector<HTMLElement>(hash)
      if (heading) {
        node.mounted.shell.scroll({ top: heading.offsetTop - 12, behavior: this.motion() })
      }
    })
  }

  private render({ scrollToTail = true }: { scrollToTail?: boolean } = {}) {
    this.ensureElements()
    const nodes = this.dag.getOrderedNodes()
    const present = new Set(nodes.map(node => node.mounted.shell))
    const { contentWidth } = this.dimensions()

    for (const child of Array.from(this.column.children)) {
      if (child instanceof HTMLElement && !present.has(child)) {
        child.remove()
      }
    }

    nodes.forEach((node, index) => {
      this.syncShell(node, index, nodes.length)
      const current = this.column.children[index]
      if (current !== node.mounted.shell) {
        this.column.insertBefore(node.mounted.shell, current ?? null)
      }
    })

    this.column.style.width = `${nodes.length * contentWidth}px`
    this.container.classList.toggle('active', this.isActive)
    document.body.classList.toggle('stack-mode', this.isActive)
    this.updateURL()
    this.scheduleLayout(scrollToTail)
  }

  private scheduleLayout(scrollToTail = false) {
    this.pendingTailScroll = this.pendingTailScroll || scrollToTail
    if (this.layoutFrame !== null) return
    this.layoutFrame = requestAnimationFrame(() => {
      this.layoutFrame = null
      const shouldScroll = this.pendingTailScroll
      this.pendingTailScroll = false
      this.layout(shouldScroll)
    })
  }

  private layout(scrollToTail: boolean) {
    this.ensureElements()
    const nodes = this.dag.getOrderedNodes()
    if (nodes.length === 0) return
    if (scrollToTail) this.scrollToIndex(nodes.length - 1, 'instant')
    this.updateStackState(nodes)
    const range = this.virtualRange(nodes)
    const tailIndex = nodes.length - 1

    nodes.forEach((node, index) => {
      const visible = index >= range.first && index <= range.last
      const folded = node.mounted.shell.classList.contains('collapsed')
      const shouldMount = index === tailIndex || (visible && !folded)
      if (shouldMount) {
        this.mountBody(node)
      } else {
        this.unmountBody(node)
      }
    })
    this.loadVisiblePendingBodies(nodes, range)
  }

  private updateStackState(nodes: DagNode[]) {
    nodes.forEach(node => node.mounted.shell.classList.remove('overlay'))
    if (this.mobile()) return
    const { titleWidth } = this.dimensions()
    const clientWidth = document.documentElement.clientWidth

    nodes.forEach((node, index) => {
      const shell = node.mounted.shell
      const rect = shell.getBoundingClientRect()
      let shouldCollapse = false

      if (index === nodes.length - 1) {
        shouldCollapse = clientWidth - rect.left <= 50
      } else {
        const nextShell = nodes[index + 1]?.mounted.shell
        if (!nextShell) return
        const nextRect = nextShell.getBoundingClientRect()
        nextShell.classList.toggle('overlay', nextRect.left < rect.right)
        shouldCollapse = nextRect.left <= rect.left + titleWidth
      }

      if (shouldCollapse && !shell.classList.contains('collapsed')) {
        shell.scrollTo({ top: 0 })
      }
      shell.classList.toggle('collapsed', shouldCollapse)
      shell.classList.toggle('hidden-note', shouldCollapse && index !== nodes.length - 1)
    })
  }

  private virtualRange(nodes: DagNode[]): VirtualRange {
    if (this.mobile()) {
      const tail = Math.max(0, nodes.length - 1)
      return { first: tail, last: tail }
    }
    const { contentWidth } = this.dimensions()
    const visibleFirst = Math.max(0, Math.floor(this.main.scrollLeft / contentWidth))
    const visibleLast = Math.ceil((this.main.scrollLeft + this.main.clientWidth) / contentWidth)
    const first = Math.max(0, visibleFirst - STACKED_OVERSCAN)
    const last = Math.min(nodes.length - 1, visibleLast + STACKED_OVERSCAN)
    return { first, last }
  }

  private loadVisiblePendingBodies(nodes: DagNode[], range: VirtualRange) {
    const tailIndex = nodes.length - 1
    nodes.forEach((node, index) => {
      if (node.document.state !== 'pending') return
      const visible = index >= range.first && index <= range.last
      const folded = node.mounted.shell.classList.contains('collapsed')
      if (index !== tailIndex && (!visible || folded)) return
      void this.loadAndApply(new URL(`/${node.slug}`, window.location.toString()), node.slug)
    })
  }

  private scrollToShell(shell: HTMLElement) {
    const index = this.dag.getOrderedNodes().findIndex(node => node.mounted.shell === shell)
    if (index >= 0) this.scrollToIndex(index)
  }

  private scrollToIndex(index: number, behavior: ScrollBehavior = this.motion()) {
    const { contentWidth } = this.dimensions()
    const maxLeft = Math.max(0, this.column.scrollWidth - this.main.clientWidth)
    const tail = this.dag.getOrderedNodes().length - 1
    const left = index === tail ? maxLeft : Math.min(index * contentWidth, maxLeft)
    this.main.scrollTo({ left, behavior })
  }

  private highlightNode(node: DagNode) {
    node.mounted.shell.classList.add('highlights')
    window.setTimeout(() => node.mounted.shell.classList.remove('highlights'), 500)
  }

  private async loadDocument(href: URL, slug: string): Promise<NoteDocument> {
    const cached = this.documentCache.get(slug)
    if (cached && cached.state !== 'pending') return cached

    const pending = this.inflight.get(slug)
    if (pending) return pending

    const promise = this.fetchContent(href, slug)
      .catch(error => {
        console.error(`Failed to fetch stacked note ${slug}:`, error)
        return this.failedDocument(slug, decodeURIComponent(href.hash))
      })
      .then(noteDocument => {
        this.documentCache.set(slug, noteDocument)
        this.inflight.delete(slug)
        return noteDocument
      })

    this.inflight.set(slug, promise)
    return promise
  }

  private async loadAndApply(href: URL, slug: string): Promise<boolean> {
    const noteDocument = await this.loadDocument(href, slug)
    const node = this.dag.get(slug)
    if (!node) return false
    node.document = noteDocument
    node.title = noteDocument.title
    node.mounted.mounted = false
    this.render({ scrollToTail: false })
    return noteDocument.state !== 'failed'
  }

  private async addToTail(href: URL, anchor?: HTMLElement) {
    return this.addInternal(href, anchor, true)
  }

  async add(href: URL, anchor?: HTMLElement) {
    return this.addInternal(href, anchor, false)
  }

  private async addInternal(href: URL, anchor: HTMLElement | undefined, append: boolean) {
    this.ensureElements()
    let slug = this.getSlug(href)

    if (href.pathname === '/') {
      if (slug === '') {
        slug = 'index' as FullSlug
      } else {
        slug = `${slug}/index` as FullSlug
      }
    }

    anchor?.classList.add('dag')

    if (this.dag.has(slug)) {
      const node = this.dag.get(slug)
      if (!node) return false
      this.scrollToShell(node.mounted.shell)
      this.highlightNode(node)
      notifyNav(slug)
      return true
    }

    const clickedSlug = anchor?.closest<HTMLElement>('.stacked-note')?.dataset.slug

    if (!append && clickedSlug && this.dag.has(clickedSlug)) {
      this.dag.truncateAfter(clickedSlug)
    }

    this.addPendingNode(slug, href, anchor ?? null)
    this.isActive = true
    this.render({ scrollToTail: true })
    notifyNav(this.getSlug(href))
    return this.loadAndApply(href, slug)
  }

  async open() {
    this.ensureElements()
    if (this.isActive) return true
    const contents = Array.from(document.getElementsByClassName('popover-hint')).flatMap(el =>
      el instanceof HTMLElement ? [el.cloneNode(true) as HTMLElement] : [],
    )
    const h1 = document.querySelector<HTMLElement>('h1')
    const slug = getFullSlug(window)
    const title =
      h1?.innerText ?? h1?.textContent ?? document.querySelector('title')?.textContent ?? slug
    const hash = decodeURIComponent(window.location.hash)
    window.location.hash = ''
    const noteDocument = {
      slug,
      title,
      hash,
      bodyHtml: this.serializeElements(contents),
      metadataHtml: '',
      state: 'ready' as const,
    }
    this.documentCache.set(slug, noteDocument)
    this.dag.addNode({
      slug,
      title,
      document: noteDocument,
      anchor: null,
      mounted: this.createMountedNote(noteDocument),
    })

    this.isActive = true
    await this.initFromParams()
    this.updateURL()
    this.render({ scrollToTail: true })
    notifyNav(getFullSlug(window))

    return true
  }

  destroy() {
    this.isActive = false

    this.dag.clear()
    this.inflight.clear()
    removeAllChildren(this.column)

    const url = new URL(window.location.toString())
    url.searchParams.delete('stackedNotes')
    window.history.replaceState({}, '', url)
    document.body.classList.remove('stack-mode')

    cleanupFns.forEach(fn => fn())
    cleanupFns.clear()
  }

  async navigate(url: URL) {
    try {
      if (!this.active) {
        await this.open()
        if (this.getSlug(url) === getFullSlug(window)) return true
      }

      const event: CustomEventMap['prenav'] = new CustomEvent('prenav', { detail: {} })
      document.dispatchEvent(event)

      return await this.add(url)
    } catch (e) {
      console.error(`Failed to navigate to ${url}: ${e}`)
      return false
    }
  }

  private getSlug(url: URL): FullSlug {
    return url.pathname.slice(1) as FullSlug
  }

  get active() {
    return this.isActive
  }
}

const STREAM_HOSTNAME = 'stream.aarnphm.xyz'
const NOTES_HOSTNAME = 'notes.aarnphm.xyz'
const APEX_ORIGIN = 'https://aarnphm.xyz'

function normalizeStreamPath(raw: string): string | null {
  if (!raw) return null
  const trimmed = raw.trim()
  if (trimmed === '') return '/'
  if (trimmed.startsWith('#')) return null

  try {
    const parsed = new URL(trimmed, `https://${STREAM_HOSTNAME}`)
    const hash = parsed.hash ?? ''
    const search = parsed.search ?? ''
    let pathname = parsed.pathname.replace(/^\/+/, '')
    if (pathname === '') return `/${search}${hash}`
    pathname = sluggify(pathname)
    const normalizedPath = `/${pathname}`
    return `${normalizedPath}${search}${hash}`
  } catch {
    return null
  }
}

function transformHostInternalLinks(root: Document | Element) {
  if (typeof window === 'undefined') return
  const host = window.location.hostname

  const anchors = root.querySelectorAll<HTMLAnchorElement>('a.internal')
  anchors.forEach(anchor => {
    if (host === STREAM_HOSTNAME) {
      const preferred = anchor.dataset.slug ?? anchor.getAttribute('href') ?? ''
      const normalizedPath = normalizeStreamPath(preferred)
      if (!normalizedPath) return

      const absoluteHref = `${APEX_ORIGIN}${
        normalizedPath.startsWith('/') ? normalizedPath : `/${normalizedPath}`
      }`
      if (anchor.getAttribute('href') !== absoluteHref) {
        anchor.setAttribute('href', absoluteHref)
      }
      anchor.dataset.noPopover = 'true'
      anchor.dataset.routerIgnore = 'true'
    } else if (host === NOTES_HOSTNAME) {
      anchor.dataset.noPopover = 'true'
    }
  })

  if (host === STREAM_HOSTNAME) {
    const transcludeRefs = root.querySelectorAll<HTMLElement>('.transclude-ref[data-href]')
    transcludeRefs.forEach(ref => {
      const preferred = ref.dataset.slug ?? ref.dataset.href ?? ''
      const normalizedPath = normalizeStreamPath(preferred)
      if (!normalizedPath) return
      const absoluteHref = `${APEX_ORIGIN}${
        normalizedPath.startsWith('/') ? normalizedPath : `/${normalizedPath}`
      }`
      ref.dataset.href = absoluteHref
      ref.dataset.routerIgnore = 'true'
    })
  }
}

const stacked = new StackedNoteManager()
window.stacked = stacked

async function navigate(url: URL, isBack: boolean = false) {
  const stackedContainer = document.getElementById('stacked-notes-container')
  if (stackedContainer?.classList.contains('active')) {
    return await window.stacked.navigate(url)
  }

  startLoading()

  p = p || new DOMParser()
  const contents = await fetchCanonical(new URL(`${url}`))
    .then(res => {
      const contentType = res.headers.get('content-type')
      if (contentType?.startsWith('text/html')) {
        return res.text()
      } else {
        window.location.assign(url)
      }
    })
    .catch(() => {
      window.location.assign(url)
    })

  if (!contents) return

  // notify about to nav
  const event: CustomEventMap['prenav'] = new CustomEvent('prenav', { detail: {} })
  document.dispatchEvent(event)

  // cleanup old
  cleanupFns.forEach(fn => fn())
  cleanupFns.clear()

  const html = p.parseFromString(contents, 'text/html')
  normalizeRelativeURLs(html, url)
  transformHostInternalLinks(html)

  let title = html.querySelector('title')?.textContent
  if (title) {
    document.title = title
  } else {
    const h1 = document.querySelector('h1')
    title = h1?.innerText ?? h1?.textContent ?? url.pathname
  }
  if (announcer.textContent !== title) {
    announcer.textContent = title
  }
  announcer.dataset.persist = ''
  html.body.appendChild(announcer)

  // morph body
  startViewTransition(() => {
    micromorph(document.body, html.body)
    transformHostInternalLinks(document)

    // scroll into place and add history
    if (!isBack) {
      if (url.hash) {
        const el = document.getElementById(decodeURIComponent(url.hash.substring(1)))
        el?.scrollIntoView()
      } else {
        window.scrollTo({ top: 0 })
      }
    }

    // now, patch head, re-executing scripts
    const elementsToRemove = document.head.querySelectorAll(':not([data-persist])')
    elementsToRemove.forEach(el => el.remove())
    const elementsToAdd = html.head.querySelectorAll(':not([data-persist])')
    elementsToAdd.forEach(el => document.head.appendChild(el))

    // delay setting the url until now
    // at this point everything is loaded so changing the url should resolve to the correct addresses
    if (!isBack) {
      history.pushState({}, '', url)
    }
    notifyNav(getFullSlug(window))
    delete announcer.dataset.persist
  })
  stopLoading()
}

window.spaNavigate = navigate
window.notifyNav = notifyNav

document.addEventListener('nav', () => {
  transformHostInternalLinks(document)
})

function createRouter() {
  if (typeof window !== 'undefined') {
    window.addEventListener('click', async event => {
      if (event.defaultPrevented) return
      const { url } = getOpts(event) ?? {}
      // dont hijack behaviour, just let browser act normally
      if (!url || event.ctrlKey || event.metaKey || event.altKey) return
      event.preventDefault()

      if (isSamePage(url) && url.hash) {
        const el = document.getElementById(decodeURIComponent(url.hash.substring(1)))
        el?.scrollIntoView()
        history.pushState({}, '', url)
        return
      }

      try {
        navigate(url, false)
      } catch (e) {
        console.error(e)
        window.location.assign(url)
      }
    })

    window.addEventListener('popstate', event => {
      const { url } = getOpts(event) ?? {}
      if (window.location.hash && window.location.pathname === url?.pathname) return
      try {
        navigate(new URL(window.location.toString()), true)
      } catch (e) {
        window.location.reload()
        console.error(e)
      }
      return
    })
  }

  return new (class Router {
    go(pathname: RelativeURL) {
      const url = new URL(pathname, window.location.toString())
      return navigate(url, false)
    }

    back() {
      return window.history.back()
    }

    forward() {
      return window.history.forward()
    }
  })()
}

createRouter()
notifyNav(getFullSlug(window))

if (!customElements.get('route-announcer')) {
  const attrs = {
    'aria-live': 'assertive',
    'aria-atomic': 'true',
    style:
      'position: absolute; left: 0; top: 0; clip: rect(0 0 0 0); clip-path: inset(50%); overflow: hidden; white-space: nowrap; width: 1px; height: 1px',
  }

  customElements.define(
    'route-announcer',
    class RouteAnnouncer extends HTMLElement {
      constructor() {
        super()
      }
      connectedCallback() {
        for (const [key, value] of Object.entries(attrs)) {
          this.setAttribute(key, value)
        }
      }
    },
  )
}

// NOTE: navigate first if there are stackedNotes
const baseUrl = new URL(document.location.toString())
const stackedNotes = baseUrl.searchParams.get('stackedNotes')

// remove elements on notes.aarnphm.xyz
if (window.location.host === 'notes.aarnphm.xyz') {
  if (!stackedNotes || stackedNotes.length === 0) {
    const slug = 'notes'
    baseUrl.searchParams.set('stackedNotes', btoa(slug.toString()).replace(/=+$/, ''))
    baseUrl.pathname = `/${slug}`

    window.stacked.navigate(baseUrl).then(data => {
      if (!data) return
      document
        .querySelectorAll(
          'main > section[class~="page-footer"], main > section[class~="page-header"], main > section[class~="page-content"], nav.breadcrumb-container, header > .keybind, header > .search, header > .graph',
        )
        .forEach(el => el.remove())
    })
  }
}
