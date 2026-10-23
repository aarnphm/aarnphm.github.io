import {
  hashStackedNoteSlug,
  type DagNode,
  type NoteDocument,
  type VirtualRange,
} from '../../../util/stacked-notes'
import { loadPageStyles } from '../page-styles'
import { cleanupHydratedRoot } from '../root-lifecycle'
import { transformHostInternalLinks } from './link-policy'

export const announcer = document.createElement('route-announcer')

const STACKED_OVERSCAN = 2

interface StackedViewHost {
  nodes(): DagNode[]
  has(slug: string): boolean
  active(): boolean
  onClick(event: MouseEvent): void
  onHover(event: MouseEvent, active: boolean): void
  onKey(event: KeyboardEvent, active: boolean): void
  onVisible(nodes: DagNode[], range: VirtualRange): void
}

export class StackedNoteView {
  container!: HTMLElement
  column!: HTMLElement
  main!: HTMLElement

  private styled!: CSSStyleDeclaration

  private events: AbortController | null = null
  private layoutFrame: number | null = null
  private pendingTailScroll = false

  constructor(private host: StackedViewHost) {}

  mobile() {
    return window.innerWidth <= 800
  }

  ensureElements() {
    const container = document.getElementById('stacked-notes-container')
    const main = container?.querySelector<HTMLElement>('#stacked-notes-main')
    const column = main?.querySelector<HTMLElement>('.stacked-notes-column')
    if (!container || !main || !column) {
      throw new Error('stacked notes container not found')
    }
    if (this.container === container && this.main === main && this.column === column) {
      if (!this.events || this.events.signal.aborted) this.setupHandlers()
      return
    }
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
    this.column.addEventListener('click', event => this.host.onClick(event), { signal })
    this.column.addEventListener('mouseover', event => this.host.onHover(event, true), { signal })
    this.column.addEventListener('mouseout', event => this.host.onHover(event, false), { signal })
    this.column.addEventListener('keydown', event => this.host.onKey(event, true), { signal })
    this.column.addEventListener('keyup', event => this.host.onKey(event, false), { signal })
  }

  release() {
    this.events?.abort()
    this.events = null
    if (this.layoutFrame !== null) cancelAnimationFrame(this.layoutFrame)
    this.layoutFrame = null
    this.pendingTailScroll = false
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

  private hashSlug(slug: string): string {
    return hashStackedNoteSlug(slug)
  }

  createMountedNote(noteDocument: NoteDocument) {
    const note = window.document.createElement('div')
    note.className = 'stacked-note'
    note.id = this.hashSlug(noteDocument.slug)
    note.dataset.slug = noteDocument.slug
    note.dataset.entering = 'true'

    const noteContent = window.document.createElement('div')
    noteContent.className = 'stacked-content'
    noteContent.dataset.virtualized = 'unmounted'

    const noteTitle = window.document.createElement('button')
    noteTitle.type = 'button'
    noteTitle.tabIndex = -1
    noteTitle.classList.add('stacked-title')
    noteTitle.textContent = noteDocument.title

    note.append(noteContent, noteTitle)
    return {
      shell: note,
      bodyHost: noteContent,
      titleRail: noteTitle,
      mounted: false,
      scrollTop: 0,
    }
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
    titleRail.setAttribute('aria-label', `Afficher ${node.document.title}`)
    this.applyGeometry(node, index, total)
  }

  private mountBody(node: DagNode) {
    if (node.mounted.mounted) return
    const { bodyHost } = node.mounted
    bodyHost.innerHTML = node.document.bodyHtml
    bodyHost.dataset.virtualized = 'mounted'
    transformHostInternalLinks(bodyHost)
    this.updateDagClasses(bodyHost)
    node.mounted.mounted = true
    void loadPageStyles(bodyHost)
    this.notifyContentMounted(node.mounted.shell, bodyHost, node.slug)
    this.scrollToHash(node)
  }

  unmountBody(node: DagNode) {
    if (!node.mounted.mounted) return
    cleanupHydratedRoot(node.mounted.bodyHost)
    node.mounted.bodyHost.replaceChildren()
    node.mounted.bodyHost.dataset.virtualized = 'unmounted'
    node.mounted.mounted = false
  }

  private updateDagClasses(root: HTMLElement) {
    root.querySelectorAll<HTMLAnchorElement>('a.internal').forEach(link => {
      const slug = link.dataset.slug
      link.classList.toggle('dag', slug !== undefined && this.host.has(slug))
    })
  }

  scrollToHash(node: DagNode) {
    const hash = node.document.hash
    if (!hash) return
    let id: string
    try {
      id = decodeURIComponent(hash.slice(1))
    } catch {
      return
    }
    requestAnimationFrame(() => {
      const heading = Array.from(node.mounted.bodyHost.querySelectorAll<HTMLElement>('[id]')).find(
        element => element.id === id,
      )
      if (heading) {
        node.mounted.shell.scroll({ top: heading.offsetTop - 12, behavior: this.motion() })
      }
    })
  }

  render({ scrollToTail = true }: { scrollToTail?: boolean } = {}) {
    this.ensureElements()
    const nodes = this.host.nodes()
    const present = new Set(nodes.map(node => node.mounted.shell))
    const { contentWidth } = this.dimensions()
    const isActive = this.host.active()

    for (const child of Array.from(this.column.children)) {
      if (child instanceof HTMLElement && !present.has(child)) {
        cleanupHydratedRoot(child)
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
    this.container.classList.toggle('active', isActive)
    document.body.classList.toggle('stack-mode', isActive)
    document.querySelector('#stacked-note-toggle')?.setAttribute('aria-checked', String(isActive))
    document.querySelector('.header')?.classList.toggle('grid', isActive)
    document.querySelector('.header')?.classList.toggle('all-col', isActive)
    nodes.forEach(node => this.updateDagClasses(node.mounted.bodyHost))
    if (isActive && nodes.length > 0) {
      document.title = nodes.map(node => node.title).join(' | ')
      if (announcer.textContent !== document.title) announcer.textContent = document.title
      if (!announcer.isConnected) document.body.append(announcer)
    }
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
    const nodes = this.host.nodes()
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
      }
      // Keep hydrated contents until the note leaves the chain, so unfolding is immediate.
      node.mounted.bodyHost.inert = folded || !visible
    })
    this.host.onVisible(nodes, range)
  }

  private updateStackState(nodes: DagNode[]) {
    if (this.mobile()) {
      nodes.forEach(node => {
        node.mounted.shell.classList.remove('collapsed', 'hidden-note', 'overlay')
        node.mounted.titleRail.tabIndex = -1
      })
      return
    }
    const { titleWidth } = this.dimensions()
    const clientWidth = document.documentElement.clientWidth
    const rects = nodes.map(node => node.mounted.shell.getBoundingClientRect())

    nodes.forEach((node, index) => {
      const shell = node.mounted.shell
      const rect = rects[index]
      shell.classList.toggle('overlay', index > 0 && rect.left < rects[index - 1].right)
      let shouldCollapse = false

      if (index === nodes.length - 1) {
        shouldCollapse = clientWidth - rect.left <= 50
      } else {
        const nextRect = rects[index + 1]
        shouldCollapse = nextRect.left <= rect.left + titleWidth
      }

      if (shouldCollapse && !shell.classList.contains('collapsed')) {
        node.mounted.scrollTop = shell.scrollTop
        shell.scrollTo({ top: 0 })
      } else if (!shouldCollapse && shell.classList.contains('collapsed')) {
        shell.scrollTo({ top: node.mounted.scrollTop })
      }
      node.mounted.titleRail.tabIndex = shouldCollapse ? 0 : -1
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

  scrollToShell(shell: HTMLElement) {
    const index = this.host.nodes().findIndex(node => node.mounted.shell === shell)
    if (index >= 0) this.scrollToIndex(index)
  }

  private scrollToIndex(index: number, behavior: ScrollBehavior = this.motion()) {
    const { contentWidth } = this.dimensions()
    const maxLeft = Math.max(0, this.column.scrollWidth - this.main.clientWidth)
    const tail = this.host.nodes().length - 1
    const left = index === tail ? maxLeft : Math.min(index * contentWidth, maxLeft)
    this.main.scrollTo({ left, behavior })
  }

  highlightNode(node: DagNode) {
    node.mounted.shell.classList.add('highlights')
    window.setTimeout(() => node.mounted.shell.classList.remove('highlights'), 500)
  }
}
