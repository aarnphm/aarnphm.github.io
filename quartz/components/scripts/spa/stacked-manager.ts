import { getFullSlug } from '../../../util/path'
import {
  clearStackedNotePayloadCache,
  Dag,
  NOTES_HOSTNAME,
  stackedNoteRootSlug,
  stackedNoteSlugs,
  stackedNotesUrl,
  type DagNode,
  type VirtualRange,
} from '../../../util/stacked-notes'
import { ensureExternalLinkTarget, isElement, isLocalHttpUrl, linkUrl } from './link-policy'
import { StackedDocumentStore } from './stacked-documents'
import { StackedNoteView } from './stacked-view'

class StackedNoteManager {
  private dag: Dag = new Dag()
  private documents = new StackedDocumentStore(slug => this.dag.has(slug))
  private view = new StackedNoteView({
    nodes: () => this.dag.getOrderedNodes(),
    has: slug => this.dag.has(slug),
    active: () => this.isActive,
    onClick: event => this.onStackClick(event),
    onHover: (event, active) => this.onStackHover(event, active),
    onKey: (event, active) => this.onStackKey(event, active),
    onVisible: (nodes, range) => this.loadVisiblePendingBodies(nodes, range),
  })

  private isActive: boolean = false

  get container(): HTMLElement {
    return this.view.container
  }

  get column(): HTMLElement {
    return this.view.column
  }

  get main(): HTMLElement {
    return this.view.main
  }

  private closestStackLink(target: EventTarget | null): HTMLAnchorElement | null {
    if (!isElement(target)) return null
    const link = target.closest('a.internal')
    if (!(link instanceof HTMLAnchorElement)) return null
    if (!this.view.column.contains(link)) return null
    if (
      'routerIgnore' in link.dataset ||
      link.hasAttribute('download') ||
      (link.target && link.target !== '_self')
    )
      return null
    const url = linkUrl(link)
    if (!url) return null
    if (ensureExternalLinkTarget(link, url)) return null
    if (!isLocalHttpUrl(url)) return null
    if (!link.dataset.slug || link.dataset.role === 'anchor') return null
    return link
  }

  private onStackClick(event: MouseEvent) {
    if (event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey) return
    const target = event.target
    if (isElement(target)) {
      const retry = target.closest('[data-stacked-retry]')
      if (retry && this.view.column.contains(retry)) {
        event.preventDefault()
        event.stopPropagation()
        const shell = retry.closest<HTMLElement>('.stacked-note')
        const slug = shell?.dataset.slug
        if (!slug) return
        this.documents.invalidate(slug)
        void this.loadAndApply(new URL(`/${slug}`, window.location.toString()), slug)
        return
      }
      const title = target.closest('.stacked-title')
      const titleLink = target.closest('a')
      if (title && this.view.column.contains(title) && (!titleLink || !title.contains(titleLink))) {
        event.preventDefault()
        event.stopPropagation()
        const shell = title.closest<HTMLElement>('.stacked-note')
        if (shell) this.view.scrollToShell(shell)
        return
      }
      const anchor = target.closest('a')
      const shell = anchor?.closest<HTMLElement>('.stacked-note')
      if (
        anchor instanceof HTMLAnchorElement &&
        shell &&
        !event.altKey &&
        !anchor.hasAttribute('download') &&
        !('routerIgnore' in anchor.dataset) &&
        (!anchor.target || anchor.target === '_self')
      ) {
        const href = linkUrl(anchor)
        if (
          href &&
          isLocalHttpUrl(href) &&
          href.hash &&
          (anchor.getAttribute('href')?.startsWith('#') ||
            this.getSlug(href) === shell.dataset.slug)
        ) {
          event.preventDefault()
          event.stopPropagation()
          const node = shell.dataset.slug ? this.dag.get(shell.dataset.slug) : undefined
          if (node) {
            node.document.hash = href.hash
            this.view.scrollToHash(node)
          }
          return
        }
      }
    }
    const link = this.closestStackLink(target)
    if (!link) return
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

  restore(url: URL, force = false): boolean {
    this.view.ensureElements()
    const shouldOpen =
      force ||
      url.hostname === NOTES_HOSTNAME ||
      url.searchParams.has('stackedNotes') ||
      window.history.state?.stackedNotes === true
    if (!shouldOpen) return false

    const slugs = stackedNoteSlugs(url)
    const previous = new Map(this.dag.getOrderedNodes().map(node => [node.slug, node]))
    const serverNotes = new Map(
      Array.from(this.view.column.querySelectorAll<HTMLElement>('.stacked-note')).flatMap(shell =>
        shell.dataset.slug ? [[shell.dataset.slug, shell] as const] : [],
      ),
    )
    for (const node of previous.values()) {
      if (!slugs.includes(node.slug)) this.view.unmountBody(node)
    }
    this.dag.clear()

    for (const slug of slugs) {
      const existing = previous.get(slug)
      if (existing) {
        this.dag.addNode(existing)
        continue
      }
      const shell = serverNotes.get(slug)
      const content = shell?.querySelector<HTMLElement>('.stacked-content')
      const title = shell?.querySelector<HTMLElement>('.stacked-title')?.textContent || slug
      const state = shell?.dataset.state
      if (content && (state === 'ready' || state === 'protected' || state === 'failed')) {
        const hash = slug === slugs[0] ? url.hash : ''
        this.documents.captureMarkup(slug, title, content.innerHTML, state, url, hash)
      } else if (slug === getFullSlug(window)) {
        this.documents.captureCurrent(slug, url)
      }
      this.addPendingNode(slug, new URL(`/${slug}${slug === slugs[0] ? url.hash : ''}`, url), null)
    }
    this.isActive = true
    this.updateURL()
    this.view.render({ scrollToTail: slugs.length > 1 })
    return true
  }

  private updateURL(mode: 'push' | 'replace' = 'replace') {
    const url = stackedNotesUrl(
      new URL(window.location.href),
      this.dag.getOrderedNodes().map(node => node.slug),
    )
    const state = { ...window.history.state, stackedNotes: this.isActive }
    if (mode === 'push' && url.href !== window.location.href) {
      window.history.pushState(state, '', url)
    } else {
      window.history.replaceState(state, '', url)
    }
  }

  getChain() {
    const url = stackedNotesUrl(
      new URL(window.location.href),
      this.dag.getOrderedNodes().map(node => node.slug),
    )
    const params = new URLSearchParams()
    for (const value of url.searchParams.getAll('stackedNotes'))
      params.append('stackedNotes', value)
    return params.toString()
  }

  private addPendingNode(slug: string, href: URL, anchor: HTMLElement | null): DagNode {
    const cached = this.documents.get(slug)
    const noteDocument = cached ?? this.documents.pendingDocument(slug, href)
    const mounted = this.view.createMountedNote(noteDocument)
    return this.dag.addNode({
      slug,
      title: noteDocument.title,
      document: noteDocument,
      anchor,
      mounted,
    })
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

  private async loadAndApply(href: URL, slug: string): Promise<boolean> {
    const node = this.dag.get(slug)
    if (!node) return false
    const noteDocument = await this.documents.loadDocument(href, slug)
    if (this.dag.get(slug) !== node || !this.isActive) return false
    if (node.document === noteDocument) return noteDocument.state !== 'failed'
    this.view.unmountBody(node)
    node.document = noteDocument
    node.title = noteDocument.title
    node.mounted.mounted = false
    this.view.render({ scrollToTail: false })
    return noteDocument.state !== 'failed'
  }

  private async addToTail(href: URL, anchor?: HTMLElement) {
    return this.addInternal(href, anchor, true)
  }

  async add(href: URL, anchor?: HTMLElement) {
    return this.addInternal(href, anchor, false)
  }

  private async addInternal(href: URL, anchor: HTMLElement | undefined, append: boolean) {
    this.view.ensureElements()
    const slug = this.getSlug(href)

    anchor?.classList.add('dag')

    if (this.dag.has(slug)) {
      const node = this.dag.get(slug)
      if (!node) return false
      this.view.scrollToShell(node.mounted.shell)
      this.view.highlightNode(node)
      node.document.hash = href.hash
      this.view.scrollToHash(node)
      if (this.view.mobile() && this.dag.getTail() !== node) {
        this.truncateAfter(slug)
        this.updateURL('push')
        this.view.render()
      }
      return true
    }

    const clickedSlug = anchor?.closest<HTMLElement>('.stacked-note')?.dataset.slug

    if (!append && clickedSlug && this.dag.has(clickedSlug)) {
      this.truncateAfter(clickedSlug)
    }

    this.addPendingNode(slug, href, anchor ?? null)
    this.isActive = true
    this.updateURL('push')
    this.view.render({ scrollToTail: true })
    return this.loadAndApply(href, slug)
  }

  private truncateAfter(slug: string) {
    const nodes = this.dag.getOrderedNodes()
    const index = nodes.findIndex(node => node.slug === slug)
    for (const node of nodes.slice(index + 1)) {
      node.anchor = null
      this.view.unmountBody(node)
    }
    this.dag.truncateAfter(slug)
  }

  async open() {
    if (this.isActive) return true
    return this.restore(new URL(window.location.href), true)
  }

  release() {
    this.view.release()
    for (const node of this.dag.getOrderedNodes()) {
      node.anchor = null
      this.view.unmountBody(node)
    }
    this.dag.clear()
    this.documents.clear()
    clearStackedNotePayloadCache()
    this.isActive = false
  }

  destroy() {
    const title = this.dag.getOrderedNodes()[0]?.title
    this.release()
    this.view.column.replaceChildren()
    const url = new URL(window.location.href)
    url.searchParams.delete('stackedNotes')
    window.history.replaceState({ ...window.history.state, stackedNotes: false }, '', url)
    this.view.render({ scrollToTail: false })
    if (title) document.title = title
  }

  async navigate(url: URL) {
    try {
      if (!this.active) {
        await this.open()
        if (this.getSlug(url) === getFullSlug(window)) return true
      }

      if (url.searchParams.has('stackedNotes')) {
        window.history.pushState({ ...window.history.state, stackedNotes: true }, '', url)
        return this.restore(url)
      }
      return await this.add(url)
    } catch (e) {
      console.error(`Failed to navigate to ${url}: ${e}`)
      return false
    }
  }

  private getSlug(url: URL): string {
    return stackedNoteRootSlug(url) ?? 'index'
  }

  get active() {
    return this.isActive
  }
}

export const stacked = new StackedNoteManager()
