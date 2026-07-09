import { getFullSlug, isFullSlug, resolveRelative, type FullSlug } from '../../util/path'
import { cleanupHydratedRoot } from './root-lifecycle'

const activeSessions = window.quartzSidePanelSessions ?? new WeakMap<HTMLElement, () => void>()
window.quartzSidePanelSessions = activeSessions
const pendingRequests =
  window.quartzSidePanelRequests ?? new WeakMap<HTMLElement, AbortController>()
window.quartzSidePanelRequests = pendingRequests

export interface SidePanelRequest {
  signal: AbortSignal
  mount: (slug: FullSlug, ...inner: HTMLElement[]) => HTMLDivElement | null
  cancel: () => void
}

function clearSidePanel(asidePanel: HTMLElement): void {
  pendingRequests.get(asidePanel)?.abort()
  pendingRequests.delete(asidePanel)
  cleanupHydratedRoot(asidePanel)
  asidePanel.replaceChildren()
  asidePanel.classList.remove('active')
  asidePanel.style.removeProperty('--sidepanel-top-offset')
}

export function disposeSidePanel(asidePanel?: HTMLElement | null): void {
  const panel =
    asidePanel ??
    document.querySelector<HTMLDivElement>("main > * > aside[class~='sidepanel-container']")
  if (!panel) return
  const dispose = activeSessions.get(panel)
  if (dispose) dispose()
  else clearSidePanel(panel)
}

export function getOrCreateSidePanel(): HTMLElement {
  const existing = document.querySelector<HTMLElement>(
    "main > * > aside[class~='sidepanel-container']",
  )
  if (existing) return existing

  const pageContent = document.querySelector<HTMLDivElement>('main > div[class~="page-body-grid"]')
  if (!pageContent) throw new Error('page-content section not found')

  const asidePanel = document.createElement('aside')
  asidePanel.classList.add('sidepanel-container')
  pageContent.appendChild(asidePanel)
  return asidePanel
}

function createSidePanel(asidePanel: HTMLElement, ...inner: HTMLElement[]): HTMLDivElement {
  disposeSidePanel(asidePanel)
  asidePanel.classList.add('active')

  const events = new AbortController()
  const headerSection = document.querySelector<HTMLElement>('main > section.header')
  const updateSidepanelOffset = (): void => {
    if (!headerSection) {
      asidePanel.style.setProperty('--sidepanel-top-offset', '0px')
      return
    }
    const headerRect = headerSection.getBoundingClientRect()
    const stickyTop = Number.parseFloat(getComputedStyle(headerSection).top) || 0
    asidePanel.style.setProperty(
      '--sidepanel-top-offset',
      `${Math.max(0, headerRect.height + stickyTop)}px`,
    )
  }
  updateSidepanelOffset()
  window.addEventListener('resize', updateSidepanelOffset, { signal: events.signal })

  const observer = headerSection ? new ResizeObserver(updateSidepanelOffset) : null
  if (observer && headerSection) observer.observe(headerSection)

  let disposed = false
  const dispose = (): void => {
    if (disposed) return
    disposed = true
    if (activeSessions.get(asidePanel) === dispose) activeSessions.delete(asidePanel)
    events.abort()
    observer?.disconnect()
    clearSidePanel(asidePanel)
  }
  activeSessions.set(asidePanel, dispose)

  const closeButton = document.createElement('button')
  closeButton.type = 'button'
  closeButton.classList.add('close-button')
  closeButton.ariaLabel = 'close button'
  closeButton.title = 'close button'
  closeButton.innerHTML =
    '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="currentColor" stroke="currentColor"><use href="#close-button"></use></svg>'
  closeButton.addEventListener('click', dispose, { signal: events.signal })

  const redirectButton = document.createElement('button')
  redirectButton.type = 'button'
  redirectButton.classList.add('redirect-button')
  redirectButton.ariaLabel = 'redirect to page'
  redirectButton.title = 'redirect to page'
  redirectButton.innerHTML =
    '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="var(--gray)" stroke="none"><use href="#triple-dots"></use></svg>'
  redirectButton.addEventListener(
    'click',
    () => {
      const slug = asidePanel.dataset.slug
      if (!slug || !isFullSlug(slug)) return
      window.spaNavigate(
        new URL(resolveRelative(getFullSlug(window), slug), window.location.toString()),
      )
    },
    { signal: events.signal },
  )

  const header = document.createElement('div')
  header.classList.add('sidepanel-header', 'all-col')
  header.append(redirectButton, closeButton)

  const sideInner = document.createElement('div')
  sideInner.classList.add('sidepanel-inner')
  sideInner.append(...inner, header)
  asidePanel.appendChild(sideInner)
  return sideInner
}

export function beginSidePanelRequest(asidePanel: HTMLElement): SidePanelRequest {
  pendingRequests.get(asidePanel)?.abort()
  const controller = new AbortController()
  pendingRequests.set(asidePanel, controller)

  const isCurrent = (): boolean =>
    !controller.signal.aborted && pendingRequests.get(asidePanel) === controller

  const cancel = (): void => {
    if (pendingRequests.get(asidePanel) === controller) pendingRequests.delete(asidePanel)
    controller.abort()
  }

  const mount = (slug: FullSlug, ...inner: HTMLElement[]): HTMLDivElement | null => {
    if (!isCurrent()) return null
    pendingRequests.delete(asidePanel)
    asidePanel.dataset.slug = slug
    return createSidePanel(asidePanel, ...inner)
  }

  return { signal: controller.signal, mount, cancel }
}
