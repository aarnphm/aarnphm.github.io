interface ToastConfig {
  containerId?: string
  containerStyles?: Partial<CSSStyleDeclaration>
  defaultDurationMs?: number
  maxVisible?: number
}

export interface ToastShowOptions {
  durationMs?: number
  styles?: Partial<CSSStyleDeclaration>
  containerId?: string
  containerStyles?: Partial<CSSStyleDeclaration>
  containerHost?: HTMLElement
  action?: { label: string; onClick: () => void }
}

export interface ToastEventDetail extends ToastShowOptions {
  message: string
}

const defaultContainerStyles: Partial<CSSStyleDeclaration> = {
  position: 'fixed',
  bottom: '1.25rem',
  right: '5rem',
  zIndex: '2147483647',
  display: 'grid',
  gridAutoFlow: 'row',
  gridAutoRows: 'minmax(0, max-content)',
  gridTemplateColumns: '1fr',
  justifyItems: 'center',
  alignItems: 'end',
  pointerEvents: 'none',
}

const baseToastStyles: Partial<CSSStyleDeclaration> = {
  boxSizing: 'content-box',
  pointerEvents: 'auto',
  position: 'relative',
  display: 'inline-flex',
  alignItems: 'center',
  gap: '0.5rem',
  padding: '0.5rem 0.75rem',
  fontSize: '12px',
  fontWeight: '500',
  opacity: '0',
  transform: 'translate3d(0, 12px, 0) scale(0.96)',
  transition: 'opacity 160ms ease, transform 210ms cubic-bezier(0.22, 1, 0.36, 1)',
  transformOrigin: 'bottom center',
  willChange: 'transform, opacity',
  width: '100%',
  minWidth: 'auto',
  maxWidth: '100%',
  gridRow: '1',
  gridColumn: '1',
  justifySelf: 'center',
  alignSelf: 'end',
}

const toastSurfaceStyles: Partial<CSSStyleDeclaration> = {
  background: 'var(--light)',
  color: 'var(--dark)',
  border: '1px solid var(--gray)',
  borderRadius: '0',
}

const transitionBufferMs = 220
const MAX_VISIBLE_DEFAULT = 3
const STACK_OFFSET_Y = 16
const STACK_OFFSET_X = 12
const STACK_SCALE_DECAY = 0.1
const STACK_MIN_SCALE = 0.81
const STACK_WIDTH_STEP = 16
const STACK_MIN_WIDTH = 100
const STACK_MAX_WIDTH = 180
const STACK_VIEWPORT_GUTTER = 20

interface ToastEntry {
  element: HTMLDivElement
  actionButton?: HTMLButtonElement
  hideTimer?: number
  teardownTimer?: number
}

export class Toast {
  private container: HTMLDivElement | null = null
  private containerId: string
  private containerStyles: Partial<CSSStyleDeclaration>
  private containerHostCleanup: (() => void) | null = null
  private readonly defaultDurationMs: number
  private readonly maxVisible: number
  private readonly toasts: ToastEntry[] = []

  constructor(config: ToastConfig = {}) {
    this.containerId = config.containerId ?? 'toast-container'
    this.containerStyles = { ...defaultContainerStyles, ...config.containerStyles }
    this.defaultDurationMs = config.defaultDurationMs ?? 1800
    this.maxVisible = Math.max(1, config.maxVisible ?? MAX_VISIBLE_DEFAULT)
  }

  show(message: string, options: ToastShowOptions = {}) {
    const container = this.ensureContainer(
      options.containerId,
      options.containerStyles,
      options.containerHost,
    )

    const toast = document.createElement('div')
    let actionButton: HTMLButtonElement | undefined
    if (options.action) {
      const text = document.createElement('span')
      text.textContent = message
      text.title = message
      Object.assign(text.style, {
        minWidth: '0',
        flex: '1',
        display: '-webkit-box',
        WebkitBoxOrient: 'vertical',
        WebkitLineClamp: '3',
        overflow: 'hidden',
        overflowWrap: 'anywhere',
      })
      actionButton = document.createElement('button')
      actionButton.type = 'button'
      actionButton.textContent = options.action.label
      Object.assign(actionButton.style, {
        flexShrink: '0',
        padding: '0.2rem 0.35rem',
        color: 'var(--dark)',
        background: 'transparent',
        border: '1px solid var(--gray)',
        borderRadius: '0',
        font: 'inherit',
        cursor: 'pointer',
      })
      toast.append(text, actionButton)
    } else {
      toast.textContent = message
    }

    Object.assign(toast.style, baseToastStyles)
    Object.assign(toast.style, toastSurfaceStyles)
    toast.dataset.toastState = 'entering'
    if (options.styles) {
      Object.assign(toast.style, options.styles)
    }

    container.appendChild(toast)
    const entry: ToastEntry = { element: toast, actionButton }
    if (actionButton && options.action) {
      const action = options.action
      actionButton.addEventListener('click', () => {
        actionButton.blur()
        this.fadeOut(entry)
        action.onClick()
      })
    }
    this.toasts.push(entry)

    this.enforceVisibleLimit()
    this.layoutToasts(entry)

    requestAnimationFrame(() => {
      this.layoutToasts()
    })

    const duration = options.durationMs ?? this.defaultDurationMs
    entry.hideTimer = window.setTimeout(() => this.fadeOut(entry), duration)
  }

  destroy() {
    for (const entry of this.toasts) {
      this.clearEntryHideTimer(entry)
      if (entry.teardownTimer !== undefined) {
        window.clearTimeout(entry.teardownTimer)
        entry.teardownTimer = undefined
      }
      entry.element.remove()
    }
    this.toasts.length = 0
    this.removeContainer()
  }

  private ensureContainer(
    containerId?: string,
    containerStyles?: Partial<CSSStyleDeclaration>,
    containerHost?: HTMLElement,
  ): HTMLDivElement {
    if (containerId && containerId !== this.containerId) {
      this.containerId = containerId
    }

    if (containerStyles) {
      this.containerStyles = { ...defaultContainerStyles, ...containerStyles }
    }

    if (this.container) {
      if (this.container.id !== this.containerId || !document.body?.contains(this.container)) {
        this.removeContainer()
      } else if (containerStyles) {
        Object.assign(this.container.style, this.containerStyles)
      }
    }

    if (!this.container) {
      const container = document.createElement('div')
      container.id = this.containerId
      container.setAttribute('role', 'status')
      container.setAttribute('aria-live', 'polite')
      container.setAttribute('aria-atomic', 'false')
      container.setAttribute('aria-relevant', 'additions text')
      Object.assign(container.style, this.containerStyles)
      this.container = container
    }

    this.setContainerHost(containerHost)
    return this.container
  }

  private setContainerHost(host?: HTMLElement) {
    const parent =
      host?.isConnected && (!(host instanceof HTMLDialogElement) || host.open)
        ? host
        : (document.body ?? document.documentElement)
    if (!this.container) return
    this.container.style.position =
      parent instanceof HTMLDialogElement && parent.matches(':modal')
        ? 'absolute'
        : (this.containerStyles.position ?? 'fixed')
    if (this.container.parentElement === parent) return
    this.containerHostCleanup?.()
    this.containerHostCleanup = null
    parent.appendChild(this.container)
    if (parent instanceof HTMLDialogElement) {
      const onClose = () => {
        if (!parent.open && this.container?.parentElement === parent) this.setContainerHost()
      }
      parent.addEventListener('close', onClose)
      this.containerHostCleanup = () => parent.removeEventListener('close', onClose)
    }
  }

  private layoutToasts(entering?: ToastEntry) {
    const visible = this.toasts.slice(-this.maxVisible)
    const hiddenCount = this.toasts.length - visible.length
    const maxDepth = visible.length - 1
    const baseWidthPx = Math.max(
      STACK_MIN_WIDTH,
      Math.min(STACK_MAX_WIDTH, window.innerWidth - STACK_VIEWPORT_GUTTER),
    )

    if (this.container) {
      this.container.style.width = `${baseWidthPx}px`
      this.container.style.setProperty('--toast-visible-count', `${visible.length}`)
      this.container.style.setProperty('--toast-base-width', `${baseWidthPx}px`)
    }

    for (let i = 0; i < hiddenCount; i++) {
      const entry = this.toasts[i]
      entry.element.style.display = 'none'
      entry.element.dataset.toastState = 'hidden'
      if (entry.actionButton) {
        entry.actionButton.tabIndex = -1
        entry.actionButton.style.visibility = 'hidden'
      }
    }

    visible.forEach((entry, idx) => {
      const depth = visible.length - 1 - idx
      const el = entry.element
      el.style.display = 'inline-flex'
      el.style.setProperty('--toast-depth', `${depth}`)
      el.dataset.toastDepth = `${depth}`

      const widthPx = Math.max(STACK_MIN_WIDTH, baseWidthPx - depth * STACK_WIDTH_STEP)
      el.style.width = `${widthPx}px`
      el.style.minWidth = `${widthPx}px`
      el.style.maxWidth = `${widthPx}px`

      if (entering && entry === entering) {
        el.style.opacity = '0'
        el.style.transform = `translate3d(${STACK_OFFSET_X}px, 16px, 0) scale(0.96)`
        el.dataset.toastState = 'entering'
      } else {
        const translateY = (depth - maxDepth) * STACK_OFFSET_Y
        const scale = Math.max(STACK_MIN_SCALE, 1 - depth * STACK_SCALE_DECAY)
        el.style.transform = `translate3d(${STACK_OFFSET_X}px, ${translateY}px, 0) scale(${scale})`
        el.style.opacity = '1'
        el.dataset.toastState = depth === 0 ? 'active' : 'stacked'
      }

      const paddingY = Math.max(6, 12 - depth * 2)
      el.style.padding = `${paddingY}px 12px`

      if (depth === 0) {
        el.style.color = 'var(--dark)'
        el.setAttribute('aria-hidden', 'false')
      } else {
        el.style.color = 'transparent'
        el.setAttribute('aria-hidden', 'true')
      }

      el.style.pointerEvents = depth === 0 ? 'auto' : 'none'
      if (entry.actionButton) {
        if (depth > 0 && document.activeElement === entry.actionButton) entry.actionButton.blur()
        entry.actionButton.tabIndex = depth === 0 ? 0 : -1
        entry.actionButton.style.visibility = depth === 0 ? 'visible' : 'hidden'
      }
      el.style.zIndex = `${100 - depth}`
      el.style.clipPath = ''
    })

    this.maybeRemoveContainer()
  }

  private fadeOut(entry: ToastEntry) {
    this.clearEntryHideTimer(entry)
    const index = this.toasts.indexOf(entry)
    if (index !== -1) {
      this.toasts.splice(index, 1)
    }

    const el = entry.element
    el.style.pointerEvents = 'none'
    el.dataset.toastState = 'exiting'
    if (entry.actionButton) {
      entry.actionButton.tabIndex = -1
      entry.actionButton.style.visibility = 'hidden'
    }

    const finalize = () => {
      el.removeEventListener('transitionend', finalize)
      entry.element.remove()
      this.maybeRemoveContainer()
    }

    el.addEventListener('transitionend', finalize, { once: true })
    entry.teardownTimer = window.setTimeout(finalize, transitionBufferMs)

    el.style.opacity = '0'
    el.style.transform = 'translate3d(0, -18px, 0) scale(0.9)'

    this.layoutToasts()
  }

  private enforceVisibleLimit() {
    while (this.toasts.length > this.maxVisible) {
      const entry = this.toasts.shift()
      if (entry) {
        this.clearEntryHideTimer(entry)
        if (entry.teardownTimer !== undefined) {
          window.clearTimeout(entry.teardownTimer)
          entry.teardownTimer = undefined
        }
        entry.element.remove()
      }
    }
    this.maybeRemoveContainer()
  }

  private clearEntryHideTimer(entry: ToastEntry) {
    if (entry.hideTimer !== undefined) {
      window.clearTimeout(entry.hideTimer)
      entry.hideTimer = undefined
    }
  }

  private removeContainer() {
    this.containerHostCleanup?.()
    this.containerHostCleanup = null
    if (this.container) {
      this.container.remove()
      this.container = null
    }
  }

  private maybeRemoveContainer() {
    if (this.container && this.toasts.length === 0 && this.container.childElementCount === 0) {
      this.removeContainer()
    }
  }
}
