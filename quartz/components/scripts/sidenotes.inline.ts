import { debounce } from '../../util/debounce'

const SIDENOTE_WIDTH = 17
const SPACING = 1
const SIDENOTE_GUTTER = 1.5
const GAP = 1
const MIN_DESKTOP_WIDTH = 1400

const LABEL_ATTRS = ['role', 'tabindex', 'aria-expanded', 'aria-haspopup', 'data-inline'] as const
const CONTENT_CLASSES = ['sidenote-left', 'sidenote-right', 'sidenote-inline'] as const

function remToPx(rem: number): number {
  return rem * parseFloat(getComputedStyle(document.documentElement).fontSize)
}

function getMainColumn(content?: Element): HTMLElement | null {
  return (
    content?.closest<HTMLElement>('.main-col') ?? document.querySelector<HTMLElement>('.main-col')
  )
}

function cssPixelValue(value: string): number {
  const parsed = Number.parseFloat(value)
  return Number.isFinite(parsed) ? parsed : 0
}

function getOffsetParentRect(content: HTMLElement): Pick<DOMRect, 'left' | 'right'> {
  const offsetParent = content.offsetParent
  if (!offsetParent) return { left: 0, right: window.innerWidth }

  const rect = offsetParent.getBoundingClientRect()
  const style = getComputedStyle(offsetParent)

  return {
    left: rect.left + cssPixelValue(style.borderLeftWidth),
    right: rect.right - cssPixelValue(style.borderRightWidth),
  }
}

function getContentWidth(): number {
  const mainColumn = getMainColumn()
  if (!mainColumn) return remToPx(35)
  return mainColumn.getBoundingClientRect().width
}

function getViewportThresholds() {
  const contentWidth = getContentWidth()
  const sidenoteWidth = remToPx(SIDENOTE_WIDTH)
  const spacing = remToPx(SPACING)
  const gutter = remToPx(SIDENOTE_GUTTER)

  return {
    ultraWide: contentWidth + 2 * (sidenoteWidth + gutter + spacing),
    medium: contentWidth + sidenoteWidth + gutter + spacing,
  }
}

type LayoutMode = 'double-sided' | 'single-sided' | 'inline'

function getLayoutMode(): LayoutMode {
  const windowWidth = window.innerWidth

  if (windowWidth < MIN_DESKTOP_WIDTH) {
    return 'inline'
  }

  const thresholds = getViewportThresholds()

  if (windowWidth > thresholds.ultraWide) {
    return 'double-sided'
  } else if (windowWidth > thresholds.medium) {
    return 'single-sided'
  } else {
    return 'inline'
  }
}

interface SidenoteState {
  span: HTMLElement
  label: HTMLElement
  content: HTMLElement
  side?: 'left' | 'right'
  controller?: AbortController
}

class SidenoteManager {
  private sidenotes: SidenoteState[] = []
  private lastBottomLeft = 0
  private lastBottomRight = 0
  private layoutMode: LayoutMode = 'inline'

  constructor() {
    this.initialize()
  }

  private cleanupHandlers(state: SidenoteState) {
    state.controller?.abort()
    state.controller = undefined
  }

  private setActiveState(state: SidenoteState, active: boolean) {
    state.span.classList.toggle('active', active)
    state.label.classList.toggle('active', active)
  }

  private setExpandedState(state: SidenoteState, expanded: boolean) {
    const { label, content } = state
    label.setAttribute('aria-expanded', expanded.toString())
    content.style.display = expanded ? 'block' : 'none'
    content.setAttribute('aria-hidden', (!expanded).toString())
    this.setActiveState(state, expanded)
  }

  private isAnchorCollapsed(label: HTMLElement): boolean {
    return !!label.closest('.callout.is-collapsed, .collapse-shell:not(.is-open)')
  }

  private measureContentHeight(content: HTMLElement): number {
    const probe = content.cloneNode(true) as HTMLElement
    probe.removeAttribute('id')
    probe.style.cssText = 'display:block;visibility:hidden;position:absolute;left:0;top:0'
    content.parentElement?.appendChild(probe)
    const height = probe.getBoundingClientRect().height
    probe.remove()
    return height
  }

  private initialize() {
    const sidenoteSpans = document.querySelectorAll<HTMLSpanElement>('.sidenote')

    sidenoteSpans.forEach(span => {
      const label = span.querySelector<HTMLSpanElement>('.sidenote-label')
      if (!label) return

      const content = span.nextElementSibling as HTMLElement | null
      if (!content || !content.classList.contains('sidenote-content')) return

      content.style.display = 'none'
      content.setAttribute('aria-hidden', 'true')

      if (!label.hasAttribute('aria-controls') && content.id) {
        label.setAttribute('aria-controls', content.id)
      }

      this.sidenotes.push({ span, label, content })
    })
  }

  private reset() {
    this.lastBottomLeft = 0
    this.lastBottomRight = 0

    this.sidenotes.forEach(state => {
      const { label, content } = state

      this.cleanupHandlers(state)

      LABEL_ATTRS.forEach(attr => label.removeAttribute(attr))
      label.style.cursor = ''
      label.style.userSelect = ''

      this.setActiveState(state, false)

      content.style.cssText = 'display:none'
      content.classList.remove(...CONTENT_CLASSES)
      content.setAttribute('aria-hidden', 'true')
    })
  }

  private positionSideToSide(state: SidenoteState): boolean {
    const { span, label, content } = state
    const labelRect = label.getBoundingClientRect()
    const contentHeight = this.measureContentHeight(content)
    const scrollTop = window.scrollY || document.documentElement.scrollTop
    const topPosition = labelRect.top + scrollTop

    const footer = document.querySelector('footer')
    const footerTop = footer ? footer.getBoundingClientRect().top + scrollTop : Infinity
    if (topPosition + contentHeight > footerTop) return false

    const wouldOverlapSidepanel = !!document.querySelector('.sidepanel-container.active')

    const allowLeft = span.getAttribute('data-allow-left') !== 'false'
    const gap = remToPx(GAP)
    const leftSpace = topPosition - this.lastBottomLeft
    const rightSpace = topPosition - this.lastBottomRight

    let side: 'left' | 'right'
    if (allowLeft && leftSpace >= contentHeight + gap) {
      side = 'left'
    } else if (!wouldOverlapSidepanel && rightSpace >= contentHeight + gap) {
      side = 'right'
    } else {
      return false
    }

    content.classList.add(`sidenote-${side}`)
    content.style.display = 'block'
    content.setAttribute('aria-hidden', 'false')

    const mainColumn = getMainColumn(content)
    if (mainColumn) {
      const gutter = remToPx(SIDENOTE_GUTTER)
      const sidenoteWidth = remToPx(SIDENOTE_WIDTH)
      const mainRect = mainColumn.getBoundingClientRect()
      const parentRect = getOffsetParentRect(content)
      const offset =
        side === 'left'
          ? mainRect.left - parentRect.left - sidenoteWidth - gutter
          : parentRect.right - mainRect.right - sidenoteWidth - gutter

      content.style.left = ''
      content.style.right = ''
      content.style[side] = `${offset}px`
    }

    const bottomPosition = topPosition + contentHeight
    if (side === 'left') this.lastBottomLeft = bottomPosition
    else this.lastBottomRight = bottomPosition

    state.side = side
    return true
  }

  private positionInline(state: SidenoteState) {
    const { label, content } = state

    this.cleanupHandlers(state)

    content.classList.add('sidenote-inline')
    content.style.display = 'none'
    content.style.position = 'static'

    label.style.cursor = 'pointer'
    label.style.userSelect = 'none'
    label.setAttribute('role', 'button')
    label.setAttribute('tabindex', '0')
    label.setAttribute('aria-haspopup', 'true')
    label.setAttribute('data-inline', '')

    const toggle = () => {
      const isExpanded = label.getAttribute('aria-expanded') === 'true'
      this.setExpandedState(state, !isExpanded)
    }

    state.controller = new AbortController()
    const { signal } = state.controller

    label.addEventListener(
      'click',
      (e: MouseEvent) => {
        e.preventDefault()
        e.stopPropagation()
        e.stopImmediatePropagation()
        toggle()
      },
      { capture: true, signal },
    )

    label.addEventListener(
      'keydown',
      (e: KeyboardEvent) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault()
          e.stopPropagation()
          toggle()
        }
      },
      { signal },
    )

    // always start collapsed
    this.setExpandedState(state, false)
  }

  public layout() {
    this.layoutMode = getLayoutMode()
    this.reset()

    this.sidenotes.forEach(state => {
      if (this.isAnchorCollapsed(state.label)) return

      const forceInline = state.span.getAttribute('data-force-inline') === 'true'

      if (this.layoutMode === 'inline' || forceInline) {
        this.positionInline(state)
      } else {
        const success = this.positionSideToSide(state)
        if (!success) {
          this.positionInline(state)
        }
      }
    })
  }
}

function setupSidenotes() {
  const manager = new SidenoteManager()
  manager.layout()

  const debouncedLayout = debounce(() => manager.layout(), 100)

  window.addEventListener('resize', debouncedLayout, { passive: true })

  // collapse toggles change document flow and anchor visibility
  document.addEventListener('callouttoggle', debouncedLayout)
  document.addEventListener('collapsibletoggle', debouncedLayout)

  // watch for sidepanel state changes
  const sidepanel = document.querySelector('.sidepanel-container')
  let observer: MutationObserver | null = null

  if (sidepanel) {
    observer = new MutationObserver(() => debouncedLayout())
    observer.observe(sidepanel, { attributes: true, attributeFilter: ['class'] })
  }

  window.addCleanup(() => {
    window.removeEventListener('resize', debouncedLayout)
    document.removeEventListener('callouttoggle', debouncedLayout)
    document.removeEventListener('collapsibletoggle', debouncedLayout)
    if (observer) {
      observer.disconnect()
    }
  })
}

document.addEventListener('nav', setupSidenotes)
document.addEventListener('contentdecrypted', setupSidenotes)
