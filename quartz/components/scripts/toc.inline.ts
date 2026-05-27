import type { RoughAnnotation } from 'rough-notation/lib/model'
import { annotate } from 'rough-notation'

let ag: RoughAnnotation | null = null
let tocController: AbortController | null = null
const tocScrollBuffer = 48
const tocHoverSigma = 42
const tocHoverRadius = tocHoverSigma * 3
const tocHoverLerp = 0.32
const tocHoverEpsilon = 0.08

interface TocButtonMetric {
  button: HTMLButtonElement
  fill: HTMLElement | null
  centerY: number
}
const observer = new IntersectionObserver(entries => {
  for (const entry of entries) {
    const slug = entry.target.id
    const tocEntryElement = document.querySelector<HTMLElement>(
      `.toc [data-for="${CSS.escape(slug)}"]`,
    )
    if (!tocEntryElement) continue

    const toc = document.querySelector<HTMLDivElement>('.toc')
    if (!toc) continue

    const windowHeight = entry.rootBounds?.height
    if (!windowHeight) continue

    const layout = toc.dataset.layout
    const inView = entry.boundingClientRect.y < windowHeight
    if (layout === 'minimal') {
      tocEntryElement.classList.toggle('in-view', inView)
      if (entry.isIntersecting && tocEntryElement instanceof HTMLButtonElement) {
        scrollTocButtonIntoView(tocEntryElement)
      }
    } else {
      tocEntryElement.classList.toggle('in-view', inView)
      tocEntryElement.parentElement?.classList.toggle('in-view', inView)
    }
  }
})

function onClick(evt: MouseEvent) {
  if (!(evt.target instanceof Element)) return

  const button = evt.target.closest<HTMLButtonElement>('button[data-href]')
  if (!button) return

  const href = button.dataset.href
  if (!href?.startsWith('#')) return

  evt.preventDefault()
  scrollToElement(href)

  const toc = button.closest<HTMLElement>('.toc')
  if (toc) {
    toc.classList.remove('is-hovering')
    hideTocLabel(toc)
  }
  resetTocButtons(toc?.querySelectorAll<HTMLButtonElement>('button[data-for]'))

  if (window.location.hash) {
    setTimeout(() => {
      scrollToElement(window.location.hash)
    }, 10)
  }
}

function scrollToElement(hash: string) {
  const elementId = hash.slice(1)
  const element = document.getElementById(elementId)
  if (!element) return

  const collapsibleParent = element.closest('.collapsible-header-content')
  if (collapsibleParent) {
    const wrapper = collapsibleParent.closest('.collapsible-header')
    const button = wrapper?.querySelector<HTMLButtonElement>('.toggle-button')
    if (button?.getAttribute('aria-expanded') === 'false') {
      button.click()
    }
  }

  const foldedTransclude = element.closest<HTMLElement>('.transclude-collapsible.is-collapsed')
  const foldButton = foldedTransclude?.querySelector<HTMLElement>('.transclude-fold')
  if (foldButton) {
    foldButton.click()
  }

  if (ag) ag.hide()

  const highlight = element.querySelector<HTMLElement>('span.highlight-span')
  if (highlight) {
    ag = annotate(highlight, {
      type: 'bracket',
      color: 'rgba(234, 157, 52, 0.45)',
      animate: false,
      multiline: true,
      brackets: ['left', 'right'],
    })

    const annotation = ag
    setTimeout(() => annotation.show(), 50)
    window.setTimeout(() => ag?.hide(), 2500)
  }

  const rect = element.getBoundingClientRect()
  const absoluteTop = window.scrollY + rect.top

  window.scrollTo({ top: absoluteTop - 100, behavior: 'smooth' })

  history.pushState(null, '', hash)
}

document.addEventListener('nav', (ev: CustomEventMap['nav']) => {
  if (ev.detail.url) {
    const url = new URL(ev.detail.url, window.location.origin)
    if (url.hash) {
      scrollToElement(decodeURIComponent(url.hash))
    }
  }
})

function setupToc() {
  tocController?.abort()
  tocController = null

  const toc = document.querySelector<HTMLElement>('.toc[data-layout="minimal"]')
  if (!toc) return

  if (getComputedStyle(toc).display === 'none') return

  const nav = toc.querySelector<HTMLElement>('#toc-vertical')
  if (!nav) return

  const buttons = toc.querySelectorAll<HTMLButtonElement>('button[data-for]')
  if (buttons.length === 0) return

  const controller = new AbortController()
  const { signal } = controller
  tocController = controller
  let metrics = readTocButtonMetrics(buttons)
  let maxScale = readTocMaxScale(nav, metrics)

  for (const button of buttons) {
    button.addEventListener('click', onClick, { signal })
  }

  let frame = 0
  let currentMouseY = 0
  let targetMouseY = 0
  let activeButton: HTMLButtonElement | null = null
  let hovering = false
  let touchedButtons = new Set<HTMLButtonElement>()

  const scheduleHover = () => {
    if (frame === 0) {
      frame = requestAnimationFrame(updateHover)
    }
  }

  const onMouseEnter = (evt: MouseEvent) => {
    hovering = true
    toc.classList.add('is-hovering')
    const navRect = nav.getBoundingClientRect()
    targetMouseY = evt.clientY - navRect.top
    currentMouseY = targetMouseY
    scheduleHover()
  }

  const onMouseLeave = () => {
    hovering = false
    toc.classList.remove('is-hovering')
    if (frame !== 0) {
      cancelAnimationFrame(frame)
      frame = 0
    }
    activeButton?.classList.remove('is-active')
    activeButton = null
    hideTocLabel(toc)
    resetTocButtons(buttons)
    touchedButtons.clear()
  }

  const updateHover = () => {
    frame = 0
    currentMouseY += (targetMouseY - currentMouseY) * tocHoverLerp

    const contentMouseY = currentMouseY + nav.scrollTop
    const nextTouchedButtons = new Set<HTMLButtonElement>()
    const startIndex = firstTocMetricIndexAt(metrics, contentMouseY - tocHoverRadius)
    const endIndex = firstTocMetricIndexAt(metrics, contentMouseY + tocHoverRadius)

    for (let index = startIndex; index < endIndex; index++) {
      const metric = metrics[index]
      updateTocButtonFill(metric, contentMouseY, maxScale)
      nextTouchedButtons.add(metric.button)
    }

    touchedButtons.forEach(button => {
      if (!nextTouchedButtons.has(button)) {
        resetTocButton(button)
      }
    })
    touchedButtons = nextTouchedButtons

    const nearestMetric = nearestTocMetric(metrics, contentMouseY)
    if (nearestMetric) {
      activeButton = updateTocLabel(toc, nearestMetric.button, activeButton, currentMouseY)
    }

    if (hovering && Math.abs(targetMouseY - currentMouseY) > tocHoverEpsilon) {
      scheduleHover()
    }
  }

  const onMouseMove = (evt: MouseEvent) => {
    const navRect = nav.getBoundingClientRect()
    targetMouseY = evt.clientY - navRect.top
    scheduleHover()
  }

  nav.addEventListener('mouseenter', onMouseEnter, { signal })
  nav.addEventListener('mouseleave', onMouseLeave, { signal })
  nav.addEventListener('mousemove', onMouseMove, { signal })
  nav.addEventListener(
    'scroll',
    () => {
      updateTocOverflow(nav)
      if (toc.classList.contains('is-hovering')) {
        scheduleHover()
      }
    },
    { passive: true, signal },
  )

  requestAnimationFrame(() => {
    metrics = readTocButtonMetrics(buttons)
    maxScale = readTocMaxScale(nav, metrics)
    updateTocOverflow(nav)
  })
}

function resetTocButtons(buttons?: NodeListOf<HTMLButtonElement>) {
  buttons?.forEach(resetTocButton)
}

function resetTocButton(button: HTMLButtonElement) {
  const fill = button.querySelector<HTMLElement>('.fill')
  button.classList.remove('is-active')
  if (!fill) return

  fill.style.animation = 'none'
  fill.style.transform = 'scaleX(1)'
  fill.style.opacity = ''
}

function readTocButtonMetrics(buttons: NodeListOf<HTMLButtonElement>): TocButtonMetric[] {
  const metrics: TocButtonMetric[] = []
  buttons.forEach(button => {
    metrics.push({
      button,
      fill: button.querySelector<HTMLElement>('.fill'),
      centerY: button.offsetTop + button.offsetHeight / 2,
    })
  })
  return metrics
}

function readTocMaxScale(nav: HTMLElement, metrics: TocButtonMetric[]): number {
  const baseWidth = Math.max(1, metrics[0]?.fill?.offsetWidth ?? 1)
  return Math.max(1, nav.clientWidth / baseWidth)
}

function firstTocMetricIndexAt(metrics: TocButtonMetric[], centerY: number): number {
  let low = 0
  let high = metrics.length
  while (low < high) {
    const mid = Math.floor((low + high) / 2)
    if (metrics[mid].centerY < centerY) {
      low = mid + 1
    } else {
      high = mid
    }
  }
  return low
}

function nearestTocMetric(metrics: TocButtonMetric[], centerY: number): TocButtonMetric | null {
  const nextIndex = firstTocMetricIndexAt(metrics, centerY)
  const previous = metrics[nextIndex - 1]
  const next = metrics[nextIndex]
  if (!previous) return next ?? null
  if (!next) return previous

  return centerY - previous.centerY <= next.centerY - centerY ? previous : next
}

function updateTocButtonFill(metric: TocButtonMetric, mouseY: number, maxScale: number): void {
  const { fill } = metric
  if (!fill) return

  const distance = mouseY - metric.centerY
  const falloff = Math.exp(-(distance * distance) / (2 * tocHoverSigma * tocHoverSigma))
  const scale = 1 + (maxScale - 1) * falloff

  fill.style.animation = 'none'
  fill.style.transform = `scaleX(${scale.toFixed(3)})`
}

function hideTocLabel(toc: HTMLElement) {
  toc.querySelector<HTMLElement>('.toc-label')?.classList.remove('is-visible')
}

function updateTocLabel(
  toc: HTMLElement,
  button: HTMLButtonElement,
  activeButton: HTMLButtonElement | null,
  labelY: number,
): HTMLButtonElement {
  const label = toc.querySelector<HTMLElement>('.toc-label')
  if (!label) return button

  if (button !== activeButton) {
    activeButton?.classList.remove('is-active')
    button.classList.add('is-active')
    const indicator = button.querySelector<HTMLElement>('.indicator')
    label.replaceChildren()
    if (indicator && indicator.childNodes.length > 0) {
      indicator.childNodes.forEach(child => label.appendChild(child.cloneNode(true)))
    } else {
      label.textContent = button.getAttribute('aria-label') ?? ''
    }
  }

  toc.style.setProperty('--toc-label-y', `${labelY.toFixed(1)}px`)
  label.classList.add('is-visible')
  return button
}

function updateTocOverflow(nav: HTMLElement) {
  const scrollable = nav.scrollHeight > nav.clientHeight + 1
  const atStart = nav.scrollTop <= 1
  const atEnd = nav.scrollTop + nav.clientHeight >= nav.scrollHeight - 1

  nav.classList.toggle('is-scrollable', scrollable)
  nav.classList.toggle('at-start', scrollable && atStart)
  nav.classList.toggle('at-end', scrollable && atEnd)
}

function scrollTocButtonIntoView(button: HTMLButtonElement) {
  const nav = button.closest<HTMLElement>('#toc-vertical')
  if (!nav || nav.scrollHeight <= nav.clientHeight + 1) return

  const navRect = nav.getBoundingClientRect()
  const buttonRect = button.getBoundingClientRect()
  const before = buttonRect.top - navRect.top - tocScrollBuffer
  const after = buttonRect.bottom - navRect.bottom + tocScrollBuffer
  let nextScroll = nav.scrollTop

  if (before < 0) {
    nextScroll += before
  } else if (after > 0) {
    nextScroll += after
  }

  if (Math.abs(nextScroll - nav.scrollTop) >= 1) {
    nav.scrollTop = nextScroll
  }
}

window.addEventListener('resize', setupToc)
document.addEventListener('nav', () => {
  setupToc()
  observer.disconnect()
  document
    .querySelectorAll('h1[id], h2[id], h3[id], h4[id], h5[id], h6[id]')
    .forEach(header => observer.observe(header))

  window.addCleanup(() => {
    tocController?.abort()
    tocController = null
  })
})
