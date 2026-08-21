import { beginSitePerformanceSample, endSitePerformanceSample } from './performance-sample'

const configuredCursors = new WeakMap<HTMLElement, AbortController>()

const isTriathlonSlug = (): boolean => {
  const slug = document.body.dataset.slug ?? ''
  return slug === 'triathlon' || slug.startsWith('triathlon/')
}

const CLOSE_CURSOR_SELECTOR =
  "[data-site-cursor-close]:not(:disabled), button[aria-label^='close' i]:not(:disabled)"
const ACTION_CURSOR_SELECTOR = '[data-site-cursor-action]:not(:disabled)'
const MAGNETIC_ICON_SELECTOR = '[data-site-cursor-icon], svg'
const BRACKET_ANCHOR_SELECTOR = '[data-site-cursor-bracket]'

const HELP_CURSOR_SELECTOR = [
  '[data-gloss]',
  '[data-tip-h][data-tip-d]',
  '[data-site-cursor-help]',
  '.tri-ratio-efficiency-row td[data-efficiency-delta]',
  '.tri-rdy-leg',
  '.tri-day-analytics-metric[aria-describedby]',
  '.stream-legend-lock',
  '.rht-arc-group',
  '.rht-legend-item',
].join(',')

const CROSSHAIR_CURSOR_SELECTOR = [
  '[data-site-cursor-crosshair]',
  '.tri-elev',
  '.tri-power-radar-svg',
  '.tri-best-power-svg',
  '.tri-best-power .tri-cax-xax',
  '.tri-curve-chart .tri-cax-xax',
  ".tri-compare-chart[data-compare-chart='power-curve'] .tri-cax-xax",
  ".tri-compare-chart:not([data-available='0']) .tri-compare-graph",
  '.tri-pmc-svg',
  '.tri-radar-svg',
  '.tri-accl-svg',
  '.tri-dist-plots',
  '.tri-engine-spark',
  '.tri-day-sleep-line-svg',
  '.tri-day-sleep-stage-svg',
].join(',')

const POINTER_CURSOR_SELECTOR = [
  'a[href]',
  'button:not(:disabled)',
  'label[for]',
  'summary',
  "[role='button']:not([aria-disabled='true'])",
  "[role='link']:not([aria-disabled='true'])",
  "[role='menuitem']:not([aria-disabled='true'])",
  "[role='option']:not([aria-disabled='true'])",
  "[role='tab']:not([aria-disabled='true'])",
].join(',')

const TEXT_CURSOR_SELECTOR = [
  'input:not([type])',
  "input[type='date']",
  "input[type='datetime-local']",
  "input[type='email']",
  "input[type='month']",
  "input[type='number']",
  "input[type='password']",
  "input[type='search']",
  "input[type='tel']",
  "input[type='text']",
  "input[type='time']",
  "input[type='url']",
  "input[type='week']",
  'textarea',
  "[contenteditable]:not([contenteditable='false'])",
].join(',')

const cursorPart = (className: string, text?: string): HTMLSpanElement => {
  const part = document.createElement('span')
  part.className = className
  if (text) part.textContent = text
  return part
}

const createCursor = (): HTMLElement => {
  const cursor = document.createElement('span')
  cursor.className = 'site-cursor'
  cursor.dataset.mode = 'diamond'
  cursor.dataset.visible = 'false'
  cursor.setAttribute('aria-hidden', 'true')
  cursor.append(
    cursorPart('site-cursor-diamond'),
    cursorPart('site-cursor-question', '?'),
    cursorPart('site-cursor-crosshair'),
    cursorPart('site-cursor-line'),
  )
  return cursor
}

const createCursorBracket = (): HTMLElement => {
  const bracket = document.createElement('span')
  bracket.className = 'site-cursor-bracket'
  bracket.dataset.visible = 'false'
  bracket.setAttribute('aria-hidden', 'true')
  return bracket
}

document.addEventListener('nav', () => {
  document.documentElement.classList.remove('site-cursor-ready')
  if (!isTriathlonSlug()) return

  const cursors = Array.from(document.querySelectorAll<HTMLElement>('.site-cursor'))
  const cursor = cursors[cursors.length - 1] ?? createCursor()
  for (const candidate of cursors) {
    if (candidate !== cursor) candidate.remove()
  }
  const brackets = Array.from(document.querySelectorAll<HTMLElement>('.site-cursor-bracket'))
  const bracket = brackets[brackets.length - 1] ?? createCursorBracket()
  for (const candidate of brackets) {
    if (candidate !== bracket) candidate.remove()
  }
  document.body.appendChild(cursor)
  document.body.appendChild(bracket)
  const line = cursor.querySelector<HTMLElement>('.site-cursor-line')
  if (!line) return
  configuredCursors.get(cursor)?.abort()
  const controller = new AbortController()
  const signal = controller.signal
  configuredCursors.set(cursor, controller)

  let frame = 0
  let x = -40
  let y = -40
  let mode:
    | 'action'
    | 'bracket'
    | 'close'
    | 'crosshair'
    | 'diamond'
    | 'help'
    | 'text'
    | 'timeline' = 'diamond'
  let visible = false
  let pointerTarget: Element | null = null
  let bracketTarget: HTMLElement | null = null
  let magneticTarget: HTMLElement | null = null
  let magneticTimer = 0
  let measuredMagnetic: HTMLElement | null = null
  let measuredMagneticRect: DOMRect | null = null
  let measuredBars: HTMLElement | null = null
  let measuredBarsRect: DOMRect | null = null
  let measuredTimelineRect: DOMRect | null = null
  let measuredBracketRect: DOMRect | null = null
  let measuredPointer: HTMLElement | null = null
  let measuredPointerAnchor: HTMLElement | null = null
  let geometryDirty = true

  const setBracketTarget = (target: HTMLElement | null): void => {
    const targetChanged = target !== bracketTarget
    if (targetChanged) {
      bracketTarget = target
      measuredBracketRect = null
      if (target) bracket.style.color = window.getComputedStyle(target).color
    }
    const nextVisible = String(target !== null)
    if (bracket.dataset.visible !== nextVisible) bracket.dataset.visible = nextVisible
    if (!target) return
    if (!targetChanged && !geometryDirty && measuredBracketRect) return
    measuredBracketRect = target.getBoundingClientRect()
    const rect = measuredBracketRect
    bracket.style.width = `${rect.width}px`
    bracket.style.height = `${rect.height}px`
    bracket.style.transform = `translate3d(${rect.left}px, ${rect.top}px, 0)`
  }

  const setMagneticTarget = (target: HTMLElement | null): void => {
    if (target === magneticTarget) return
    magneticTarget?.removeAttribute('data-site-cursor-active')
    magneticTarget = target
    window.clearTimeout(magneticTimer)
    magneticTimer = 0
    if (target) {
      target.dataset.siteCursorActive = 'true'
      cursor.dataset.magnetic = 'true'
    } else {
      cursor.dataset.magnetic = 'release'
      magneticTimer = window.setTimeout(() => {
        magneticTimer = 0
        if (!magneticTarget) delete cursor.dataset.magnetic
      }, 110)
    }
  }

  const render = (): void => {
    const startedAt = beginSitePerformanceSample()
    frame = 0
    let renderX = x
    let renderY = y
    let lineScale: number | null = null
    let lineOffset = 0
    const close = pointerTarget?.closest<HTMLElement>(CLOSE_CURSOR_SELECTOR) ?? null
    const action = pointerTarget?.closest<HTMLElement>(ACTION_CURSOR_SELECTOR) ?? null
    const magnetic = close ?? action
    const bars = pointerTarget?.closest<HTMLElement>('.tri-bars') ?? null
    if (magnetic) {
      setBracketTarget(null)
      const anchor = magnetic.querySelector<HTMLElement>(MAGNETIC_ICON_SELECTOR) ?? magnetic
      if (geometryDirty || magnetic !== measuredMagnetic || !measuredMagneticRect) {
        measuredMagnetic = magnetic
        measuredMagneticRect = anchor.getBoundingClientRect()
      }
      const rect = measuredMagneticRect
      setMagneticTarget(magnetic)
      renderX = rect.left + rect.width / 2
      renderY = rect.top + rect.height / 2
      mode = close ? 'close' : 'action'
      visible = true
    } else if (bars) {
      setBracketTarget(null)
      setMagneticTarget(null)
      if (geometryDirty || bars !== measuredBars || !measuredBarsRect) {
        measuredBars = bars
        measuredBarsRect = bars.getBoundingClientRect()
        measuredTimelineRect =
          bars.closest<HTMLElement>('.tri-scroll')?.getBoundingClientRect() ?? null
      }
      const barsRect = measuredBarsRect
      const timelineRect = measuredTimelineRect
      const lineBottom = Math.max(barsRect.bottom, timelineRect?.bottom ?? barsRect.bottom)
      const lineHeight = lineBottom - barsRect.top
      lineOffset = barsRect.top + lineHeight / 2 - renderY
      lineScale = Math.max(1, lineHeight / 24)
      mode = 'timeline'
      visible = true
    } else {
      setMagneticTarget(null)
      const help = pointerTarget?.closest<HTMLElement>(HELP_CURSOR_SELECTOR) ?? null
      const pointer = pointerTarget?.closest<HTMLElement>(POINTER_CURSOR_SELECTOR) ?? null
      const text = pointerTarget?.closest<HTMLElement>(TEXT_CURSOR_SELECTOR) ?? null
      const crosshair = pointerTarget?.closest<HTMLElement>(CROSSHAIR_CURSOR_SELECTOR) ?? null
      if (help) {
        setBracketTarget(null)
        mode = 'help'
        visible = true
      } else if (pointer) {
        if (pointer !== measuredPointer) {
          measuredPointer = pointer
          measuredPointerAnchor = pointer.querySelector<HTMLElement>(BRACKET_ANCHOR_SELECTOR)
        }
        setBracketTarget(measuredPointerAnchor ?? pointer)
        mode = 'bracket'
        visible = true
      } else if (text) {
        setBracketTarget(null)
        lineScale = 2 / 3
        mode = 'text'
        visible = true
      } else if (crosshair) {
        setBracketTarget(null)
        mode = 'crosshair'
        visible = true
      } else {
        setBracketTarget(null)
        mode = 'diamond'
        visible = pointerTarget != null
      }
    }
    cursor.style.transform = `translate3d(${renderX - 12}px, ${renderY - 12}px, 0)`
    line.style.transform =
      lineScale === null
        ? ''
        : `translate(-50%, calc(-50% + ${lineOffset.toFixed(1)}px)) scaleY(${lineScale.toFixed(3)})`
    if (cursor.dataset.mode !== mode) cursor.dataset.mode = mode
    const nextVisible = String(visible)
    if (cursor.dataset.visible !== nextVisible) cursor.dataset.visible = nextVisible
    geometryDirty = false
    endSitePerformanceSample('cursor', startedAt)
  }

  const schedule = (): void => {
    if (frame !== 0) return
    frame = window.requestAnimationFrame(render)
  }

  const onMove = (event: PointerEvent): void => {
    if (event.pointerType !== 'mouse') return
    pointerTarget = event.target instanceof Element ? event.target : null
    x = event.clientX
    y = event.clientY
    schedule()
  }

  const onClick = (event: MouseEvent): void => {
    const target = event.target instanceof Element ? event.target : null
    if (target?.closest(CLOSE_CURSOR_SELECTOR)) {
      const nextTarget = document.elementFromPoint(event.clientX, event.clientY)
      pointerTarget = nextTarget?.closest(CLOSE_CURSOR_SELECTOR) ? null : nextTarget
      x = event.clientX
      y = event.clientY
      schedule()
      return
    }
    if (!target?.closest('.tri-bars')) return
    pointerTarget = target
    x = event.clientX
    y = event.clientY
    schedule()
  }

  const hide = (): void => {
    pointerTarget = null
    setBracketTarget(null)
    mode = 'diamond'
    visible = false
    schedule()
  }

  const invalidateGeometry = (): void => {
    geometryDirty = true
    pointerTarget = document.elementFromPoint(x, y)
    if (pointerTarget || visible || magneticTarget || bracketTarget || mode === 'timeline')
      schedule()
  }

  const invalidateScrollGeometry = (): void => {
    geometryDirty = true
    if (pointerTarget || visible || magneticTarget || bracketTarget || mode === 'timeline')
      schedule()
  }

  document.documentElement.classList.add('site-cursor-ready')
  document.addEventListener('pointermove', onMove, { signal })
  document.addEventListener('click', onClick, { signal })
  document.addEventListener('pointerleave', hide, { signal })
  window.addEventListener('blur', hide, { signal })
  window.addEventListener('resize', invalidateGeometry, { signal })
  window.addEventListener('scroll', invalidateScrollGeometry, {
    capture: true,
    passive: true,
    signal,
  })
  window.addCleanup(() => {
    controller.abort()
    if (frame !== 0) window.cancelAnimationFrame(frame)
    window.clearTimeout(magneticTimer)
    bracket.dataset.visible = 'false'
    magneticTarget?.removeAttribute('data-site-cursor-active')
    frame = 0
    delete cursor.dataset.magnetic
    if (configuredCursors.get(cursor) === controller) {
      document.documentElement.classList.remove('site-cursor-ready')
      configuredCursors.delete(cursor)
    }
  })
})
