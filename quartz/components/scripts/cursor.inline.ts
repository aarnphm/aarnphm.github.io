import { beginSitePerformanceSample, endSitePerformanceSample } from './performance-sample'

const configuredCursors = new WeakMap<HTMLElement, AbortController>()

const CLOSE_CURSOR_SELECTOR = "[data-site-cursor-close], button[aria-label^='close' i]"
const ACTION_CURSOR_SELECTOR = '[data-site-cursor-action]'
const MAGNETIC_ICON_SELECTOR = '[data-site-cursor-icon], svg'

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
  '.tri-compare-map',
  ".tri-compare-chart[data-compare-chart='power-curve'] .tri-cax-xax",
  ".tri-compare-chart:not([data-available='0']) .tri-compare-graph",
  '.tri-pmc-svg',
  '.tri-radar-svg',
  '.tri-accl-svg',
  '.tri-dist-plots',
  '.tri-engine-spark',
  '.tri-day-sleep-line-svg',
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
    cursorPart('site-cursor-dot'),
    cursorPart('site-cursor-question', '?'),
    cursorPart('site-cursor-crosshair'),
    cursorPart('site-cursor-line'),
  )
  return cursor
}

document.addEventListener('nav', () => {
  const cursors = Array.from(document.querySelectorAll<HTMLElement>('.site-cursor'))
  const cursor = cursors[cursors.length - 1] ?? createCursor()
  for (const candidate of cursors) {
    if (candidate !== cursor) candidate.remove()
  }
  document.body.appendChild(cursor)
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
    | 'close'
    | 'crosshair'
    | 'diamond'
    | 'help'
    | 'pointer'
    | 'text'
    | 'timeline' = 'diamond'
  let visible = false
  let lineScale = 1
  let pointerTarget: Element | null = null
  let magneticTarget: HTMLElement | null = null
  let magneticTimer = 0
  let measuredMagnetic: HTMLElement | null = null
  let measuredMagneticRect: DOMRect | null = null
  let measuredBars: HTMLElement | null = null
  let measuredBarsRect: DOMRect | null = null
  let measuredTimelineRect: DOMRect | null = null
  let geometryDirty = true

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
    const close = pointerTarget?.closest<HTMLElement>(CLOSE_CURSOR_SELECTOR) ?? null
    const action = pointerTarget?.closest<HTMLElement>(ACTION_CURSOR_SELECTOR) ?? null
    const magnetic = close ?? action
    const bars = pointerTarget?.closest<HTMLElement>('.tri-bars') ?? null
    if (magnetic) {
      const anchor = magnetic.querySelector<HTMLElement>(MAGNETIC_ICON_SELECTOR) ?? magnetic
      if (geometryDirty || magnetic !== measuredMagnetic || !measuredMagneticRect) {
        measuredMagnetic = magnetic
        measuredMagneticRect = anchor.getBoundingClientRect()
      }
      const rect = measuredMagneticRect
      setMagneticTarget(magnetic)
      renderX = rect.left + rect.width / 2
      renderY = rect.top + rect.height / 2
      lineScale = 1
      mode = close ? 'close' : 'action'
      visible = true
    } else if (bars) {
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
      renderY = barsRect.top + lineHeight / 2
      lineScale = Math.max(1, lineHeight / 24)
      mode = 'timeline'
      visible = true
    } else {
      setMagneticTarget(null)
      const help = pointerTarget?.closest<HTMLElement>(HELP_CURSOR_SELECTOR) ?? null
      const pointer = pointerTarget?.closest<HTMLElement>(POINTER_CURSOR_SELECTOR) ?? null
      const text = pointerTarget?.closest<HTMLElement>(TEXT_CURSOR_SELECTOR) ?? null
      const crosshair = pointerTarget?.closest<HTMLElement>(CROSSHAIR_CURSOR_SELECTOR) ?? null
      lineScale = 1
      if (help) {
        mode = 'help'
        visible = true
      } else if (pointer) {
        mode = 'pointer'
        visible = true
      } else if (text) {
        lineScale = 2 / 3
        mode = 'text'
        visible = true
      } else if (crosshair) {
        mode = 'crosshair'
        visible = true
      } else {
        mode = 'diamond'
        visible = pointerTarget != null
      }
    }
    cursor.style.transform = `translate3d(${renderX - 12}px, ${renderY - 12}px, 0)`
    line.style.transform = `translate(-50%, -50%) scaleY(${lineScale.toFixed(3)})`
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
    mode = 'diamond'
    visible = false
    schedule()
  }

  const invalidateGeometry = (): void => {
    geometryDirty = true
    if (pointerTarget) schedule()
  }

  document.documentElement.classList.add('site-cursor-ready')
  document.addEventListener('pointermove', onMove, { signal })
  document.addEventListener('click', onClick, { signal })
  document.addEventListener('pointerleave', hide, { signal })
  window.addEventListener('blur', hide, { signal })
  window.addEventListener('resize', invalidateGeometry, { signal })
  window.addEventListener('scroll', invalidateGeometry, { capture: true, passive: true, signal })
  window.addCleanup(() => {
    controller.abort()
    if (frame !== 0) window.cancelAnimationFrame(frame)
    window.clearTimeout(magneticTimer)
    magneticTarget?.removeAttribute('data-site-cursor-active')
    frame = 0
    delete cursor.dataset.magnetic
    if (configuredCursors.get(cursor) === controller) {
      document.documentElement.classList.remove('site-cursor-ready')
      configuredCursors.delete(cursor)
    }
  })
})
