import { rootNavSignal } from './root-lifecycle'

const configuredCursors = new WeakMap<HTMLElement, AbortSignal>()

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
  const signal = rootNavSignal(cursor)
  if (configuredCursors.get(cursor) === signal) return
  configuredCursors.set(cursor, signal)

  let frame = 0
  let x = -40
  let y = -40
  let mode: 'crosshair' | 'diamond' | 'help' | 'pointer' | 'text' | 'timeline' = 'diamond'
  let visible = false
  let lineScale = 1
  let pointerTarget: Element | null = null

  const render = (): void => {
    frame = 0
    const bars = pointerTarget?.closest<HTMLElement>('.tri-bars') ?? null
    if (bars) {
      const barsRect = bars.getBoundingClientRect()
      const timelineRect = bars.closest<HTMLElement>('.tri-scroll')?.getBoundingClientRect()
      const lineBottom = Math.max(barsRect.bottom, timelineRect?.bottom ?? barsRect.bottom)
      const lineHeight = lineBottom - barsRect.top
      y = barsRect.top + lineHeight / 2
      lineScale = Math.max(1, lineHeight / 24)
      mode = 'timeline'
      visible = true
    } else {
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
    cursor.style.transform = `translate3d(${x - 12}px, ${y - 12}px, 0)`
    line.style.transform = `translate(-50%, -50%) scaleY(${lineScale.toFixed(3)})`
    if (cursor.dataset.mode !== mode) cursor.dataset.mode = mode
    const nextVisible = String(visible)
    if (cursor.dataset.visible !== nextVisible) cursor.dataset.visible = nextVisible
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

  document.documentElement.classList.add('site-cursor-ready')
  document.addEventListener('pointermove', onMove, { signal })
  document.addEventListener('click', onClick, { signal })
  document.addEventListener('pointerleave', hide, { signal })
  window.addEventListener('blur', hide, { signal })
  signal.addEventListener(
    'abort',
    () => {
      if (frame !== 0) window.cancelAnimationFrame(frame)
      frame = 0
      document.documentElement.classList.remove('site-cursor-ready')
      configuredCursors.delete(cursor)
    },
    { once: true },
  )
})
