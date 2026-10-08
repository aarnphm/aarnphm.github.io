import { start } from '../../functional'
import {
  initialDatePickerModel,
  updateDatePicker,
  type DatePickerEffect,
  type DatePickerMessage,
  type DatePickerModel,
} from './date-picker-model'
import {
  addMonths,
  clampIsoDate,
  isoDate,
  isoMonth,
  monthGridDates,
  monthOf,
  parseIsoDate,
  parseIsoMonth,
  shiftIsoDate,
  todayParts,
} from './dates'

/** Fired on the panel to redraw an open picker after its bounds or selection change elsewhere. */
export const DATE_PICKER_RENDER_EVENT = 'datepicker:render'

export interface DatePickerLabels {
  text: (key: 'previous month' | 'next month' | 'clear' | 'today' | 'date picker') => string
  monthYear: (iso: string) => string
  weekdayNarrow: (day: number) => string
}

export const englishDateLabels: DatePickerLabels = {
  text: key => key,
  monthYear: iso =>
    new Date(`${iso}T00:00:00`)
      .toLocaleDateString('en', { month: 'long', year: 'numeric' })
      .toLowerCase(),
  weekdayNarrow: day =>
    new Date(2026, 0, 4 + day).toLocaleDateString('en', { weekday: 'narrow' }).toLowerCase(),
}

export interface DatePickerOptions {
  id: string
  label: string
  labels?: DatePickerLabels
  selected: () => string | undefined
  min?: () => string | undefined
  max?: () => string | undefined
  onOpen?: () => void
  onSelect: (date: string) => void
  onClear?: () => void
}

export interface DatePicker {
  wrap: HTMLElement
  trigger: HTMLButtonElement
  /** The trigger's text; the caller writes the formatted value here. */
  text: HTMLElement
  panel: HTMLElement
  render: () => void
  close: () => void
  mount: () => () => void
}

const SVGNS = 'http://www.w3.org/2000/svg'

const icon = (className: string, d: string, width: string): SVGElement => {
  const svg = document.createElementNS(SVGNS, 'svg')
  svg.setAttribute('class', className)
  svg.setAttribute('viewBox', '0 0 16 16')
  svg.setAttribute('fill', 'none')
  svg.setAttribute('aria-hidden', 'true')
  svg.setAttribute('focusable', 'false')
  const path = document.createElementNS(SVGNS, 'path')
  for (const [name, value] of Object.entries({
    d,
    stroke: 'currentColor',
    'stroke-width': width,
    'stroke-linecap': 'round',
    'stroke-linejoin': 'round',
  }))
    path.setAttribute(name, value)
  svg.append(path)
  return svg
}

export const calendarIcon = (): SVGElement =>
  icon(
    'g-datepicker-icon',
    'M4.5 2v2M11.5 2v2M3.5 5.5h9M4 3.5h8a1 1 0 0 1 1 1v7a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1v-7a1 1 0 0 1 1-1Z',
    '1.35',
  )

export const arrowIcon = (direction: -1 | 1): SVGElement =>
  icon('g-icon', direction < 0 ? 'M10 3.5 5.5 8l4.5 4.5' : 'M6 3.5 10.5 8 6 12.5', '1.6')

const button = (className: string, text?: string, attrs?: Record<string, string>) => {
  const element = document.createElement('button')
  element.type = 'button'
  element.className = className
  if (text !== undefined) element.textContent = text
  if (attrs) for (const name in attrs) element.setAttribute(name, attrs[name])
  return element
}

const element = (tag: string, className: string, text?: string): HTMLElement => {
  const node = document.createElement(tag)
  node.className = className
  if (text !== undefined) node.textContent = text
  return node
}

const PANEL_HEIGHT = 304
const PANEL_MIN_WIDTH = 238

// Fixed placement under the trigger, flipped above it when the panel would leave the viewport.
const place = (trigger: HTMLElement, panel: HTMLElement): void => {
  const rect = trigger.getBoundingClientRect()
  const width = Math.min(Math.max(PANEL_MIN_WIDTH, rect.width), window.innerWidth - 16)
  const left = Math.max(8, Math.min(rect.left, window.innerWidth - width - 8))
  const below = rect.bottom + 6
  const top =
    below + PANEL_HEIGHT > window.innerHeight ? Math.max(8, rect.top - PANEL_HEIGHT - 6) : below
  panel.style.inlineSize = `${width}px`
  panel.style.insetInlineStart = `${left}px`
  panel.style.insetBlockStart = `${top}px`
}

const KEY_OFFSETS: Record<string, number> = {
  ArrowLeft: -1,
  ArrowRight: 1,
  ArrowUp: -7,
  ArrowDown: 7,
  Home: -42,
  End: 42,
  PageUp: -31,
  PageDown: 31,
}

export const buildDatePicker = (options: DatePickerOptions): DatePicker => {
  const labels = options.labels ?? englishDateLabels
  const min = () => options.min?.()
  const max = () => options.max?.()
  const wrap = element('div', 'g-datepicker')
  const trigger = button('g-datepicker-trigger', undefined, {
    'aria-label': options.label,
    'aria-haspopup': 'dialog',
    'aria-expanded': 'false',
  })
  const text = element('span', 'g-datepicker-text')
  trigger.append(text, calendarIcon())
  const panel = element('div', 'g-datepicker-panel')
  panel.id = options.id
  panel.tabIndex = -1
  panel.setAttribute('role', 'dialog')
  panel.setAttribute('aria-label', `${options.label} · ${labels.text('date picker')}`)
  panel.setAttribute('popover', 'auto')
  trigger.setAttribute('aria-controls', panel.id)
  wrap.append(trigger, panel)

  const closePanel = (restoreFocus = false): void => {
    if (panel.matches(':popover-open') && typeof panel.hidePopover === 'function')
      panel.hidePopover()
    panel.removeAttribute('data-open')
    trigger.setAttribute('aria-expanded', 'false')
    if (restoreFocus) trigger.focus()
  }

  const renderPanel = (): void => {
    const lo = min()
    const hi = max()
    const now = todayParts()
    const today = isoDate(now.year, now.month, now.day)
    const selected = program.retrieve().selected ?? hi ?? lo ?? today
    const selectedParts = parseIsoDate(selected) ?? parseIsoDate(hi) ?? parseIsoDate(lo) ?? now
    const view = parseIsoMonth(program.retrieve().viewMonth) ?? monthOf(selectedParts)
    if (lo) panel.dataset.minDate = lo
    else delete panel.dataset.minDate
    if (hi) panel.dataset.maxDate = hi
    else delete panel.dataset.maxDate

    const loParts = parseIsoDate(lo)
    const hiParts = parseIsoDate(hi)
    const prevMonth = addMonths(view, -1)
    const nextMonth = addMonths(view, 1)
    const head = element('div', 'g-cal-head')
    const title = element(
      'span',
      'g-cal-title',
      labels.monthYear(isoDate(view.year, view.month, 1)),
    )
    const prev = button('g-icon-button', undefined, { 'aria-label': labels.text('previous month') })
    const next = button('g-icon-button', undefined, { 'aria-label': labels.text('next month') })
    prev.append(arrowIcon(-1))
    next.append(arrowIcon(1))
    if (loParts && isoMonth(prevMonth) < isoMonth(monthOf(loParts))) prev.disabled = true
    if (hiParts && isoMonth(nextMonth) > isoMonth(monthOf(hiParts))) next.disabled = true
    prev.addEventListener('click', () =>
      program.dispatch({ type: 'move-month', viewMonth: isoMonth(prevMonth) }),
    )
    next.addEventListener('click', () =>
      program.dispatch({ type: 'move-month', viewMonth: isoMonth(nextMonth) }),
    )
    head.append(title, prev, next)

    const week = element('div', 'g-cal-week')
    week.setAttribute('aria-hidden', 'true')
    for (let day = 0; day < 7; day += 1) week.append(element('span', '', labels.weekdayNarrow(day)))

    const grid = element('div', 'g-cal-grid')
    for (const value of monthGridDates(view)) {
      const day = button('g-cal-day', String(Number(value.slice(8))), {
        'data-date': value,
        'aria-label': value,
        'aria-pressed': String(value === selected),
      })
      if (Number(value.slice(5, 7)) !== view.month) day.dataset.outside = ''
      if (value === today) {
        day.dataset.today = ''
        day.setAttribute('aria-current', 'date')
      }
      if ((lo && value < lo) || (hi && value > hi)) day.disabled = true
      else day.addEventListener('click', () => program.dispatch({ type: 'select', date: value }))
      grid.append(day)
    }

    const foot = element('div', 'g-cal-foot')
    const clear = button('g-text-button', labels.text('clear'))
    const todayButton = button('g-text-button', labels.text('today'))
    clear.addEventListener('click', () => program.dispatch({ type: 'clear' }))
    todayButton.addEventListener('click', () =>
      program.dispatch({ type: 'select', date: clampIsoDate(today, lo, hi) }),
    )
    foot.append(clear, todayButton)
    panel.replaceChildren(head, week, grid, foot)
  }

  const focusSelection = (): void => {
    const target =
      panel.querySelector<HTMLButtonElement>('.g-cal-day[aria-pressed="true"]:not(:disabled)') ??
      panel.querySelector<HTMLButtonElement>('.g-cal-day:not(:disabled)')
    target?.focus()
  }

  const onTriggerClick = (): void => {
    if (panel.matches(':popover-open') || panel.dataset.open === 'true') {
      program.dispatch({ type: 'close' })
      return
    }
    options.onOpen?.()
    const selected = options.selected()
    const parts = parseIsoDate(selected) ?? parseIsoDate(max()) ?? todayParts()
    program.dispatch({ type: 'open', selected, viewMonth: isoMonth(monthOf(parts)) })
  }
  const onPanelToggle = (): void => {
    const open = panel.matches(':popover-open')
    trigger.setAttribute('aria-expanded', String(open))
    if (!open) program.dispatch({ type: 'close' })
  }
  const onPanelKeydown = (event: KeyboardEvent): void => {
    if (event.key === 'Escape') {
      program.dispatch({ type: 'close', restoreFocus: true })
      return
    }
    const offset = KEY_OFFSETS[event.key]
    if (offset === undefined) return
    const active = document.activeElement
    if (!(active instanceof HTMLButtonElement) || !active.classList.contains('g-cal-day')) return
    const target = shiftIsoDate(
      active.dataset.date,
      offset,
      panel.dataset.minDate,
      panel.dataset.maxDate,
    )
    if (!target) return
    event.preventDefault()
    program.dispatch({ type: 'focus-date', date: target.date, viewMonth: target.viewMonth })
  }
  const onRenderRequest = (): void =>
    program.dispatch({ type: 'sync', selected: options.selected() })

  const program = start<DatePickerModel, DatePickerMessage, DatePickerEffect>({
    init: () => ({ model: initialDatePickerModel(), effects: [] }),
    reduce: updateDatePicker,
    effects: effect => {
      if (effect.type === 'notify-select') {
        options.onSelect(effect.date)
        return
      }
      if (effect.type === 'notify-clear') {
        options.onClear?.()
        return
      }
      if (effect.type === 'close-panel') {
        closePanel(effect.restoreFocus)
        return
      }
      renderPanel()
      place(trigger, panel)
      if (!panel.matches(':popover-open')) {
        if (typeof panel.showPopover === 'function') panel.showPopover()
        else panel.dataset.open = 'true'
      }
      trigger.setAttribute('aria-expanded', 'true')
      if (effect.focusDate)
        panel
          .querySelector<HTMLButtonElement>(`.g-cal-day[data-date="${effect.focusDate}"]`)
          ?.focus()
      else focusSelection()
    },
  })

  return {
    wrap,
    trigger,
    text,
    panel,
    render: onRenderRequest,
    close: () => program.dispatch({ type: 'close' }),
    mount: () => {
      trigger.addEventListener('click', onTriggerClick)
      panel.addEventListener('toggle', onPanelToggle)
      panel.addEventListener('keydown', onPanelKeydown)
      panel.addEventListener(DATE_PICKER_RENDER_EVENT, onRenderRequest)
      return () => {
        trigger.removeEventListener('click', onTriggerClick)
        panel.removeEventListener('toggle', onPanelToggle)
        panel.removeEventListener('keydown', onPanelKeydown)
        panel.removeEventListener(DATE_PICKER_RENDER_EVENT, onRenderRequest)
        program.stop()
      }
    },
  }
}
