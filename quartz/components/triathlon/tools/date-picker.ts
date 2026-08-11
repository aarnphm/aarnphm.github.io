import type { TriathlonFormatter } from '../runtime/formatter'
import { start } from '../../../functional'
import { el } from '../runtime/dom'
import { svg } from '../runtime/dom'
import {
  initialDatePickerModel,
  updateDatePicker,
  type DatePickerEffect,
  type DatePickerMessage,
  type DatePickerModel,
} from './date-picker-model'

export interface PredDateParts {
  year: number
  month: number
  day: number
}

export interface PredMonthParts {
  year: number
  month: number
}

export interface PredDatePicker {
  wrap: HTMLElement
  trigger: HTMLButtonElement
  panel: HTMLElement
  render: () => void
  close: () => void
  mount: () => () => void
}

export interface DatePickerOptions {
  id: string
  formatter: TriathlonFormatter
  label: string
  selected: () => string | undefined
  min: () => string | undefined
  max: () => string | undefined
  onOpen: () => void
  onSelect: (date: string) => void
  onClear: () => void
}

export const predDatePad = (value: number): string => String(value).padStart(2, '0')

export const predDateValue = (year: number, month: number, day: number): string =>
  `${year}-${predDatePad(month)}-${predDatePad(day)}`

export const predDateFromLocal = (date: Date): string =>
  predDateValue(date.getFullYear(), date.getMonth() + 1, date.getDate())

export const parsePredDate = (value: string | undefined): PredDateParts | null => {
  if (!value) return null
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value)
  if (!match) return null
  const year = Number(match[1])
  const month = Number(match[2])
  const day = Number(match[3])
  const date = new Date(year, month - 1, day)
  if (date.getFullYear() !== year || date.getMonth() !== month - 1 || date.getDate() !== day)
    return null
  return { year, month, day }
}

export const predMonthValue = (parts: PredMonthParts): string =>
  `${parts.year}-${predDatePad(parts.month)}`

export const parsePredMonth = (value: string | undefined): PredMonthParts | null => {
  if (!value) return null
  const match = /^(\d{4})-(\d{2})$/.exec(value)
  if (!match) return null
  const year = Number(match[1])
  const month = Number(match[2])
  if (month < 1 || month > 12) return null
  return { year, month }
}

export const predMonthFromDate = (parts: PredDateParts): PredMonthParts => ({
  year: parts.year,
  month: parts.month,
})

export const addPredMonths = (parts: PredMonthParts, delta: number): PredMonthParts => {
  const date = new Date(parts.year, parts.month - 1 + delta, 1)
  return { year: date.getFullYear(), month: date.getMonth() + 1 }
}

export const predTodayParts = (): PredDateParts => {
  const date = new Date()
  return { year: date.getFullYear(), month: date.getMonth() + 1, day: date.getDate() }
}

export const clampPredDate = (
  value: string,
  min: string | undefined,
  max: string | undefined,
): string => {
  if (min && value < min) return min
  if (max && value > max) return max
  return value
}

export const predButton = (
  cls: string,
  text?: string,
  attrs?: Record<string, string>,
): HTMLButtonElement => {
  const button = document.createElement('button')
  button.className = cls
  button.type = 'button'
  if (text !== undefined) button.textContent = text
  if (attrs) for (const k in attrs) button.setAttribute(k, attrs[k])
  return button
}

export const buildPredCalendarIcon = (): SVGElement => {
  const icon = svg('svg', {
    class: 'tri-pred-date-ico',
    viewBox: '0 0 16 16',
    fill: 'none',
    'aria-hidden': 'true',
    focusable: 'false',
  })
  icon.append(
    svg('path', {
      d: 'M4.5 2v2M11.5 2v2M3.5 5.5h9M4 3.5h8a1 1 0 0 1 1 1v7a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1v-7a1 1 0 0 1 1-1Z',
      stroke: 'currentColor',
      'stroke-width': '1.35',
      'stroke-linecap': 'round',
      'stroke-linejoin': 'round',
    }),
  )
  return icon
}

export const buildPredCalendarArrow = (direction: -1 | 1): SVGElement => {
  const icon = svg('svg', {
    class: 'tri-pred-cal-arrow',
    viewBox: '0 0 16 16',
    fill: 'none',
    'aria-hidden': 'true',
    focusable: 'false',
  })
  icon.append(
    svg('path', {
      d: direction < 0 ? 'M10 3.5 5.5 8l4.5 4.5' : 'M6 3.5 10.5 8 6 12.5',
      stroke: 'currentColor',
      'stroke-width': '1.6',
      'stroke-linecap': 'round',
      'stroke-linejoin': 'round',
    }),
  )
  return icon
}

export const positionPredCalendar = (trigger: HTMLElement, panel: HTMLElement): void => {
  const rect = trigger.getBoundingClientRect()
  const width = Math.min(Math.max(238, rect.width), window.innerWidth - 16)
  const height = 304
  const left = Math.max(8, Math.min(rect.left, window.innerWidth - width - 8))
  const below = rect.bottom + 6
  const top = below + height > window.innerHeight ? Math.max(8, rect.top - height - 6) : below
  panel.style.inlineSize = `${width}px`
  panel.style.insetInlineStart = `${left}px`
  panel.style.insetBlockStart = `${top}px`
}

export const focusPredCalendarSelection = (panel: HTMLElement): void => {
  const selected =
    panel.querySelector<HTMLButtonElement>('.tri-pred-cal-day--selected:not(:disabled)') ??
    panel.querySelector<HTMLButtonElement>('.tri-pred-cal-day:not(:disabled)')
  selected?.focus()
}

export const predCalendarFocusTarget = (
  currentValue: string | undefined,
  offset: number,
  min: string | undefined,
  max: string | undefined,
): { date: string; viewMonth: string } | null => {
  const current = parsePredDate(currentValue)
  if (!current) return null
  const date = new Date(current.year, current.month - 1, current.day + offset)
  const next = clampPredDate(predDateFromLocal(date), min, max)
  const nextParts = parsePredDate(next)
  if (!nextParts) return null
  return { date: next, viewMonth: predMonthValue(predMonthFromDate(nextParts)) }
}

export const buildDatePicker = (options: DatePickerOptions): PredDatePicker => {
  const wrap = el('div', 'tri-pred-date-wrap')
  const trigger = predButton('tri-pred-date', undefined, {
    'aria-label': options.label,
    'aria-haspopup': 'dialog',
    'aria-expanded': 'false',
  })
  const text = el('span', 'tri-pred-date-text')
  trigger.append(text, buildPredCalendarIcon())
  const panel = el('div', 'tri-pred-calendar', undefined, {
    role: 'dialog',
    'aria-label': `${options.label} · ${options.formatter.text('date picker')}`,
    popover: 'auto',
  })
  panel.id = options.id
  panel.tabIndex = -1
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
    const min = options.min()
    const max = options.max()
    const today = predDateFromLocal(new Date())
    const selected =
      program.retrieve().selected ??
      max ??
      min ??
      predDateValue(predTodayParts().year, predTodayParts().month, predTodayParts().day)
    const selectedParts =
      parsePredDate(selected) ?? parsePredDate(max) ?? parsePredDate(min) ?? predTodayParts()
    const view = parsePredMonth(program.retrieve().viewMonth) ?? predMonthFromDate(selectedParts)
    if (min) panel.dataset.minDate = min
    else delete panel.dataset.minDate
    if (max) panel.dataset.maxDate = max
    else delete panel.dataset.maxDate

    const minMonth = parsePredDate(min)
    const maxMonth = parsePredDate(max)
    const prevMonth = addPredMonths(view, -1)
    const nextMonth = addPredMonths(view, 1)
    const monthTitle = options.formatter.monthYear(predDateValue(view.year, view.month, 1))
    const head = el('div', 'tri-pred-cal-head')
    const title = el('span', 'tri-pred-cal-title', monthTitle)
    const prev = predButton('tri-pred-cal-nav', undefined, {
      'aria-label': options.formatter.text('previous month'),
    })
    const next = predButton('tri-pred-cal-nav', undefined, {
      'aria-label': options.formatter.text('next month'),
    })
    prev.appendChild(buildPredCalendarArrow(-1))
    next.appendChild(buildPredCalendarArrow(1))
    if (minMonth && predMonthValue(prevMonth) < predMonthValue(predMonthFromDate(minMonth)))
      prev.disabled = true
    if (maxMonth && predMonthValue(nextMonth) > predMonthValue(predMonthFromDate(maxMonth)))
      next.disabled = true
    prev.addEventListener('click', () => {
      program.dispatch({ type: 'move-month', viewMonth: predMonthValue(prevMonth) })
    })
    next.addEventListener('click', () => {
      program.dispatch({ type: 'move-month', viewMonth: predMonthValue(nextMonth) })
    })
    head.append(title, prev, next)

    const week = el('div', 'tri-pred-cal-week')
    for (let day = 0; day < 7; day += 1)
      week.appendChild(el('span', 'tri-pred-cal-weekday', options.formatter.weekdayNarrow(day)))

    const grid = el('div', 'tri-pred-cal-grid')
    const monthStart = new Date(view.year, view.month - 1, 1)
    const gridStart = new Date(view.year, view.month - 1, 1 - monthStart.getDay())
    for (let i = 0; i < 42; i += 1) {
      const date = new Date(gridStart.getFullYear(), gridStart.getMonth(), gridStart.getDate() + i)
      const value = predDateFromLocal(date)
      const day = predButton('tri-pred-cal-day', String(date.getDate()), {
        'data-date': value,
        'aria-label': value,
      })
      if (date.getMonth() !== view.month - 1) day.classList.add('tri-pred-cal-day--muted')
      if (value === selected) {
        day.classList.add('tri-pred-cal-day--selected')
        day.setAttribute('aria-current', 'date')
      }
      if (value === today) day.classList.add('tri-pred-cal-day--today')
      if ((min && value < min) || (max && value > max)) day.disabled = true
      else
        day.addEventListener('click', () => {
          program.dispatch({ type: 'select', date: value })
        })
      grid.appendChild(day)
    }

    const foot = el('div', 'tri-pred-cal-foot')
    const clear = predButton('tri-pred-cal-action', options.formatter.text('clear'))
    const now = predButton('tri-pred-cal-action', options.formatter.text('today'))
    clear.addEventListener('click', () => {
      program.dispatch({ type: 'clear' })
    })
    now.addEventListener('click', () => {
      program.dispatch({ type: 'select', date: clampPredDate(today, min, max) })
    })
    foot.append(clear, now)
    panel.replaceChildren(head, week, grid, foot)
  }

  const onTriggerClick = (): void => {
    if (panel.matches(':popover-open') || panel.dataset.open === 'true') {
      program.dispatch({ type: 'close' })
      return
    }
    options.onOpen()
    const selected = options.selected()
    const selectedParts =
      parsePredDate(selected) ?? parsePredDate(options.max()) ?? predTodayParts()
    program.dispatch({
      type: 'open',
      selected,
      viewMonth: predMonthValue(predMonthFromDate(selectedParts)),
    })
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
    const offsets: Record<string, number> = {
      ArrowLeft: -1,
      ArrowRight: 1,
      ArrowUp: -7,
      ArrowDown: 7,
      Home: -42,
      End: 42,
      PageUp: -31,
      PageDown: 31,
    }
    const offset = offsets[event.key]
    if (offset === undefined) return
    const active = document.activeElement
    if (!(active instanceof HTMLButtonElement) || !active.classList.contains('tri-pred-cal-day'))
      return
    const target = predCalendarFocusTarget(
      active.dataset.date,
      offset,
      panel.dataset.minDate,
      panel.dataset.maxDate,
    )
    if (!target) return
    event.preventDefault()
    program.dispatch({ type: 'focus-date', date: target.date, viewMonth: target.viewMonth })
  }
  const onDateRender = (): void => program.dispatch({ type: 'sync', selected: options.selected() })

  const program = start<DatePickerModel, DatePickerMessage, DatePickerEffect>({
    init: () => ({ model: initialDatePickerModel(), effects: [] }),
    reduce: updateDatePicker,
    effects: effect => {
      if (effect.type === 'notify-select') {
        options.onSelect(effect.date)
        return
      }
      if (effect.type === 'notify-clear') {
        options.onClear()
        return
      }
      if (effect.type === 'close-panel') {
        closePanel(effect.restoreFocus)
        return
      }
      renderPanel()
      positionPredCalendar(trigger, panel)
      if (!panel.matches(':popover-open')) {
        if (typeof panel.showPopover === 'function') panel.showPopover()
        else panel.dataset.open = 'true'
      }
      trigger.setAttribute('aria-expanded', 'true')
      if (effect.focusDate)
        panel
          .querySelector<HTMLButtonElement>(`.tri-pred-cal-day[data-date="${effect.focusDate}"]`)
          ?.focus()
      else focusPredCalendarSelection(panel)
    },
  })

  const close = (): void => program.dispatch({ type: 'close' })
  const render = (): void => program.dispatch({ type: 'sync', selected: options.selected() })
  const mount = (): (() => void) => {
    trigger.addEventListener('click', onTriggerClick)
    panel.addEventListener('toggle', onPanelToggle)
    panel.addEventListener('keydown', onPanelKeydown)
    panel.addEventListener('tri:date-render', onDateRender)
    return () => {
      trigger.removeEventListener('click', onTriggerClick)
      panel.removeEventListener('toggle', onPanelToggle)
      panel.removeEventListener('keydown', onPanelKeydown)
      panel.removeEventListener('tri:date-render', onDateRender)
      program.stop()
    }
  }

  return { wrap, trigger, panel, render, close, mount }
}

export const buildPredDatePicker = (
  formatter: TriathlonFormatter,
  block: HTMLElement,
  onOpen: () => void,
  onSelect: (date: string) => void,
  onClear: () => void,
): PredDatePicker =>
  buildDatePicker({
    id: 'tri-pred-comparison-date',
    formatter,
    label: formatter.text('comparison date'),
    selected: () => block.dataset.compareDate,
    min: () => block.dataset.compareMin,
    max: () => block.dataset.compareMax,
    onOpen,
    onSelect,
    onClear,
  })
