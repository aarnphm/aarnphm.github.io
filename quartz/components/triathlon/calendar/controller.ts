import type { TriathlonContext } from '../runtime/context'
import type { TrainingView } from './TrainingCalendar'
import { rootNavSignal } from '../../scripts/root-lifecycle'
import { detailHead } from '../analytics/search'
import { applyI18n, el } from '../runtime/dom'
import {
  calendarDateLabel,
  calendarMonthLabel,
  calendarToday,
  calendarWeekdayLabel,
} from './display'
import { mountTrainingCalendar } from './training'

const configuredCalendars = new WeakMap<HTMLElement, AbortSignal>()
const DAY_MS = 86_400_000

const mountCalendar = (calendar: HTMLElement, context: TriathlonContext): (() => void) => {
  const embedded = calendar.dataset.calendarEmbedded === 'true'
  const inPanel = Boolean(calendar.closest('.tri-calendar-panel'))
  const inSidepanel = Boolean(calendar.closest('.sidepanel-container'))
  const localHash = embedded || inSidepanel
  const year = Number(calendar.dataset.calendarYear)
  const buttons = calendar.querySelectorAll<HTMLButtonElement>('[data-calendar-select]')
  const panels = calendar.querySelectorAll<HTMLElement>('[data-calendar-panel]')
  const views = calendar.querySelector<HTMLElement>('.tri-calendar-views')
  const detail = calendar.querySelector<HTMLElement>('.tri-calendar-detail')
  const pop = calendar.querySelector<HTMLElement>('.tri-calendar-pop')
  let active: HTMLElement | null = null
  let activeX: number | null = null
  let hideTimer: number | undefined
  let restoringFocus = false
  let previewDismissed = false
  let shown: string | null = null
  let returnTo: HTMLElement | null = null

  // Sidepanel imports may rename DOM ids; resolve controls within this calendar instance.
  if (inSidepanel) {
    for (const node of [calendar, ...calendar.querySelectorAll<HTMLElement>('[id]')])
      if (!node.id.startsWith('sidepanel-')) node.id = `sidepanel-${node.id}`
  }
  const title = calendar.querySelector<HTMLElement>('.tri-calendar-title-group :is(h1, h2)[id]')
  if (title) calendar.setAttribute('aria-labelledby', title.id)
  const yearTrigger = calendar.querySelector<HTMLButtonElement>('[data-calendar-year-select]')
  const yearMenu = calendar.querySelector<HTMLElement>('.tri-calendar-season-menu')
  const yearLabel = calendar.querySelector<HTMLElement>('[data-calendar-year-label]')
  const yearValue = calendar.querySelector<HTMLElement>('.tri-calendar-season-value')
  if (yearTrigger && yearMenu && yearLabel && yearValue) {
    yearTrigger.setAttribute('aria-controls', yearMenu.id)
    yearTrigger.setAttribute('aria-labelledby', `${yearLabel.id} ${yearValue.id}`)
    yearMenu.setAttribute('aria-labelledby', yearLabel.id)
  }
  for (const month of calendar.querySelectorAll<HTMLElement>('.tri-calendar-month')) {
    const heading = month.querySelector('h2')
    if (heading) month.setAttribute('aria-labelledby', heading.id)
  }
  for (const button of buttons) {
    const panel = Array.from(panels).find(
      panel => panel.dataset.calendarPanel === button.dataset.calendarSelect,
    )
    if (panel) button.setAttribute('aria-controls', panel.id)
  }
  if (detail)
    for (const button of calendar.querySelectorAll<HTMLButtonElement>('[data-calendar-card-open]'))
      button.setAttribute('aria-controls', detail.id)

  const template = (id: string): HTMLTemplateElement | null =>
    calendar.querySelector<HTMLTemplateElement>(
      `template[data-calendar-card-template="${CSS.escape(id)}"]`,
    )
  const rowFor = (id: string): HTMLElement | null =>
    Array.from(calendar.querySelectorAll<HTMLElement>('[data-calendar-event]')).find(
      row => row.dataset.calendarCard === id,
    ) ?? null

  const cancelHide = (): void => {
    window.clearTimeout(hideTimer)
    hideTimer = undefined
  }
  const hideCard = (): void => {
    cancelHide()
    active?.removeAttribute('aria-describedby')
    active = null
    if (!pop) return
    if (pop.matches(':popover-open')) pop.hidePopover()
    delete pop.dataset.open
    pop.replaceChildren()
  }
  // A narrow calendar shows the detail in place of the views instead of beside them.
  const detailCoversViews = (): boolean =>
    Boolean(shown && views && window.getComputedStyle(views).visibility === 'hidden')
  const updateHash = (hash: string): void => {
    if (localHash) return
    const url = new URL(window.location.href)
    url.hash = inPanel ? `calendar${hash ? `-${hash}` : ''}` : hash
    window.history.replaceState(window.history.state, '', url)
  }
  const viewHash = (): string => `${year}${calendar.dataset.calendarView === 'year' ? '-year' : ''}`

  const focusable = (node: HTMLElement | null | undefined): node is HTMLElement =>
    Boolean(
      node?.isConnected &&
      !node.closest('[inert]') &&
      node.getClientRects().length > 0 &&
      window.getComputedStyle(node).visibility !== 'hidden',
    )
  // The event's own control in the visible view: its race day in the year grid, or its list button.
  const eventControl = (id: string): HTMLElement | null => {
    const key = CSS.escape(id)
    if (calendar.dataset.calendarView !== 'year')
      return (
        rowFor(id)?.querySelector<HTMLElement>(
          '.tri-calendar-event-heading [data-calendar-card-open]',
        ) ?? null
      )
    const year = calendar.querySelector<HTMLElement>('[data-calendar-panel="year"]')
    const target = rowFor(id)?.dataset.calendarEvent
    return (
      year?.querySelector<HTMLElement>(
        `[data-calendar-day][data-race][data-calendar-card~="${key}"]`,
      ) ??
      year?.querySelector<HTMLElement>(`[data-calendar-day][data-calendar-card~="${key}"]`) ??
      (target
        ? (year?.querySelector<HTMLElement>(`[data-calendar-target="${CSS.escape(target)}"]`) ??
          null)
        : null)
    )
  }
  const closeDetail = (restoreFocus = false): void => {
    if (!detail || !shown) return
    const id = shown
    const from = returnTo
    shown = null
    returnTo = null
    delete calendar.dataset.calendarDetail
    detail.setAttribute('aria-hidden', 'true')
    detail.removeAttribute('aria-label')
    detail.replaceChildren()
    for (const node of calendar.querySelectorAll<HTMLElement>('[data-selected]'))
      delete node.dataset.selected
    for (const button of calendar.querySelectorAll<HTMLElement>('[data-calendar-card-open]'))
      button.setAttribute('aria-expanded', 'false')
    if (!restoreFocus) return
    // Focus goes back to what opened the detail, or to the event in whichever view is showing now.
    const target = [from, eventControl(id)].find(focusable)
    restoringFocus = true
    target?.focus()
    restoringFocus = false
  }
  // Both views share one grid cell so switching never changes the calendar's height.
  const select = (view: 'list' | 'year'): void => {
    hideCard()
    if (detailCoversViews()) closeDetail()
    calendar.dataset.calendarView = view
    for (const button of buttons)
      button.setAttribute('aria-pressed', String(button.dataset.calendarSelect === view))
    for (const panel of panels) panel.inert = panel.dataset.calendarPanel !== view
  }

  const fillCountdown = (root: HTMLElement): void => {
    const today = Date.parse(`${calendarToday()}T00:00:00Z`)
    for (const node of root.querySelectorAll<HTMLElement>('[data-card-countdown]')) {
      const days = Math.round(
        (Date.parse(`${node.dataset.cardCountdown}T00:00:00Z`) - today) / DAY_MS,
      )
      const value = node.querySelector('dd')
      node.hidden = !(days >= 0) || !value
      if (value) value.textContent = String(days)
    }
  }
  /**
   * Opens the event beside the views with its whole schedule. A dated trigger scrolls to that day,
   * and the other events on that date stay one click away.
   */
  const openDetail = (
    id: string,
    date: string | null,
    others: readonly string[] = [],
    focus = true,
    from: HTMLElement | null = null,
  ): boolean => {
    const source = template(id)
    const node = source?.content.firstElementChild?.cloneNode(true)
    if (!detail || !source || !(node instanceof HTMLElement)) return false
    // A sibling link lives inside the detail it replaces; keep the control that opened the first one.
    const opener = from && !detail.contains(from) ? from : returnTo
    hideCard()
    closeDetail()
    returnTo = opener
    for (const entry of node.querySelectorAll('[data-card-preview]')) entry.remove()
    const name = source.content.querySelector('.tri-calendar-card-head strong')?.textContent ?? ''
    const { cardStart, cardEnd } = source.dataset
    const { head, back } = detailHead('', name, context.formatter.text('go back'))
    back.dataset.calendarDetailClose = ''
    const when = head.querySelector<HTMLElement>('.tri-pop-date')
    if (when && cardStart) {
      when.dataset.calendarDate = cardStart
      if (cardEnd && cardEnd !== cardStart) when.dataset.calendarEnd = cardEnd
      when.dataset.calendarPart = 'full'
    } else if (when) when.dataset.i18n = 'date pending'
    node.prepend(head)
    const siblings = others
      .map(other => ({
        id: other,
        name: template(other)?.content.querySelector('.tri-calendar-card-head strong')?.textContent,
      }))
      .filter((entry): entry is { id: string; name: string } => Boolean(entry.name))
    if (date && siblings.length > 0) {
      const also = el('p', 'tri-calendar-card-also')
      const label = el('span', undefined, 'also on this day')
      label.dataset.i18n = 'also on this day'
      also.append(label)
      for (const sibling of siblings) {
        const button = el('button', undefined, sibling.name, { type: 'button' })
        button.dataset.calendarDetailOpen = sibling.id
        button.dataset.calendarCardDate = date
        button.dataset.calendarOthers = [id, ...siblings.map(entry => entry.id)]
          .filter(entry => entry !== sibling.id)
          .join(' ')
        also.append(button)
      }
      const anchor = node.querySelector('.tri-calendar-card-meta') ?? head
      anchor.after(also)
    }
    const day = date
      ? Array.from(node.querySelectorAll<HTMLElement>('[data-card-date]')).find(
          entry => entry.dataset.cardDate === date,
        )
      : null
    if (day) day.dataset.cardFocus = 'true'
    shown = id
    calendar.dataset.calendarDetail = id
    detail.replaceChildren(node)
    detail.setAttribute('aria-hidden', 'false')
    detail.setAttribute('aria-label', name)
    fillCountdown(detail)
    localizeIn(detail)
    const row = rowFor(id)
    if (row) row.dataset.selected = 'true'
    for (const button of row?.querySelectorAll<HTMLElement>('[data-calendar-card-open]') ?? [])
      button.setAttribute('aria-expanded', 'true')
    for (const cell of calendar.querySelectorAll<HTMLElement>('[data-calendar-day]'))
      if (cell.dataset.calendarCard?.split(' ').includes(id)) cell.dataset.selected = 'true'
    detail.scrollTop = day ? day.offsetTop - node.offsetTop - 8 : 0
    // Covering the views, the detail starts at their top edge, which can sit above the viewport.
    const top = detail.getBoundingClientRect().top
    if (detailCoversViews() && (top < 0 || top > window.innerHeight - 80))
      detail.scrollIntoView({ block: 'start', behavior: 'instant' })
    if (focus) back.focus({ preventScroll: true })
    return true
  }
  const showEvent = (target: string, source?: HTMLElement, focus = true): boolean => {
    const row = Array.from(calendar.querySelectorAll<HTMLElement>('[data-calendar-event]')).find(
      event => event.dataset.calendarEvent === target,
    )
    const id = row?.dataset.calendarCard
    if (!row || !id) return false
    const date = source?.dataset.calendarCardDate ?? null
    const others = (source?.dataset.calendarCard ?? '')
      .split(' ')
      .filter(other => other && other !== id)
    if (!openDetail(id, date, others, focus, source ?? null)) return false
    if (calendar.dataset.calendarView === 'list' && !detailCoversViews())
      row.scrollIntoView({ block: 'nearest', behavior: 'instant' })
    return true
  }
  const onClick = (event: MouseEvent): void => {
    if (!(event.target instanceof Element)) return
    if (event.target.closest('[data-calendar-detail-close]')) {
      closeDetail(true)
      updateHash(viewHash())
      return
    }
    const sibling = event.target.closest<HTMLElement>('[data-calendar-detail-open]')
    const siblingId = sibling?.dataset.calendarDetailOpen
    if (sibling && siblingId) {
      const others = (sibling.dataset.calendarOthers ?? '').split(' ').filter(Boolean)
      if (openDetail(siblingId, sibling.dataset.calendarCardDate ?? null, others, true, sibling))
        updateHash(rowFor(siblingId)?.dataset.calendarEvent ?? '')
      return
    }
    const button = event.target.closest<HTMLButtonElement>('[data-calendar-select]')
    const selected = button?.dataset.calendarSelect
    if (selected === 'list' || selected === 'year') {
      select(selected)
      updateHash(shown ? (rowFor(shown)?.dataset.calendarEvent ?? '') : viewHash())
      return
    }
    if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey || event.button !== 0)
      return
    const target = event.target.closest<HTMLAnchorElement>('[data-calendar-target]')
    const targetId = target?.dataset.calendarTarget
    if (target && targetId && showEvent(targetId, target)) {
      event.preventDefault()
      updateHash(targetId)
      return
    }
    const row = event.target.closest<HTMLElement>('[data-calendar-event]')
    const opener =
      event.target.closest<HTMLButtonElement>('[data-calendar-card-open]') ??
      (event.target.closest('a, button')
        ? null
        : row?.querySelector<HTMLButtonElement>('[data-calendar-card-open]'))
    const id = row?.dataset.calendarCard
    if (!opener || !row || !id) return
    const date = opener.dataset.calendarCardDate ?? null
    if (shown === id && !date) {
      closeDetail()
      updateHash(viewHash())
    } else if (openDetail(id, date, [], true, opener)) updateHash(row.dataset.calendarEvent ?? '')
  }
  const onHashChange = (): void => {
    if (localHash) return
    const hash = inPanel
      ? window.location.hash.replace(/^#calendar(?:-|#)?/, '')
      : window.location.hash.slice(1)
    if (hash.startsWith('race-') && showEvent(hash, undefined, false)) return
    closeDetail()
    select(hash === 'year' || hash === `${year}-year` ? 'year' : 'list')
  }
  const localizeIn = (root: HTMLElement): void => {
    const locale = context.presentation.locale
    applyI18n(root, context.presentation)
    for (const node of root.querySelectorAll<HTMLElement>('[data-calendar-month]')) {
      node.textContent = calendarMonthLabel(
        year,
        Number(node.dataset.calendarMonth),
        locale,
        node.hasAttribute('data-calendar-short'),
      )
    }
    for (const node of root.querySelectorAll<HTMLElement>('[data-calendar-weekday]')) {
      const label = calendarWeekdayLabel(Number(node.dataset.calendarWeekday), locale)
      node.textContent = label.slice(0, 1)
      node.title = label
    }
    for (const node of root.querySelectorAll<HTMLElement>('[data-calendar-date]')) {
      const date = node.dataset.calendarDate
      const part = node.dataset.calendarPart
      if (date && (part === 'month' || part === 'weekday' || part === 'full' || part === 'short'))
        node.textContent = calendarDateLabel(date, node.dataset.calendarEnd ?? null, part, locale)
    }
    for (const node of root.querySelectorAll<HTMLElement>('[data-calendar-edition]')) {
      node.textContent = context.formatter
        .text('{year} schedule')
        .replace('{year}', node.dataset.calendarEdition ?? '')
    }
    for (const node of root.querySelectorAll<HTMLElement>('[data-calendar-date-label]')) {
      const date = node.dataset.calendarDateLabel
      if (date)
        node.setAttribute(
          'aria-label',
          calendarDateLabel(date, node.dataset.calendarEnd ?? null, 'full', locale),
        )
    }
    for (const node of root.querySelectorAll<HTMLElement>('[data-calendar-day]')) {
      const date = node.dataset.calendarDay
      if (date)
        node.setAttribute(
          'aria-label',
          `${calendarDateLabel(date, null, 'full', locale)}: ${node.dataset.calendarEventName ?? ''}`,
        )
    }
  }
  const localize = (): void => {
    localizeIn(calendar)
    if (pop && active) localizeIn(pop)
    detail
      ?.querySelector('[data-calendar-detail-close]')
      ?.setAttribute('aria-label', context.formatter.text('go back'))
  }

  /**
   * A dated trigger shows that day's schedule and keeps the course only on the event's race days.
   * Several ids without a date (a month holding more than one race) reduce each card to its header.
   */
  const card = (id: string, date: string | null, brief: boolean): Element | null => {
    const source = template(id)
    const node = source?.content.firstElementChild?.cloneNode(true)
    if (!source || !(node instanceof Element)) return null
    for (const entry of node.querySelectorAll('[data-card-detail]')) entry.remove()
    const { cardStart, cardEnd } = source.dataset
    const raceDay = !date || Boolean(cardStart && cardEnd && date >= cardStart && date <= cardEnd)
    const days = Array.from(node.querySelectorAll<HTMLElement>('[data-card-date]'))
    const day = brief
      ? null
      : date
        ? days.find(entry => entry.dataset.cardDate === date)
        : days.find(entry => entry.dataset.cardRace === 'true')
    for (const entry of days) if (entry !== day) entry.remove()
    const keepCourse = !brief && raceDay
    if (!keepCourse) for (const entry of node.querySelectorAll('[data-card-course]')) entry.remove()
    if (brief) node.querySelector('.tri-calendar-card-pending')?.remove()
    return node
  }
  const place = (trigger: HTMLElement, pointerX: number | null): void => {
    if (!pop) return
    const anchor = trigger.getBoundingClientRect()
    const gap = 6
    // The card leaves its trigger uncovered so the pointer can still reach and click it.
    const available = Math.max(anchor.top, window.innerHeight - anchor.bottom) - gap - 8
    pop.style.maxHeight = `${Math.max(0, Math.min(window.innerHeight - 16, available))}px`
    const box = pop.getBoundingClientRect()
    const x = pointerX ?? anchor.left
    const left = Math.min(Math.max(8, x - 12), window.innerWidth - box.width - 8)
    const below = anchor.bottom + gap
    const above = anchor.top - gap - box.height
    const top =
      below + box.height <= window.innerHeight - 8
        ? below
        : above >= 8
          ? above
          : window.innerHeight - anchor.bottom > anchor.top
            ? Math.min(below, window.innerHeight - box.height - 8)
            : 8
    pop.style.left = `${Math.max(8, left).toFixed(0)}px`
    pop.style.top = `${Math.max(8, Math.min(top, window.innerHeight - box.height - 8)).toFixed(0)}px`
  }
  const showCard = (trigger: HTMLElement, pointerX: number | null): void => {
    if (!pop || trigger === active) return
    const ids = (trigger.dataset.calendarCard ?? '').split(' ').filter(Boolean)
    // The open detail already shows this event.
    if (ids.length === 1 && ids[0] === shown) return hideCard()
    cancelHide()
    const date = trigger.dataset.calendarCardDate ?? null
    const cards = ids
      .map(id => card(id, date, ids.length > 1 && !date))
      .filter((node): node is Element => node !== null)
    if (cards.length === 0) return hideCard()
    active?.removeAttribute('aria-describedby')
    active = trigger
    pop.replaceChildren(...cards)
    localizeIn(pop)
    if (typeof pop.showPopover === 'function') {
      if (!pop.matches(':popover-open')) pop.showPopover()
    } else pop.dataset.open = 'true'
    activeX = pointerX
    place(trigger, pointerX)
    trigger.setAttribute('aria-describedby', pop.id)
  }
  const trigger = (target: EventTarget | null): HTMLElement | null =>
    target instanceof Element ? (target.closest<HTMLElement>('[data-calendar-card]') ?? null) : null
  const inCard = (target: EventTarget | null): boolean =>
    Boolean(pop && target instanceof Node && pop.contains(target))
  const onPointerOver = (event: MouseEvent): void => {
    if (previewDismissed) return
    if (inCard(event.target)) return cancelHide()
    const next = trigger(event.target)
    if (!next || !calendar.contains(next)) return
    cancelHide()
    showCard(next, event.clientX)
  }
  // Hiding a card exposes the row underneath and emits mouseover without pointer movement.
  const onPointerMove = (event: MouseEvent): void => {
    if (!previewDismissed) return
    previewDismissed = false
    onPointerOver(event)
  }
  // A card taller than the viewport scrolls, so the pointer may cross the gap into it; the short
  // delay keeps the card open on the way.
  const onPointerOut = (event: MouseEvent): void => {
    if (!active || inCard(event.relatedTarget)) return
    const next = trigger(event.relatedTarget)
    if (next === active || (next && calendar.contains(next))) return
    cancelHide()
    hideTimer = window.setTimeout(hideCard, 180)
  }
  const onFocusIn = (event: FocusEvent): void => {
    if (restoringFocus || inCard(event.target)) return
    const next = trigger(event.target)
    if (next) showCard(next, null)
    else hideCard()
  }
  const onFocusOut = (event: FocusEvent): void => {
    if (restoringFocus) return
    const next = event.relatedTarget
    if (next instanceof Node && (inCard(next) || active?.contains(next))) return
    if (!(next instanceof Node && calendar.contains(next))) hideCard()
  }
  // Scrolls elsewhere on the page leave the card alone; the card follows its own trigger.
  const onScroll = (event: Event): void => {
    if (!active) return
    const target = event.target
    if (target instanceof Element && !target.contains(active)) return
    const anchor = active.getBoundingClientRect()
    if (anchor.bottom < 0 || anchor.top > window.innerHeight) hideCard()
    else place(active, activeX)
  }
  // Escape dismisses the hover card first; from inside the calendar it then closes the detail.
  const onKeyDown = (event: KeyboardEvent): void => {
    if (event.key !== 'Escape') return
    if (active) {
      event.preventDefault()
      event.stopPropagation()
      previewDismissed = true
      hideCard()
      return
    }
    if (!shown || event.currentTarget !== calendar) return
    event.preventDefault()
    event.stopPropagation()
    closeDetail(true)
    updateHash(viewHash())
  }
  const onResize = (): void => {
    if (active) place(active, activeX)
  }
  const today = calendarToday()
  let hasNext = false
  for (const row of calendar.querySelectorAll<HTMLElement>('[data-calendar-event]')) {
    const next: boolean =
      !hasNext && Boolean(row.dataset.calendarStart && row.dataset.calendarStart >= today)
    const label = row.querySelector<HTMLElement>('.tri-calendar-next')
    if (label) label.hidden = !next
    if (next) row.dataset.calendarNext = 'true'
    else delete row.dataset.calendarNext
    hasNext ||= next
  }
  calendar.addEventListener('click', onClick)
  calendar.addEventListener('mouseover', onPointerOver)
  calendar.addEventListener('mousemove', onPointerMove)
  calendar.addEventListener('mouseout', onPointerOut)
  calendar.addEventListener('focusin', onFocusIn)
  calendar.addEventListener('focusout', onFocusOut)
  calendar.addEventListener('keydown', onKeyDown)
  document.addEventListener('keydown', onKeyDown)
  window.addEventListener('resize', onResize)
  window.addEventListener('scroll', onScroll, { capture: true, passive: true })
  window.addEventListener('hashchange', onHashChange)
  window.addEventListener('tri:locale', localize)
  localize()
  select(calendar.dataset.calendarView === 'year' ? 'year' : 'list')
  onHashChange()
  return () => {
    hideCard()
    closeDetail()
    calendar.removeEventListener('click', onClick)
    calendar.removeEventListener('mouseover', onPointerOver)
    calendar.removeEventListener('mousemove', onPointerMove)
    calendar.removeEventListener('mouseout', onPointerOut)
    calendar.removeEventListener('focusin', onFocusIn)
    calendar.removeEventListener('focusout', onFocusOut)
    calendar.removeEventListener('keydown', onKeyDown)
    document.removeEventListener('keydown', onKeyDown)
    window.removeEventListener('resize', onResize)
    window.removeEventListener('scroll', onScroll, { capture: true })
    window.removeEventListener('hashchange', onHashChange)
    window.removeEventListener('tri:locale', localize)
  }
}

const mountCalendarSet = (root: HTMLElement, context: TriathlonContext): (() => void) => {
  const calendars = Array.from(root.querySelectorAll<HTMLElement>('[data-calendar-year]'))
  let active = calendars.find(calendar => !calendar.hidden)
  const localHash =
    root.dataset.calendarEmbedded === 'true' || Boolean(root.closest('.sidepanel-container'))
  const inPanel = Boolean(root.closest('.tri-calendar-panel'))
  let dispose: (() => void) | undefined
  let source: 'races' | 'training' =
    root.dataset.calendarSource === 'training' ? 'training' : 'races'
  // The overlay keeps the source switch in its panel bar, outside the calendar set.
  const sourceScope = root.closest<HTMLElement>('.tri-calendar-panel') ?? root
  const sourceButtons = Array.from(
    sourceScope.querySelectorAll<HTMLButtonElement>('[data-calendar-source-select]'),
  )
  const sourcePanels = Array.from(
    root.querySelectorAll<HTMLElement>('[data-calendar-source-panel]'),
  )
  const trainingRoot = root.querySelector<HTMLElement>('[data-training-calendar]')
  const updateHash = (hash: string): void => {
    if (localHash) return
    const url = new URL(window.location.href)
    url.hash = `${inPanel ? 'calendar-' : ''}${hash}`
    window.history.replaceState(window.history.state, '', url)
  }
  const training = trainingRoot ? mountTrainingCalendar(trainingRoot, context, updateHash) : null

  if (root.closest('.sidepanel-container'))
    for (const panel of sourcePanels)
      if (!panel.id.startsWith('sidepanel-')) panel.id = `sidepanel-${panel.id}`
  for (const button of sourceButtons) {
    const panel = sourcePanels.find(
      panel => panel.dataset.calendarSourcePanel === button.dataset.calendarSourceSelect,
    )
    if (panel) button.setAttribute('aria-controls', panel.id)
  }

  const selectSource = (
    next: 'races' | 'training',
    navigate = false,
    date?: string,
    view?: TrainingView,
  ): void => {
    if (navigate && next === source) return
    source = next
    root.dataset.calendarSource = next
    for (const button of sourceButtons)
      button.setAttribute('aria-pressed', String(button.dataset.calendarSourceSelect === next))
    for (const panel of sourcePanels) {
      panel.hidden = panel.dataset.calendarSourcePanel !== next
      panel.inert = panel.hidden
    }
    if (next === 'training') {
      dispose?.()
      dispose = undefined
      training?.activate(date, view)
      if (navigate && training) updateHash(training.location())
    } else if (active) {
      training?.deactivate()
      if (navigate)
        updateHash(
          `${active.dataset.calendarYear}${active.dataset.calendarView === 'year' ? '-year' : ''}`,
        )
      dispose ??= mountCalendar(active, context)
    }
  }
  const localizeSources = (): void => {
    const controls = sourceScope.querySelector<HTMLElement>('.tri-calendar-source-controls')
    if (controls) applyI18n(controls, context.presentation)
  }
  const onSourceClick = (event: MouseEvent): void => {
    if (!(event.currentTarget instanceof HTMLButtonElement)) return
    const selected = event.currentTarget.dataset.calendarSourceSelect
    if (selected !== 'races' && selected !== 'training') return
    closeYearPickers()
    selectSource(selected, true)
  }

  const closeYearPicker = (picker: HTMLElement, restoreFocus = false): void => {
    const menu = picker.querySelector<HTMLElement>('.tri-calendar-season-menu')
    const trigger = picker.querySelector<HTMLButtonElement>('[data-calendar-year-select]')
    if (menu) menu.hidden = true
    trigger?.setAttribute('aria-expanded', 'false')
    if (restoreFocus) trigger?.focus({ preventScroll: true })
  }
  const closeYearPickers = (): void => {
    for (const picker of root.querySelectorAll<HTMLElement>('.tri-calendar-season-picker'))
      closeYearPicker(picker)
  }
  const focusYearOption = (picker: HTMLElement, option: HTMLButtonElement): void => {
    for (const candidate of picker.querySelectorAll<HTMLButtonElement>(
      '[data-calendar-year-option]',
    ))
      candidate.tabIndex = candidate === option ? 0 : -1
    option.focus({ preventScroll: true })
  }
  const openYearPicker = (picker: HTMLElement): void => {
    const menu = picker.querySelector<HTMLElement>('.tri-calendar-season-menu')
    const trigger = picker.querySelector<HTMLButtonElement>('[data-calendar-year-select]')
    if (!menu || !trigger) return
    closeYearPickers()
    menu.hidden = false
    trigger.setAttribute('aria-expanded', 'true')
    const selected = menu.querySelector<HTMLButtonElement>('[aria-selected="true"]')
    if (selected) focusYearOption(picker, selected)
  }
  const activate = (year: string, fromSelect = false): void => {
    const next = calendars.find(calendar => calendar.dataset.calendarYear === year)
    if (!next) return
    closeYearPickers()
    if (next === active) {
      if (fromSelect)
        next
          .querySelector<HTMLButtonElement>('[data-calendar-year-select]')
          ?.focus({ preventScroll: true })
      return
    }
    const view = active?.dataset.calendarView === 'year' ? 'year' : 'list'
    dispose?.()
    for (const calendar of calendars) {
      calendar.hidden = calendar !== next
      calendar.inert = calendar !== next
    }
    next.dataset.calendarView = view
    if (fromSelect && !localHash) {
      const url = new URL(window.location.href)
      url.hash = `${inPanel ? 'calendar-' : ''}${year}${view === 'year' ? '-year' : ''}`
      window.history.replaceState(window.history.state, '', url)
    }
    active = next
    if (source === 'races') dispose = mountCalendar(next, context)
    if (fromSelect)
      next
        .querySelector<HTMLButtonElement>('[data-calendar-year-select]')
        ?.focus({ preventScroll: true })
  }
  const onHashChange = (): void => {
    if (localHash) return
    const hash = inPanel
      ? window.location.hash.replace(/^#calendar(?:-|#)?/, '')
      : window.location.hash.slice(1)
    const trainingHash = /^training(?:-(\d{4}-\d{2}-\d{2}))?(?:-(list|week|month))?$/.exec(hash)
    if (trainingHash) {
      selectSource(
        'training',
        false,
        trainingHash[1],
        trainingHash[2] === 'list' || trainingHash[2] === 'week' || trainingHash[2] === 'month'
          ? trainingHash[2]
          : undefined,
      )
      return
    }
    selectSource('races')
    const year = /^(\d{4})(?:-year)?$/.exec(hash)?.[1] ?? /(?:^race-.+)-(\d{4})$/.exec(hash)?.[1]
    activate(year ?? root.dataset.calendarDefaultYear ?? '')
  }
  const onYearClick = (event: MouseEvent): void => {
    if (!(event.target instanceof Element)) return
    const picker = event.target.closest<HTMLElement>('.tri-calendar-season-picker')
    if (!picker) return
    const option = event.target.closest<HTMLButtonElement>('[data-calendar-year-option]')
    if (option?.dataset.calendarYearOption) {
      activate(option.dataset.calendarYearOption, true)
      return
    }
    if (!event.target.closest('[data-calendar-year-select]')) return
    if (picker.querySelector<HTMLElement>('.tri-calendar-season-menu')?.hidden)
      openYearPicker(picker)
    else closeYearPicker(picker)
  }
  const onYearKeydown = (event: KeyboardEvent): void => {
    if (!(event.target instanceof Element)) return
    const picker = event.target.closest<HTMLElement>('.tri-calendar-season-picker')
    if (!picker) return
    const menu = picker.querySelector<HTMLElement>('.tri-calendar-season-menu')
    if (!menu) return
    if (event.key === 'Escape' && !menu.hidden) {
      event.preventDefault()
      event.stopPropagation()
      closeYearPicker(picker, true)
      return
    }
    if (event.target.closest('[data-calendar-year-select]')) {
      if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return
      event.preventDefault()
      openYearPicker(picker)
      return
    }
    if (menu.hidden) return
    const options = Array.from(
      menu.querySelectorAll<HTMLButtonElement>('[data-calendar-year-option]'),
    )
    const current = options.findIndex(option => option === document.activeElement)
    const index =
      event.key === 'Home'
        ? 0
        : event.key === 'End'
          ? options.length - 1
          : event.key === 'ArrowDown'
            ? Math.min(options.length - 1, current + 1)
            : event.key === 'ArrowUp'
              ? Math.max(0, current - 1)
              : -1
    const option = options[index]
    if (!option) return
    event.preventDefault()
    focusYearOption(picker, option)
  }
  const onYearFocusout = (event: FocusEvent): void => {
    if (!(event.target instanceof Element)) return
    const picker = event.target.closest<HTMLElement>('.tri-calendar-season-picker')
    if (!picker || (event.relatedTarget instanceof Node && picker.contains(event.relatedTarget)))
      return
    closeYearPicker(picker)
  }
  const onPointerdown = (event: PointerEvent): void => {
    for (const picker of root.querySelectorAll<HTMLElement>('.tri-calendar-season-picker'))
      if (!event.composedPath().includes(picker)) closeYearPicker(picker)
  }
  for (const button of sourceButtons) button.addEventListener('click', onSourceClick)
  root.addEventListener('click', onYearClick)
  root.addEventListener('keydown', onYearKeydown, true)
  root.addEventListener('focusout', onYearFocusout)
  document.addEventListener('pointerdown', onPointerdown)
  window.addEventListener('hashchange', onHashChange)
  const unsubscribe = context.events.subscribe('presentation', localizeSources)
  localizeSources()
  if (localHash) selectSource(source)
  else onHashChange()
  return () => {
    dispose?.()
    training?.dispose()
    unsubscribe()
    closeYearPickers()
    for (const button of sourceButtons) button.removeEventListener('click', onSourceClick)
    root.removeEventListener('click', onYearClick)
    root.removeEventListener('keydown', onYearKeydown, true)
    root.removeEventListener('focusout', onYearFocusout)
    document.removeEventListener('pointerdown', onPointerdown)
    window.removeEventListener('hashchange', onHashChange)
  }
}

export const setupCalendars = (context: TriathlonContext): (() => void) => {
  const mounted = new Map<HTMLElement, () => void>()
  const mount = (): void => {
    for (const calendar of document.querySelectorAll<HTMLElement>('[data-calendar-set]')) {
      if (mounted.has(calendar)) continue
      const signal = rootNavSignal(calendar)
      if (configuredCalendars.get(calendar) === signal) continue
      configuredCalendars.set(calendar, signal)
      const dispose = mountCalendarSet(calendar, context)
      const cleanup = (): void => {
        signal.removeEventListener('abort', cleanup)
        dispose()
        mounted.delete(calendar)
        if (configuredCalendars.get(calendar) === signal) configuredCalendars.delete(calendar)
      }
      mounted.set(calendar, cleanup)
      signal.addEventListener('abort', cleanup, { once: true })
    }
  }
  mount()
  document.addEventListener('contentdecrypted', mount)
  return () => {
    document.removeEventListener('contentdecrypted', mount)
    for (const cleanup of mounted.values()) cleanup()
    mounted.clear()
  }
}
