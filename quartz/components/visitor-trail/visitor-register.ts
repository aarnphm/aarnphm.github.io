import { subscribeVisitorCountry } from '../../util/visitor-country'
import { localUtcOffset, visitorLocale, visitorLocaleTitle } from './visitor-locale'
import { anonymousVisitor } from './visitor-name'

// The weather response's country for this browser, per register. Memory only; see visitor-locale.ts.
const weatherCountries = new WeakMap<HTMLElement, string | null>()

const ownLocaleTitle = (root: HTMLElement) => {
  const view = root.ownerDocument.defaultView
  return view ? visitorLocaleTitle(visitorLocale(view, weatherCountries.get(root) ?? null)) : ''
}

export type VisitorVisit = {
  id: string
  own: boolean
  arrived: number
  observed: boolean
  left: number | null
  // The UTC offset the visitor published, e.g. "UTC−4"; null for an older client.
  offset: string | null
}

export function renderVisitorRegister(root: HTMLElement, visits: VisitorVisit[]) {
  const document = root.ownerDocument
  const list = root.querySelector<HTMLOListElement>('[data-visitor-register]')
  const empty = root.querySelector<HTMLElement>('[data-visitor-register-empty]')
  const total = root.querySelector<HTMLElement>('[data-visitor-register-total]')
  if (!list || !empty || !total) return
  const time = new Intl.DateTimeFormat(undefined, {
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  })
  const fullTime = new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'medium' })
  const clock = (at: number, label: 'Arrived' | 'First seen' | 'Left') => {
    const element = document.createElement('time')
    element.dateTime = new Date(at).toISOString()
    element.textContent = time.format(at)
    if (label === 'First seen') {
      const approximate = document.createElement('span')
      approximate.className = 'visitor-register-approximate'
      approximate.textContent = '~'
      approximate.setAttribute('aria-hidden', 'true')
      element.prepend(approximate)
    }
    element.title = `${label} ${fullTime.format(at)}`
    element.setAttribute('aria-label', element.title)
    return element
  }
  const ordered = [...visits].sort(
    (a, b) =>
      Number(a.left !== null) - Number(b.left !== null) ||
      (b.left ?? b.arrived) - (a.left ?? a.arrived),
  )
  list.replaceChildren(
    ...ordered.map(visit => {
      const row = document.createElement('li')
      row.dataset.visitorVisit = visit.id
      row.dataset.visitState = visit.left === null ? 'here' : 'left'
      if (visit.own) row.dataset.visitOwn = ''
      const name = document.createElement('span')
      name.textContent = `${anonymousVisitor(visit.id)}${visit.own ? ' (you)' : ''}`
      name.title = name.textContent
      const departure =
        visit.left === null ? document.createElement('span') : clock(visit.left, 'Left')
      if (visit.left === null) departure.textContent = 'here'
      const offset = document.createElement('span')
      offset.dataset.visitorOffset = ''
      const view = document.defaultView
      offset.textContent = (visit.own && view ? localUtcOffset(view) : visit.offset) ?? '–'
      if (visit.own) offset.title = ownLocaleTitle(root)
      row.append(
        name,
        offset,
        clock(visit.arrived, visit.observed ? 'First seen' : 'Arrived'),
        departure,
      )
      return row
    }),
  )
  empty.hidden = visits.length > 0
  const here = visits.filter(visit => visit.left === null).length
  total.textContent = `${here} here · local time`
  total.title = time.resolvedOptions().timeZone
}

export function mountVisitorRegister(root: HTMLElement, signal: AbortSignal) {
  const view = root.ownerDocument.defaultView
  const toggle = root.querySelector<HTMLButtonElement>('.trail-visitor-count')
  const panel = root.querySelector<HTMLElement>('#homepage-visitor-panel')
  const retry = panel?.querySelector<HTMLButtonElement>('[data-trail-retry]')
  if (!view || !toggle || !panel || !retry)
    throw new Error('Visitor register controls are incomplete')
  retry.addEventListener('click', () => view.location.reload(), { signal })
  subscribeVisitorCountry(
    root.ownerDocument,
    country => {
      weatherCountries.set(root, country)
      const cell = root.querySelector<HTMLElement>('[data-visit-own] [data-visitor-offset]')
      if (cell) cell.title = ownLocaleTitle(root)
    },
    signal,
  )
  const hover = view.matchMedia('(hover: hover) and (pointer: fine)')
  let pinned = false
  let closeTimer = 0
  const cancelClose = () => view.clearTimeout(closeTimer)
  const place = () => {
    if (!panel.matches(':popover-open')) return
    const anchor = toggle.getBoundingClientRect()
    const bounds = panel.getBoundingClientRect()
    panel.style.left = `${Math.max(8, Math.min(view.innerWidth - bounds.width - 8, anchor.right - bounds.width))}px`
    // Hangs 4px under the 16px glyph centred in the toggle, as the weather panel does.
    const below = anchor.top + anchor.height / 2 + 12
    panel.style.top = `${Math.max(8, Math.min(view.innerHeight - bounds.height - 8, below))}px`
  }
  const show = () => {
    cancelClose()
    if (!panel.matches(':popover-open')) panel.showPopover({ source: toggle })
    place()
  }
  const close = () => {
    cancelClose()
    pinned = false
    if (panel.matches(':popover-open')) panel.hidePopover()
  }
  const dismissPanel = () => {
    close()
    toggle.focus({ preventScroll: true })
  }
  toggle.addEventListener(
    'click',
    event => {
      event.preventDefault()
      if (pinned && panel.matches(':popover-open')) close()
      else {
        pinned = true
        show()
      }
    },
    { signal },
  )
  for (const element of [toggle, panel]) {
    element.addEventListener(
      'pointerenter',
      event => {
        if (hover.matches && event.pointerType !== 'touch') show()
      },
      { signal },
    )
    element.addEventListener(
      'pointerleave',
      () => {
        cancelClose()
        closeTimer = view.setTimeout(() => {
          if (
            !pinned &&
            !panel.matches(':hover') &&
            !toggle.matches(':hover') &&
            !panel.contains(root.ownerDocument.activeElement)
          )
            close()
        }, 180)
      },
      { signal },
    )
  }
  panel.addEventListener(
    'toggle',
    () => {
      const open = panel.matches(':popover-open')
      toggle.setAttribute('aria-expanded', String(open))
      if (open) place()
      else pinned = false
    },
    { signal },
  )
  root.addEventListener(
    'keydown',
    event => {
      if (event.key !== 'Escape' || !panel.matches(':popover-open')) return
      event.preventDefault()
      event.stopPropagation()
      dismissPanel()
    },
    { signal },
  )
  view.addEventListener('resize', place, { signal })
  const resize = new ResizeObserver(place)
  resize.observe(panel)
  signal.addEventListener(
    'abort',
    () => {
      close()
      resize.disconnect()
    },
    { once: true },
  )
}
