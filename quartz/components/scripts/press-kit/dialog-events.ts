import type { FormatKey, ThemeKey } from '../press-kit-card'
import { isActivityCardStatKey } from '../../../util/press-kit-activity'
import { currentNavSignal } from '../nav-lifecycle'
import { calendarCardData } from '../press-kit-calendar'
import { isBlockDisplayKey } from '../press-kit-day'
import { runsFromText } from '../press-kit-text'
import { isIncludeKey, mountAnchor, refreshLocale, setTab, syncControls } from './control-view'
import { copy } from './link-sharing'
import {
  canShareFiles,
  copyImage,
  download,
  draw,
  loadArt,
  loadFonts,
  resolveFonts,
  schedule,
  shareImage,
} from './picture-sharing'
import {
  $,
  blockDisplays,
  FORMAT_KEYS,
  readPrefs,
  selectedActivities,
  selectedBlocks,
  selectedMetrics,
  session,
  show,
  writePrefs,
  type OpenRequest,
  type PressKitTab,
} from './session-state'
import {
  absolute,
  activeDayBlocks,
  activitySubject,
  availableTabs,
  citationAllowed,
  entrySubject,
  pageSubject,
  quoteFrom,
  selectedRange,
  shareUrl,
  sourceOf,
  subjectActivities,
  syncMetricIncludes,
  titleText,
} from './share-subject'
import { editedPropositions } from './tractatus'
import { layoutMetricControls, mountDayControls, mountMetricControls } from './training-controls'

const THEME_KEYS: ThemeKey[] = ['papier', 'encre', 'rose', 'sauge']

const CHORD_WINDOW_MS = 1000

// A selection survives this long after it collapses, so a button click elsewhere can still quote it.
const RANGE_TTL_MS = 60_000

export function open(request: OpenRequest = {}) {
  if (!session.dialog || !session.page) return
  if (session.dialog.open) {
    if (request.tab) setTab(request.tab)
    return
  }
  readPrefs()
  session.fonts = resolveFonts()
  loadFonts(session.fonts)
  loadArt()

  session.subject = request.calendar
    ? {
        kind: 'calendar',
        calendar: { ...request.calendar, url: absolute(request.calendar.url) },
        url: absolute(request.calendar.url),
        title: runsFromText(request.calendar.title),
        description: [],
        excerpt: [],
        tags: [],
        crumbs: [],
        headings: [],
      }
    : request.activity
      ? activitySubject(request.activity)
      : request.entry
        ? entrySubject(request.entry, request.entryUrl)
        : pageSubject()
  if (request.calendar) session.format = 'portrait'
  mountDayControls()
  mountMetricControls()
  const title = $('#press-kit-title')
  if (title)
    title.textContent =
      session.subject.kind === 'calendar'
        ? 'kit de presse · calendrier'
        : session.subject.kind === 'activity'
          ? 'kit de presse · activité'
          : session.subject.kind === 'day'
            ? 'kit de presse · journée'
            : session.subject.kind === 'entry'
              ? 'kit de presse · entrée'
              : 'kit de presse'
  const recent =
    session.lastRange && performance.now() - session.lastRange.at < RANGE_TTL_MS
      ? session.lastRange.range
      : null
  const candidate = request.range ?? selectedRange() ?? recent
  // An entry card only quotes text from inside that entry.
  const usable =
    citationAllowed() &&
    candidate &&
    sourceOf(candidate.commonAncestorContainer) &&
    (!request.entry || request.entry.contains(candidate.commonAncestorContainer))
      ? candidate
      : null
  session.quote = quoteFrom(usable)
  show.tractatus = Boolean(session.quote.propositions?.length)
  session.anchor = session.quote.fragment ? 'passage' : 'page'
  session.noteEdited = false
  const note = $<HTMLTextAreaElement>('#press-kit-note')
  if (note) note.value = ''
  session.tab = request.tab ?? (usable ? 'quote' : 'note')
  if (!availableTabs().includes(session.tab)) session.tab = 'note'

  mountAnchor(currentNavSignal())
  const shareImageButton = $<HTMLButtonElement>('[data-action="share-image"]')
  if (shareImageButton) shareImageButton.hidden = !canShareFiles()
  const shareLinkButton = $<HTMLButtonElement>('[data-action="share-link"]')
  if (shareLinkButton) shareLinkButton.hidden = typeof navigator.share !== 'function'

  hidePill()
  syncControls()
  session.dialog.showModal()
  layoutMetricControls()
  $<HTMLButtonElement>(`#press-kit-tab-${session.tab}`)?.focus()
  draw()
}

export function openCalendar(trigger: HTMLElement) {
  const calendar = trigger.closest<HTMLElement>('.tri-calendar[data-calendar-year]')
  if (!calendar || session.dialog?.open) return
  const data = calendarCardData(calendar)
  if (!data) return
  trigger.focus({ preventScroll: true })
  open({ calendar: data, tab: 'note' })
}

function cycle<T>(list: T[], value: T, delta: number): T {
  return list[(list.indexOf(value) + delta + list.length) % list.length]
}

function isEditable(target: EventTarget | null): boolean {
  if (!(target instanceof Element)) return false
  return (
    target.matches('input:not([type="radio"]):not([type="checkbox"]), textarea, select') ||
    (target as HTMLElement).isContentEditable ||
    target.closest('[role="listbox"]') !== null
  )
}

function onDialogKeyDown(event: KeyboardEvent) {
  if (event.isComposing || event.metaKey || event.ctrlKey || event.altKey) return
  const { key } = event
  if (key === 'Escape') {
    event.stopPropagation()
    return
  }
  if (event.target instanceof Element && event.target.matches('.press-kit-tab')) {
    if (key === 'ArrowRight' || key === 'ArrowLeft') {
      setTab(cycle(availableTabs(), session.tab, key === 'ArrowRight' ? 1 : -1), true)
      event.preventDefault()
      return
    }
    if (key === 'Home' || key === 'End') {
      setTab(key === 'Home' ? availableTabs()[0] : availableTabs().at(-1)!, true)
      event.preventDefault()
      return
    }
  }
  if (isEditable(event.target)) return
  const imageTab = session.tab !== 'links'
  if (key === '1' || key === '2' || key === '3') {
    const next = availableTabs()[Number(key) - 1]
    if (!next) return
    setTab(next, true)
  } else if (imageTab && (key === 'f' || key === 'F')) {
    session.format = cycle(FORMAT_KEYS, session.format, key === 'F' ? -1 : 1)
    writePrefs()
    syncControls()
    schedule()
  } else if (imageTab && (key === 't' || key === 'T')) {
    session.theme = cycle(THEME_KEYS, session.theme, key === 'T' ? -1 : 1)
    writePrefs()
    syncControls()
    schedule()
  } else if (imageTab && key === 'd') {
    void download()
  } else if (imageTab && key === 'c') {
    void copyImage()
  } else {
    return
  }
  event.preventDefault()
  event.stopPropagation()
}

export function onDocumentKeyDown(event: KeyboardEvent) {
  if (session.dialog?.open) return
  if (event.ctrlKey || event.metaKey || event.altKey || isEditable(event.target)) {
    session.pendingG = -Infinity
    return
  }
  if (event.key === 's' && event.timeStamp - session.pendingG < CHORD_WINDOW_MS) {
    event.preventDefault()
    session.pendingG = -Infinity
    open()
    return
  }
  session.pendingG = event.key === 'g' ? event.timeStamp : -Infinity
}

export function hidePill() {
  session.pill?.remove()
  session.pill = null
}

/** A small "citer" button under a fresh selection, as on Substack and Medium. */
function showPill(range: Range) {
  const rects = range.getClientRects()
  const rect = rects[rects.length - 1] ?? range.getBoundingClientRect()
  if (!rect || (rect.width === 0 && rect.height === 0)) return hidePill()
  if (!session.pill) {
    session.pill = document.createElement('button')
    session.pill.type = 'button'
    session.pill.className = 'press-kit-pill'
    session.pill.textContent = 'citer'
    session.pill.setAttribute('aria-label', 'citer la sélection en image')
    // Keeps the selection alive through the click.
    session.pill.addEventListener('pointerdown', event => event.preventDefault())
    session.pill.addEventListener('click', () =>
      open({ tab: 'quote', range: session.lastRange?.range }),
    )
    document.body.append(session.pill)
  }
  const width = session.pill.offsetWidth || 48
  const left = Math.min(
    Math.max(8, rect.right + window.scrollX - width / 2),
    window.scrollX + document.documentElement.clientWidth - width - 8,
  )
  session.pill.style.left = `${left}px`
  session.pill.style.top = `${rect.bottom + window.scrollY + 8}px`
}

export function onSelectionSettled() {
  if (session.dialog?.open || !citationAllowed()) return
  const range = selectedRange()
  if (!range) {
    hidePill()
    return
  }
  session.lastRange = { range, at: performance.now() }
  showPill(range)
}

export function bindDialog(signal: AbortSignal) {
  const modal = session.dialog!
  modal.addEventListener('keydown', onDialogKeyDown, { signal })
  modal.addEventListener(
    'close',
    () => {
      window.clearTimeout(session.blobTimer)
      if (session.frame) cancelAnimationFrame(session.frame)
      session.frame = 0
    },
    { signal },
  )
  const close = $('.press-kit-close')
  if (close) {
    close.dataset.siteCursorClose = ''
    close.addEventListener('click', () => modal.close(), { signal })
  }
  modal.addEventListener(
    'click',
    event => {
      if (event.target !== modal) return
      const rect = modal.getBoundingClientRect()
      const inside =
        event.clientX >= rect.left &&
        event.clientX <= rect.right &&
        event.clientY >= rect.top &&
        event.clientY <= rect.bottom
      if (!inside) modal.close()
    },
    { signal },
  )
  modal
    .querySelectorAll<HTMLButtonElement>('.press-kit-tab')
    .forEach(button =>
      button.addEventListener('click', () => setTab(button.dataset.tab as PressKitTab), { signal }),
    )
  modal.querySelectorAll<HTMLButtonElement>('[data-include]').forEach(button => {
    button.addEventListener(
      'click',
      () => {
        if (!isIncludeKey(button.name)) return
        show[button.name] = isBlockDisplayKey(button.name)
          ? button.getAttribute('aria-pressed') !== 'true'
          : !show[button.name]
        const displayKey = button.name
        if (isBlockDisplayKey(displayKey))
          for (const display of blockDisplays.values()) display[displayKey] = show[displayKey]
        if (isActivityCardStatKey(button.name)) {
          for (const keys of selectedMetrics.values()) {
            if (show[button.name]) keys.add(button.name)
            else keys.delete(button.name)
          }
        }
        button.setAttribute('aria-pressed', String(show[button.name]))
        syncControls()
        writePrefs()
        schedule()
      },
      { signal },
    )
  })
  modal.addEventListener(
    'change',
    event => {
      const input = event.target
      if (!(input instanceof HTMLInputElement)) return
      if (input.name === 'press-kit-language' && (input.value === 'en' || input.value === 'fr')) {
        session.language = input.value
        refreshLocale()
        return
      }
      if (input.name === 'press-kit-format') session.format = input.value as FormatKey
      else if (
        input.name === 'press-kit-power-window' &&
        (input.value === '30' || input.value === '300')
      ) {
        session.powerWindow = input.value === '30' ? 30 : 300
        schedule()
        return
      } else if (
        input.name === 'press-kit-power-scale' &&
        (input.value === 'linear' || input.value === 'log')
      ) {
        session.powerCurveScale = input.value
        schedule()
        return
      } else if (input.name === 'press-kit-theme') session.theme = input.value as ThemeKey
      else return
      writePrefs()
      schedule()
    },
    { signal },
  )
  $<HTMLTextAreaElement>('#press-kit-quote')?.addEventListener(
    'input',
    event => {
      if (!(event.target instanceof HTMLTextAreaElement)) return
      const value = event.target.value
      const propositions = show.tractatus
        ? editedPropositions(value, session.quote.propositions ?? [])
        : []
      session.quote = {
        ...session.quote,
        runs: runsFromText(value),
        propositions,
        chosen: true,
        fragment: undefined,
      }
      if (!propositions.length) show.tractatus = false
      syncControls()
      if (session.anchor === 'passage')
        session.anchor = session.quote.section ? `#${session.quote.section.id}` : 'page'
      schedule()
    },
    { signal },
  )
  $<HTMLInputElement>('#press-kit-attribution')?.addEventListener(
    'input',
    event => {
      if (!(event.target instanceof HTMLInputElement)) return
      session.quote = { ...session.quote, attribution: event.target.value }
      schedule()
    },
    { signal },
  )
  $<HTMLTextAreaElement>('#press-kit-note')?.addEventListener(
    'input',
    () => {
      session.noteEdited = true
      schedule()
    },
    { signal },
  )
  modal.addEventListener(
    'click',
    event => {
      const target = event.target instanceof Element ? event.target : null
      const displayButton = target?.closest<HTMLButtonElement>('[data-block-display]')
      const displayKey = displayButton?.dataset.blockDisplay
      const displayBlock = displayButton?.dataset.displayBlock
      if (displayButton && displayKey && displayBlock && isBlockDisplayKey(displayKey)) {
        const display = blockDisplays.get(displayBlock)
        if (!display) return
        display[displayKey] = !display[displayKey]
        syncControls()
        schedule()
        return
      }
      const metricButton = target?.closest<HTMLButtonElement>(
        '[data-activity-metric], [data-metric-action]',
      )
      const metricUrl = metricButton?.dataset.metricActivity
      if (metricButton && metricUrl) {
        const activity = subjectActivities().find(activity => activity.url === metricUrl)
        const keys = selectedMetrics.get(metricUrl)
        if (!activity || !keys) return
        const key = metricButton.dataset.activityMetric
        if (key) {
          if (keys.has(key)) keys.delete(key)
          else keys.add(key)
        } else {
          keys.clear()
          if (metricButton.dataset.metricAction === 'all')
            for (const stat of activity.stats) keys.add(stat.key)
        }
        syncMetricIncludes()
        syncControls()
        schedule()
        return
      }
      const activityButton = target?.closest<HTMLButtonElement>('[data-day-activity]')
      const activityUrl = activityButton?.dataset.dayActivity
      if (activityButton && activityUrl) {
        if (selectedActivities.has(activityUrl)) selectedActivities.delete(activityUrl)
        else selectedActivities.add(activityUrl)
        if (
          session.anchor.startsWith('block:') &&
          !activeDayBlocks().some(block => `block:${block.id}` === session.anchor)
        )
          session.anchor = 'page'
        syncControls()
        schedule()
        return
      }
      const blockButton = target?.closest<HTMLButtonElement>('[data-day-block]')
      const blockId = blockButton?.dataset.dayBlock
      if (blockButton && blockId) {
        if (selectedBlocks.has(blockId)) selectedBlocks.delete(blockId)
        else selectedBlocks.add(blockId)
        blockButton.setAttribute('aria-pressed', String(selectedBlocks.has(blockId)))
        syncControls()
        schedule()
        return
      }
      const action = target?.closest<HTMLElement>('[data-action]')?.dataset.action
      const copyKind = target?.closest<HTMLElement>('[data-copy]')
      if (action === 'download') void download()
      else if (action === 'copy-image') void copyImage()
      else if (action === 'share-image') void shareImage()
      else if (action === 'share-link') {
        const text = $<HTMLTextAreaElement>('#press-kit-note')?.value.trim()
        void navigator
          .share({ title: titleText(), text: text || undefined, url: shareUrl() })
          .catch(() => {})
      } else if (copyKind) {
        void copy(copyKind.dataset.copy!, copyKind.textContent!.trim())
      }
    },
    { signal },
  )
}
