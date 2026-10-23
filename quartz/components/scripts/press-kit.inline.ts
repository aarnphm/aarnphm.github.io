import type { PressKitPage } from '../PressKit'
import { PressKitEditor } from '../PressKitEditor'
import { TRI_LOCALE_KEY } from '../triathlon/runtime/preferences'
import { currentNavSignal } from './nav-lifecycle'
import { activityCardFromElement } from './press-kit-metrics'
import { mountOverlay, refreshLocale } from './press-kit/control-view'
import {
  bindDialog,
  hidePill,
  onDocumentKeyDown,
  onSelectionSettled,
  open,
  openCalendar,
} from './press-kit/dialog-events'
import { linkCards } from './press-kit/link-sharing'
import { schedule } from './press-kit/picture-sharing'
import {
  $,
  blockDisplays,
  selectedActivities,
  selectedBlocks,
  selectedMetrics,
  session,
  show,
  type PressKitTab,
} from './press-kit/session-state'
import { titleText } from './press-kit/share-subject'
import { layoutMetricControls } from './press-kit/training-controls'

document.addEventListener('nav', () => {
  const signal = currentNavSignal()
  if (session.activeSignal === signal) return
  session.activeSignal = signal
  session.editor = null
  session.drawVersion++
  session.rendering = null

  session.dialog = document.querySelector<HTMLDialogElement>('dialog.press-kit')
  const data = session.dialog?.querySelector('.press-kit-data')?.textContent
  session.page = data ? (JSON.parse(data) as PressKitPage) : null
  session.subject = null
  selectedBlocks.clear()
  selectedActivities.clear()
  blockDisplays.clear()
  selectedMetrics.clear()
  linkCards.clear()
  session.overlay = 'card'
  session.language = null
  show.transparent = false
  show.entry = false
  show.cyclingPower = false
  show.powerCurve = false
  show.cyclingLaps = false
  session.powerWindow = 30
  session.powerCurveScale = 'log'
  show.runPace = false
  show.runLaps = false
  session.anchorSelect = null
  session.unmountAnchor = null
  session.overlaySelect = null
  session.lastRange = null
  session.pendingG = -Infinity
  hidePill()
  signal.addEventListener('abort', hidePill, { once: true })
  if (!session.dialog || !session.page) return

  bindDialog(signal)
  mountOverlay(signal)
  session.dialog.addEventListener('toggle', layoutMetricControls, { capture: true, signal })
  const controls = $('.press-kit-controls')
  const canvas = $<HTMLCanvasElement>('.press-kit-canvas')
  if (controls && canvas)
    session.editor = new PressKitEditor(
      canvas,
      controls,
      signal,
      () => {
        window.clearTimeout(session.blobTimer)
        session.blob = null
      },
      schedule,
      titleText,
    )
  if (controls) {
    let width = 0
    const observer = new ResizeObserver(entries => {
      const next = entries[0]?.contentRect.width ?? 0
      if (next === width) return
      width = next
      layoutMetricControls()
    })
    observer.observe(controls)
    window.addCleanup(() => observer.disconnect())
  }
  document.addEventListener('keydown', onDocumentKeyDown, { signal })
  window.addEventListener('tri:locale', refreshLocale, { signal })
  window.addEventListener(
    'storage',
    event => {
      if (event.key === TRI_LOCALE_KEY) refreshLocale()
    },
    { signal },
  )
  document.addEventListener(
    'presskitopen',
    event => open({ tab: (event as CustomEventMap['presskitopen']).detail?.tab }),
    { signal },
  )
  document.addEventListener(
    'click',
    event => {
      const trigger =
        event.target instanceof Element
          ? event.target.closest<HTMLElement>('[data-press-kit], [data-press-kit-calendar]')
          : null
      if (!trigger || trigger.closest('.press-kit')) return
      event.preventDefault()
      if (trigger.hasAttribute('data-press-kit-calendar')) {
        openCalendar(trigger)
        return
      }
      const entryId = trigger.dataset.pressKitEntry
      const entry = entryId
        ? (trigger.closest<HTMLElement>('.stream-entry') ??
          document.getElementById(entryId) ??
          undefined)
        : undefined
      const activityData = trigger.dataset.pressKitActivity
      const activity = activityCardFromElement(trigger)
      if (activityData && !activity) return
      open({
        tab: (trigger.dataset.pressKit || undefined) as PressKitTab | undefined,
        entry,
        entryUrl: trigger.dataset.pressKitUrl,
        activity: activity ?? undefined,
      })
    },
    { signal },
  )

  let settle = 0
  const settleSoon = (delay: number) => {
    window.clearTimeout(settle)
    settle = window.setTimeout(onSelectionSettled, delay)
  }
  document.addEventListener(
    'mouseup',
    event => {
      // Modifier selections belong to the comment composer.
      if (event.button !== 0 || event.metaKey || event.altKey || event.ctrlKey) return
      if (event.target instanceof Element && event.target.closest('.press-kit, .press-kit-pill'))
        return
      settleSoon(10)
    },
    { signal },
  )
  document.addEventListener(
    'selectionchange',
    () => {
      const selection = window.getSelection()
      if (!selection || selection.isCollapsed) hidePill()
      // Touch selections have no mouseup; wait for the handles to settle.
      else if (matchMedia('(pointer: coarse)').matches) settleSoon(450)
    },
    { signal },
  )
  signal.addEventListener('abort', () => window.clearTimeout(settle), { once: true })
})
