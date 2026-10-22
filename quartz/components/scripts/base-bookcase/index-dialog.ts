import type { BookcaseScene } from './scene'
import { registerEscapeHandler } from '../escape-handler'

export function bindIndex(s: BookcaseScene) {
  s.selectFromIndex = (event: Event) => {
    if (!(event.target instanceof Element)) return
    const row = event.target.closest<HTMLLIElement>('.bookcase-entry')
    if (!row || Number(row.dataset.bookIndex) === s.focus) return
    const index = Number(row.dataset.bookIndex)
    s.focusCamera(index)
    s.select(index)
  }

  s.indexPane.addEventListener('focusin', s.selectFromIndex, { signal: s.signal })

  s.hoverFromIndex = (event: PointerEvent) => {
    if (!event.shiftKey || event.pointerType === 'touch' || event.buttons !== 0 || s.dialog.open)
      return
    s.selectFromIndex(event)
  }

  s.indexPane.addEventListener('pointerover', s.hoverFromIndex, { signal: s.signal })

  s.indexPane.addEventListener('pointermove', s.hoverFromIndex, { signal: s.signal })

  s.groupEntries = new Map(
    s.groups.map(group => [
      group,
      {
        entries: [...group.querySelectorAll<HTMLLIElement>('.bookcase-entry')],
        count: group.querySelector<HTMLElement>('.bookcase-index-count'),
      },
    ]),
  )

  s.disposeEscape = registerEscapeHandler(s.dialog, () => s.dialog.close())

  s.dialog.addEventListener(
    'click',
    event => {
      if (event.target !== s.dialog) return
      const bounds = s.dialog.getBoundingClientRect()
      if (
        event.clientX < bounds.left ||
        event.clientX > bounds.right ||
        event.clientY < bounds.top ||
        event.clientY > bounds.bottom
      )
        s.dialog.close()
    },
    { signal: s.signal },
  )

  s.dialog.addEventListener(
    'close',
    () => {
      s.hardcover?.clear()
      const index = s.previewIndex
      s.previewIndex = -1
      if (
        !s.signal.aborted &&
        s.root.isConnected &&
        s.matches().some(row => Number(row.dataset.bookIndex) === index)
      ) {
        s.focusCamera(index)
        s.select(index, true)
      }
      const fallback = s.entryByIndex
        .get(s.focus)
        ?.querySelector<HTMLAnchorElement>('.bookcase-entry-link')
      const target = s.returnFocus?.isConnected ? s.returnFocus : fallback
      target?.focus({ preventScroll: true })
    },
    { signal: s.signal },
  )

  s.dialog.addEventListener(
    'wheel',
    event => {
      if (event.ctrlKey || event.metaKey || !s.dialog.open) return
      const delta = Math.abs(event.deltaY) >= Math.abs(event.deltaX) ? event.deltaY : event.deltaX
      if (!delta) return
      const body =
        event.target instanceof Element ? event.target.closest('.bookcase-preview-body') : null
      if (
        body instanceof HTMLElement &&
        ((delta > 0 && body.scrollTop + body.clientHeight < body.scrollHeight - 1) ||
          (delta < 0 && body.scrollTop > 0))
      )
        return
      event.preventDefault()
      const now = performance.now()
      if (now - s.previewWheelStep < 220) return
      if (now - s.previewWheelAt > 160 || Math.sign(delta) !== Math.sign(s.previewWheelDelta))
        s.previewWheelDelta = 0
      s.previewWheelAt = now
      s.previewWheelDelta += delta * s.wheelUnit(event, s.dialog.clientHeight)
      if (Math.abs(s.previewWheelDelta) < 64) return
      s.browsePreview(Math.sign(s.previewWheelDelta))
      s.previewWheelDelta = 0
      s.previewWheelStep = now
    },
    { passive: false, signal: s.signal },
  )

  s.dialog.addEventListener(
    'keydown',
    event => {
      if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
        event.preventDefault()
        s.browsePreview(event.key === 'ArrowLeft' ? -1 : 1)
      }
    },
    { signal: s.signal },
  )

  s.refresh = () => {
    if (!s.root.isConnected) return
    const searching = Boolean(
      document.querySelector<HTMLInputElement>('#base-search-input')?.value.trim(),
    )
    if (searching && !s.searchOpen)
      s.searchOpen = new Map(s.groups.map(group => [group, group.open]))
    for (const group of s.groups) {
      const info = s.groupEntries.get(group)
      const shown =
        info?.entries.filter(
          entry => entry.style.display !== 'none' && !entry.hasAttribute('data-topic-hidden'),
        ) ?? []
      if (info?.count) info.count.textContent = String(shown.length)
      if (searching && shown.length) group.open = true
      else if (!searching && s.searchOpen) group.open = s.searchOpen.get(group) ?? true
    }
    if (!searching) s.searchOpen = undefined
    s.visibleEntries = undefined
    const visible = s.matches()
    const rearranged = s.arrange(visible)
    // A new topic or search keeps the chosen book only while it stays in view.
    const selected = visible.find(entry => Number(entry.dataset.bookIndex) === s.focus)
    s.root.toggleAttribute('data-empty', visible.length === 0)
    if (selected) {
      if (rearranged) s.focusCamera(s.focus)
      s.select(s.focus)
    } else s.deselect()
    if (s.dialog.open) {
      if (visible.some(entry => Number(entry.dataset.bookIndex) === s.previewIndex))
        s.renderPreview()
      else s.dialog.close()
    }
  }

  document.addEventListener('base-search-results', s.refresh, { signal: s.signal })
}
