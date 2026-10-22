import type { BookcaseScene } from './scene'
import { PULL_DELAY_MS } from './camera'
import { PULL_MS } from './state'

const PRELOAD = 3

export const SETTLE_MS = 180

export function mountSelection(s: BookcaseScene) {
  s.copyDetail = (row: HTMLLIElement, target: Element | null) => {
    const detail = row.querySelector('.bookcase-entry-detail')
    target?.replaceChildren(...[...(detail?.childNodes ?? [])].map(node => node.cloneNode(true)))
  }

  s.preloadCovers = (visible: HTMLLIElement[], position: number) => {
    const urls = new Set<string>()
    for (let i = position - PRELOAD; i <= position + PRELOAD; i++) {
      const neighbor = visible[i]
      const url = neighbor
        ? s.books.get(Number(neighbor.dataset.bookIndex))?.dataset.cover
        : undefined
      if (url) urls.add(url)
    }
    for (const url of s.preloaded.keys()) if (!urls.has(url)) s.preloaded.delete(url)
    for (const url of urls) {
      if (s.preloaded.has(url)) continue
      const image = new Image()
      image.decoding = 'async'
      image.src = url
      s.preloaded.set(url, image)
    }
  }

  s.select = (index: number, reveal = false) => {
    const row = s.entryByIndex.get(index)
    const book = s.books.get(index)
    if (!row || !book) return
    if (reveal) {
      const group = row.closest('details')
      if (group instanceof HTMLDetailsElement) group.open = true
      const bounds = row.getBoundingClientRect()
      const pane = s.indexPane.getBoundingClientRect()
      const top = bounds.top - pane.top - pane.height / 2 + bounds.height / 2
      // Scroll before changing the drawing, so scrolling cannot force its new layout synchronously.
      s.indexPane.scrollBy({ top, behavior: s.reducedMotion.matches ? 'instant' : 'smooth' })
    }
    const previous = s.entryByIndex.get(s.focus)
    if (previous !== row) {
      previous?.removeAttribute('data-current')
      previous?.querySelector('.bookcase-entry-link')?.removeAttribute('aria-current')
    }
    s.focus = index
    row.setAttribute('data-current', '')
    row.querySelector('.bookcase-entry-link')?.setAttribute('aria-current', 'true')
    const groupLabel = row.closest('.bookcase-index-group')?.querySelector('.bookcase-index-label')
    if (s.shelfLabel)
      s.shelfLabel.textContent = (s.topic ? row.dataset.topicShelf : groupLabel?.textContent) ?? ''
    const visible = s.matches()
    const position = visible.indexOf(row)
    if (s.readout)
      s.readout.textContent = `${String(position + 1).padStart(3, '0')} / ${visible.length}`
    if (
      !s.dialog.open &&
      (!s.lifted ||
        s.lifted.querySelector<SVGGElement>('.bc-book')?.dataset.bookIndex !== String(index))
    )
      s.lift(index)
    s.preloadCovers(visible, position)
  }

  // Nothing is chosen until hover, a click or the keyboard picks a book; scrolling only moves the

  // Nothing is chosen until hover, a click or the keyboard picks a book; scrolling only moves the
  // camera or the list. Escape, a click on the empty stage or leaving the bookcase lowers it.

  // Nothing is chosen until hover, a click or the keyboard picks a book; scrolling only moves the
  // camera or the list. Escape, a click on the empty stage or leaving the bookcase lowers it.
  s.deselect = () => {
    s.cancelWheel()
    const row = s.entryByIndex.get(s.focus)
    row?.removeAttribute('data-current')
    row?.querySelector('.bookcase-entry-link')?.removeAttribute('aria-current')
    s.focus = -1
    s.lower()
    if (s.readout) s.readout.textContent = `– / ${s.matches().length}`
    if (s.shelfLabel) s.shelfLabel.textContent = ''
  }

  s.focusCamera = (index: number) => {
    s.cancelWheel()
    s.cameraDirty = true
    const book = s.books.get(index)
    if (!book) return
    const scale = Math.max(
      s.fitScale,
      Math.min(1.25, s.viewportHeight / 600, s.viewportWidth / (s.fitWidth + 80)),
    )
    const y = Number(book.dataset.cy) - s.originY
    const next = {
      x:
        s.box.width * scale <= s.viewportWidth
          ? (s.viewportWidth - s.box.width * scale) / 2
          : s.viewportWidth / 2 - (Number(book.dataset.cx) - s.originX) * scale,
      y: s.viewportHeight / 2 - y * scale,
      scale,
    }
    s.fitted = false
    if (!s.reducedMotion.matches) {
      if (!s.aim) s.steppedAt = performance.now()
      s.aim = next
      // The book waits for the view to close in before it slides out.
      if (s.pull && Number(s.pull.currentTime ?? 0) < 16) {
        s.pull.effect?.updateTiming({ delay: PULL_DELAY_MS })
        s.reveal?.effect?.updateTiming({ delay: PULL_DELAY_MS + PULL_MS * 0.45 })
      }
    } else {
      s.aim = undefined
      s.camera = next
    }
    s.schedule()
  }

  s.renderPreview = (enter = false) => {
    const row = s.entryByIndex.get(s.previewIndex)
    if (!row) return
    const visible = s.matches()
    const position = visible.indexOf(row)
    s.preloadCovers(visible, position)
    const link = row.querySelector<HTMLAnchorElement>('.bookcase-entry-link')
    const previewLink = s.dialog.querySelector<HTMLAnchorElement>('.bookcase-preview-link')
    const title = row.querySelector('.bookcase-entry-title')?.textContent ?? ''
    s.dialog.setAttribute('aria-label', `Preview: ${title}`)
    const heading = s.dialog.querySelector('.bookcase-preview-title')
    if (heading) heading.textContent = title
    const byline = s.dialog.querySelector('.bookcase-preview-byline')
    if (byline) byline.textContent = row.querySelector('.bookcase-entry-byline')?.textContent ?? ''
    const positionLabel = s.dialog.querySelector('.bookcase-preview-position')
    if (positionLabel) positionLabel.textContent = `${position + 1} / ${visible.length}`
    if (s.previousButton) s.previousButton.disabled = position <= 0
    if (s.nextButton) s.nextButton.disabled = position < 0 || position >= visible.length - 1
    if (previewLink && link) {
      previewLink.href = link.href
      previewLink.dataset.slug = link.dataset.slug ?? ''
    }
    s.copyDetail(row, s.dialog.querySelector('.bookcase-preview-detail'))
    const book = s.books.get(s.previewIndex)
    if (book) {
      const style = book.style
      void s.hardcover
        ?.show({
          cover: book.dataset.fullCover ?? book.dataset.cover,
          thumbnail: book.dataset.cover,
          spine: style.getPropertyValue('--spine').trim(),
          ink: style.getPropertyValue('--spine-ink').trim(),
          title,
          author: row.querySelector('.bookcase-entry-byline')?.textContent ?? '',
          spineArt: book.dataset.spineArt,
          spineArtDetail: book.dataset.spineArtDetail,
          spineArtDetailWidth: Number(book.dataset.spineArtDetailWidth) || undefined,
          spineArtDetailHeight: Number(book.dataset.spineArtDetailHeight) || undefined,
        })
        .then(ready => {
          if (ready && enter && s.dialog.open) s.hardcover?.reveal()
        })
    }
  }

  s.openPreview = (index: number) => {
    if (!s.dialog.open) {
      const active = document.activeElement
      s.returnFocus = active instanceof HTMLElement && active !== document.body ? active : undefined
      s.dialog.showModal()
    }
    s.aim = undefined
    s.previewIndex = index
    s.previewWheelDelta = 0
    s.previewWheelStep = -Infinity
    s.renderPreview(true)
  }

  s.browsePreview = (delta: number) => {
    const visible = s.matches()
    const position = visible.findIndex(entry => Number(entry.dataset.bookIndex) === s.previewIndex)
    const row = visible[position + delta]
    if (!row) return
    s.previewIndex = Number(row.dataset.bookIndex)
    s.renderPreview()
  }

  s.wheelUnit = (event: WheelEvent, height: number) =>
    event.deltaMode === WheelEvent.DOM_DELTA_LINE
      ? 16
      : event.deltaMode === WheelEvent.DOM_DELTA_PAGE
        ? height
        : 1

  s.finishWheel = () => {
    s.wheelTimer = undefined
    if (s.signal.aborted || s.dialog.open) return
    s.camera.x = s.scrollOrigin.x - s.viewport.scrollLeft
    s.camera.y = s.scrollOrigin.y - s.viewport.scrollTop
    s.schedule()
  }
}
