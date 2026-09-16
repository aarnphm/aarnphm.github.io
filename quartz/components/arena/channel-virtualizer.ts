import {
  ARENA_CARD_PAGE_SIZE,
  arenaCardPageSource,
  parseArenaCardPage,
  type ArenaSectionName,
} from './channel-data'
import { arenaRowAt, arenaRowOffsets, arenaVisibleRows } from './channel-window'

const OVERSCAN = 600
const FOCUSABLE = '.arena-block-clickable, a[href], button:not([disabled]), [tabindex="0"]'

export function mountArenaVirtualGrid(
  grid: HTMLElement,
  assetBase: string,
  blockIds: readonly string[],
  signal: AbortSignal,
  onLayoutChange: () => void,
) {
  const name = grid.dataset.arenaGrid
  if (name !== 'pinned' && name !== 'later' && name !== 'blocks') return null
  const section: ArenaSectionName = name
  const count = Number(grid.dataset.arenaCount)
  const startIndex = Number(grid.dataset.arenaStart)
  if (!Number.isInteger(count) || count <= 0 || !Number.isInteger(startIndex)) return null

  const cards = new Map<number, string>()
  const loadedPages = new Set<number>()
  const pendingPages = new Set<number>()
  const failedPages = new Set<number>()
  const rows = new Map<number, HTMLElement>()
  const measured = new Map<number, number>()
  let columns = 1
  let estimate = 80
  let offsets = arenaRowOffsets(count, columns, estimate, measured)
  let layoutKey = ''
  let frame = 0
  let pendingFocus: { index: number; backwards: boolean } | null = null

  for (const card of grid.querySelectorAll<HTMLElement>(':scope > .arena-block')) {
    const index = Number(card.dataset.blockIndex) - startIndex
    cards.set(index, card.outerHTML)
  }
  if (cards.size === Math.min(count, ARENA_CARD_PAGE_SIZE)) loadedPages.add(0)
  grid.replaceChildren()
  grid.dataset.arenaVirtual = 'true'
  grid.setAttribute('role', 'list')
  grid.setAttribute('aria-label', section)

  const resizeObserver = new ResizeObserver(schedule)
  resizeObserver.observe(grid)

  async function loadPage(offset: number) {
    if (loadedPages.has(offset) || pendingPages.has(offset) || failedPages.has(offset)) return
    pendingPages.add(offset)
    try {
      const response = await fetch(arenaCardPageSource(assetBase, section, offset), { signal })
      if (!response.ok) throw new Error(`Could not load Arena cards (${response.status})`)
      const page = parseArenaCardPage(await response.json())
      const template = document.createElement('template')
      template.innerHTML = page.html
      const elements = Array.from(template.content.children)
      if (
        page.offset !== offset ||
        page.total !== count ||
        elements.length !== Math.min(ARENA_CARD_PAGE_SIZE, count - offset) ||
        !elements.every(
          (element, index) =>
            element instanceof HTMLElement &&
            element.classList.contains('arena-block') &&
            element.dataset.blockId === blockIds[startIndex + offset + index] &&
            Number(element.dataset.blockIndex) === startIndex + offset + index,
        )
      )
        throw new Error('Arena card page does not match the section')
      if (signal.aborted) return
      elements.forEach((element, index) => cards.set(offset + index, element.outerHTML))
      loadedPages.add(offset)
    } catch (error) {
      if (signal.aborted) return
      failedPages.add(offset)
      console.error(error)
    } finally {
      pendingPages.delete(offset)
      schedule()
    }
  }

  function cardElement(index: number): HTMLElement {
    const html = cards.get(index)
    if (html) {
      const template = document.createElement('template')
      template.innerHTML = html
      const card = template.content.firstElementChild
      if (card instanceof HTMLElement) {
        card.setAttribute('role', 'listitem')
        card.setAttribute('aria-posinset', String(index + 1))
        card.setAttribute('aria-setsize', String(count))
        return card
      }
    }
    const placeholder = document.createElement('div')
    placeholder.className = 'arena-block arena-block-placeholder'
    placeholder.dataset.virtualIndex = String(index)
    placeholder.style.minBlockSize = `${estimate}px`
    const offset = Math.floor(index / ARENA_CARD_PAGE_SIZE) * ARENA_CARD_PAGE_SIZE
    if (failedPages.has(offset)) {
      const retry = document.createElement('button')
      retry.type = 'button'
      retry.textContent = 'could not load blocks · retry'
      retry.addEventListener(
        'click',
        () => {
          failedPages.delete(offset)
          void loadPage(offset)
          schedule()
        },
        { signal },
      )
      placeholder.append(retry)
      placeholder.dataset.failed = 'true'
    } else {
      placeholder.textContent = 'loading…'
      placeholder.setAttribute('aria-hidden', 'true')
      void loadPage(offset)
    }
    return placeholder
  }

  function focusedIndex(): number | null {
    const active = document.activeElement
    if (!(active instanceof Element) || !grid.contains(active)) return null
    const card = active.closest<HTMLElement>('[data-block-index]')
    return card ? Number(card.dataset.blockIndex) - startIndex : null
  }

  function removeRow(index: number, row: HTMLElement) {
    resizeObserver.unobserve(row)
    row.remove()
    rows.delete(index)
  }

  function render() {
    frame = 0
    if (signal.aborted || document.documentElement.classList.contains('arena-modal-open')) return
    if (grid.closest('details:not([open])')) {
      for (const [index, row] of rows) removeRow(index, row)
      return
    }
    const bounds = grid.getBoundingClientRect()
    if (bounds.width === 0) return
    const style = getComputedStyle(grid)
    const list = grid.dataset.viewMode === 'list'
    const nextColumns = list ? 1 : style.gridTemplateColumns.split(' ').length
    const key = `${grid.dataset.viewMode}:${bounds.width}:${nextColumns}`
    const anchorRow = arenaRowAt(offsets, -bounds.top)
    const anchorIndex = anchorRow * columns
    const previousAnchor = offsets[anchorRow]
    const previousTotal = offsets.at(-1) ?? 0
    const focus = focusedIndex()

    if (layoutKey !== key) {
      layoutKey = key
      columns = nextColumns
      estimate = list ? 54 : 80
      measured.clear()
      if (focus !== null) pendingFocus = { index: focus, backwards: false }
      for (const [index, row] of rows) removeRow(index, row)
    } else {
      for (const [index, row] of rows) {
        if (!row.querySelector('.arena-block-placeholder')) {
          measured.set(index, row.getBoundingClientRect().height)
        }
      }
    }
    offsets = arenaRowOffsets(count, columns, estimate, measured)
    const total = offsets.at(-1) ?? 0
    const adjustment =
      bounds.top < 0
        ? -bounds.top >= previousTotal
          ? total - previousTotal
          : offsets[Math.floor(anchorIndex / columns)] - previousAnchor
        : 0
    const blockSize = `${total}px`
    if (grid.style.blockSize !== blockSize) {
      grid.style.blockSize = blockSize
      onLayoutChange()
    }
    if (adjustment !== 0) window.scrollBy({ top: adjustment, behavior: 'instant' })

    const visible = new Set(
      arenaVisibleRows(offsets, -bounds.top + adjustment, window.innerHeight, OVERSCAN),
    )
    if (focus !== null) visible.add(Math.floor(focus / columns))
    if (pendingFocus) visible.add(Math.floor(pendingFocus.index / columns))
    for (const [index, row] of rows) {
      if (!visible.has(index)) removeRow(index, row)
    }

    for (const index of [...visible].sort((a, b) => a - b)) {
      let row = rows.get(index)
      if (!row) {
        row = document.createElement('div')
        row.className = 'arena-virtual-row'
        row.setAttribute('role', 'presentation')
        for (let card = index * columns; card < Math.min(count, (index + 1) * columns); card++) {
          row.append(cardElement(card))
        }
        rows.set(index, row)
        const next = [...grid.children].find(
          child => Number(child.getAttribute('data-row')) > index,
        )
        row.dataset.row = String(index)
        grid.insertBefore(row, next ?? null)
        resizeObserver.observe(row)
      } else {
        for (const placeholder of row.querySelectorAll<HTMLElement>('.arena-block-placeholder')) {
          const card = Number(placeholder.dataset.virtualIndex)
          const failed = failedPages.has(
            Math.floor(card / ARENA_CARD_PAGE_SIZE) * ARENA_CARD_PAGE_SIZE,
          )
          if (cards.has(card) || failed !== (placeholder.dataset.failed === 'true')) {
            placeholder.replaceWith(cardElement(card))
          }
        }
      }
      row.style.insetBlockStart = `${offsets[index]}px`
    }
    if (pendingFocus) {
      const card = grid.querySelector<HTMLElement>(
        `[data-block-index="${startIndex + pendingFocus.index}"]`,
      )
      const controls = card?.querySelectorAll<HTMLElement>(FOCUSABLE)
      const target = pendingFocus.backwards ? controls?.[controls.length - 1] : controls?.[0]
      if (target) {
        pendingFocus = null
        target.focus()
      }
    }
  }

  function schedule() {
    if (!signal.aborted && frame === 0) frame = window.requestAnimationFrame(render)
  }

  grid.addEventListener(
    'keydown',
    event => {
      if (event.key !== 'Tab' || event.altKey || event.ctrlKey || event.metaKey) return
      const index = focusedIndex()
      if (index === null) return
      const card = grid.querySelector<HTMLElement>(`[data-block-index="${startIndex + index}"]`)
      const controls = card?.querySelectorAll<HTMLElement>(FOCUSABLE)
      const edge = event.shiftKey ? controls?.[0] : controls?.[controls.length - 1]
      const next = index + (event.shiftKey ? -1 : 1)
      if (edge !== document.activeElement || next < 0 || next >= count) return
      if (grid.querySelector(`[data-block-index="${startIndex + next}"]`)) return
      event.preventDefault()
      pendingFocus = { index: next, backwards: event.shiftKey }
      schedule()
    },
    { signal },
  )
  grid.addEventListener('focusout', schedule, { signal })
  window.addEventListener('scroll', schedule, { signal, passive: true })
  window.addEventListener('resize', schedule, { signal, passive: true })
  grid.closest('details')?.addEventListener('toggle', schedule, { signal })
  signal.addEventListener(
    'abort',
    () => {
      window.cancelAnimationFrame(frame)
      resizeObserver.disconnect()
      cards.clear()
      rows.clear()
    },
    { once: true },
  )
  schedule()
  return { refresh: schedule }
}
