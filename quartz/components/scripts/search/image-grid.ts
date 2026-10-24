export type ImageGridDirection = 'ArrowUp' | 'ArrowDown' | 'ArrowLeft' | 'ArrowRight'

export function isImageGridDirection(key: string): key is ImageGridDirection {
  return key === 'ArrowUp' || key === 'ArrowDown' || key === 'ArrowLeft' || key === 'ArrowRight'
}

export type ImageGrid = ReturnType<typeof createImageGrid>

export function createImageGrid(container: HTMLElement) {
  let gallery: HTMLElement | null = null
  let cards: HTMLElement[] = []
  let frame = 0
  let disposed = false

  function pack() {
    frame = 0
    if (!gallery?.isConnected || gallery.clientWidth === 0) return
    const style = getComputedStyle(gallery)
    const gap = parseFloat(style.columnGap) || 0
    const tracks = style.gridTemplateColumns.split(' ').filter(Boolean)
    const columns = Math.max(1, Math.min(cards.length, tracks.length))
    const width = (gallery.clientWidth - gap * (columns - 1)) / columns
    const heights = Array.from({ length: columns }, () => 0)
    gallery.dataset.packed = ''

    for (const card of cards) card.style.width = `${width}px`
    const sizes = cards.map(card => card.getBoundingClientRect().height)
    for (const [index, card] of cards.entries()) {
      const column = heights.indexOf(Math.min(...heights))
      card.style.transform = `translate(${column * (width + gap)}px, ${heights[column]}px)`
      heights[column] += sizes[index] + gap
    }
    gallery.style.height = `${Math.max(0, Math.max(...heights) - gap)}px`
  }

  function schedule() {
    if (!disposed && !frame) frame = requestAnimationFrame(pack)
  }

  const resize = new ResizeObserver(schedule)
  function refresh() {
    const next = container.querySelector<HTMLElement>(':scope > .image-search-grid')
    if (!next && !gallery) return
    if (
      next === gallery &&
      next?.childElementCount === cards.length &&
      next?.lastElementChild === cards.at(-1)
    )
      return
    gallery = next
    cards = gallery ? [...gallery.querySelectorAll<HTMLElement>(':scope > .image-search-row')] : []
    resize.disconnect()
    resize.observe(container)
    for (const card of cards) resize.observe(card)
    schedule()
  }

  const mutation = new MutationObserver(refresh)
  mutation.observe(container, { childList: true, subtree: true })
  resize.observe(container)
  refresh()

  return {
    get active() {
      refresh()
      return cards.length > 0
    },
    target(active: Element | null, direction: ImageGridDirection): HTMLElement | null {
      refresh()
      if (frame) {
        cancelAnimationFrame(frame)
        pack()
      }
      const current = active?.closest<HTMLElement>('.image-search-row')
      const positions = cards.map(card => ({ card, rect: card.getBoundingClientRect() }))
      const source = positions.find(({ card }) => card === current)
      if (!source) {
        positions.sort((a, b) => a.rect.top - b.rect.top || a.rect.left - b.rect.left)
        const entry = direction === 'ArrowUp' ? positions.at(-1) : positions[0]
        return entry?.card.querySelector<HTMLAnchorElement>('a.result-card') ?? null
      }

      const horizontal = direction === 'ArrowLeft' || direction === 'ArrowRight'
      const forward = direction === 'ArrowRight' || direction === 'ArrowDown'
      const start = horizontal ? 'left' : 'top'
      const end = horizontal ? 'right' : 'bottom'
      const crossStart = horizontal ? 'top' : 'left'
      const crossEnd = horizontal ? 'bottom' : 'right'
      const candidates = positions.flatMap(({ card, rect }) => {
        if (card === source.card) return []
        const primary = forward ? rect[start] - source.rect[end] : source.rect[start] - rect[end]
        if (primary < -1) return []
        const cross = Math.max(
          0,
          rect[crossStart] - source.rect[crossEnd],
          source.rect[crossStart] - rect[crossEnd],
        )
        const center = Math.abs(
          (rect[crossStart] + rect[crossEnd] - source.rect[crossStart] - source.rect[crossEnd]) / 2,
        )
        return [{ card, aligned: cross === 0, distance: primary + cross * 2, center }]
      })
      // Stay in the same visual row or column before considering diagonal neighbors.
      candidates.sort(
        (a, b) =>
          Number(b.aligned) - Number(a.aligned) || a.distance - b.distance || a.center - b.center,
      )
      return candidates[0]?.card.querySelector<HTMLAnchorElement>('a.result-card') ?? null
    },
    dispose() {
      disposed = true
      cancelAnimationFrame(frame)
      resize.disconnect()
      mutation.disconnect()
      cards = []
      gallery = null
    },
  }
}
