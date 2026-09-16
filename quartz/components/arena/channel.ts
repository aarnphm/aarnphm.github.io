import { arenaModalSource, parseArenaBlockOrder } from './channel-data'
import { mountArenaVirtualGrid } from './channel-virtualizer'

export function mountArenaChannel(root: HTMLElement) {
  const assetBase = root.dataset.arenaAssets
  if (!assetBase) return null

  const blockIds = parseArenaBlockOrder(root.dataset.arenaBlockOrder ?? '[]')
  const lifetime = new AbortController()
  const modalCache = new Map<string, string>()
  const grids = Array.from(root.querySelectorAll<HTMLElement>('[data-arena-grid]')).map(grid =>
    mountArenaVirtualGrid(grid, assetBase, blockIds, lifetime.signal, refresh),
  )
  function refresh() {
    for (const grid of grids) grid?.refresh()
  }
  root.addEventListener('toggle', refresh, { capture: true, signal: lifetime.signal })

  return {
    blockIds,
    refresh,
    searchSource: `${assetBase}/search.json`,
    async loadModal(blockId: string, signal: AbortSignal): Promise<HTMLElement> {
      if (!blockIds.includes(blockId)) throw new Error('Unknown Arena block')
      let html = modalCache.get(blockId)
      if (html === undefined) {
        const response = await fetch(arenaModalSource(assetBase, blockId), { signal })
        if (!response.ok) throw new Error(`Could not load Arena block (${response.status})`)
        html = await response.text()
        if (signal.aborted || lifetime.signal.aborted)
          throw new DOMException('Aborted', 'AbortError')
        modalCache.set(blockId, html)
        if (modalCache.size > 12) {
          const oldest = modalCache.keys().next().value
          if (oldest !== undefined) modalCache.delete(oldest)
        }
      }
      const template = document.createElement('template')
      template.innerHTML = html
      const modal = template.content.querySelector<HTMLElement>('.arena-block-modal-data')
      if (!modal || modal.dataset.blockId !== blockId)
        throw new Error('Invalid Arena block content')
      return modal
    },
    destroy() {
      lifetime.abort()
      modalCache.clear()
    },
  }
}
