export type Cover = {
  url: string
  spine?: string
  ink?: string
  art?: string
  artDetail?: string
  artDetailWidth?: string
  artDetailHeight?: string
}

export function retainCovers(root: HTMLElement, covers: Map<string, Cover>): void {
  const nodes = root.querySelectorAll<HTMLElement | SVGElement>('[data-bookcase-cover]')
  for (const node of nodes) {
    const slug = node.dataset.bookcaseCover
    const url = node instanceof HTMLImageElement ? node.getAttribute('src') : node.dataset.cover
    if (!slug || !url?.includes('/static/bookcase/')) continue
    const cover = { ...covers.get(slug), url }
    if (node.matches('.bc-book')) {
      cover.spine = node.style.getPropertyValue('--spine')
      cover.ink = node.style.getPropertyValue('--spine-ink')
      if (node.dataset.spineArt) cover.art = node.dataset.spineArt
      if (node.dataset.spineArtDetail) {
        cover.artDetail = node.dataset.spineArtDetail
        cover.artDetailWidth = node.dataset.spineArtDetailWidth
        cover.artDetailHeight = node.dataset.spineArtDetailHeight
      }
    }
    covers.set(slug, cover)
  }
  for (const node of nodes) {
    const slug = node.dataset.bookcaseCover
    const cover = slug ? covers.get(slug) : undefined
    if (!cover) continue
    if (node instanceof HTMLImageElement) {
      if (node.getAttribute('src') !== cover.url) node.src = cover.url
      continue
    }
    if (node.dataset.cover !== cover.url) node.dataset.cover = cover.url
    if (cover.art && node.dataset.spineArt !== cover.art) node.dataset.spineArt = cover.art
    if (cover.artDetail && node.dataset.spineArtDetail !== cover.artDetail) {
      node.dataset.spineArtDetail = cover.artDetail
      node.dataset.spineArtDetailWidth = cover.artDetailWidth ?? ''
      node.dataset.spineArtDetailHeight = cover.artDetailHeight ?? ''
    }
    if (node.hasAttribute('data-spine-sampled')) {
      if (cover.spine && node.style.getPropertyValue('--spine') !== cover.spine)
        node.style.setProperty('--spine', cover.spine)
      if (cover.ink && node.style.getPropertyValue('--spine-ink') !== cover.ink)
        node.style.setProperty('--spine-ink', cover.ink)
    }
  }
}
