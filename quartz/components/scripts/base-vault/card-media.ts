const PRELOAD = 3
const UPGRADE_DELAY_MS = 300
const PULL_MS = 220
const EASE = 'cubic-bezier(0.23, 1, 0.32, 1)'

export function createVaultCardMedia(
  dialog: HTMLDialogElement,
  rows: Map<number, HTMLLIElement>,
  reduced: MediaQueryList,
) {
  const total = rows.size
  let cardIndex = -1
  const titleOf = (row: HTMLLIElement) => row.querySelector('.vault-entry-title')?.textContent ?? ''
  const poster = dialog.querySelector<HTMLElement>('.vault-card-poster')
  const posterImage = poster?.querySelector('img')
  const previous = dialog.querySelector<HTMLButtonElement>('[data-card="previous"]')
  const next = dialog.querySelector<HTMLButtonElement>('[data-card="next"]')

  const preloaded = new Map<string, HTMLImageElement>()
  let upgradeTimer: ReturnType<typeof setTimeout> | undefined
  let activeOriginal: HTMLImageElement | undefined
  let pendingPosterUrl: string | undefined
  let entranceAnim: Animation | undefined
  let entranceFlying = false
  const preloadCovers = (centerIndex: number) => {
    const urls = new Set<string>()
    for (let i = centerIndex - PRELOAD; i <= centerIndex + PRELOAD; i++) {
      const url = rows.get(i)?.dataset.cover
      if (url) urls.add(url)
    }
    for (const [url, img] of preloaded) {
      if (!urls.has(url)) {
        img.removeAttribute('src')
        preloaded.delete(url)
      }
    }
    for (const url of urls) {
      if (preloaded.has(url)) continue
      const img = new Image()
      img.decoding = 'async'
      img.src = url
      preloaded.set(url, img)
    }
  }

  const clearPreloads = () => {
    for (const img of preloaded.values()) img.removeAttribute('src')
    preloaded.clear()
  }

  const cancelUpgrade = () => {
    if (upgradeTimer !== undefined) {
      clearTimeout(upgradeTimer)
      upgradeTimer = undefined
    }
    pendingPosterUrl = undefined
    if (activeOriginal) {
      activeOriginal.onload = null
      activeOriginal.onerror = null
      activeOriginal.removeAttribute('src')
      activeOriginal = undefined
    }
  }

  const flushPosterUpgrade = () => {
    if (entranceFlying || !pendingPosterUrl || !posterImage || !dialog.open) return
    posterImage.src = pendingPosterUrl
    pendingPosterUrl = undefined
  }

  const render = (shown: number, enter = false) => {
    cardIndex = shown
    const row = rows.get(shown)
    if (!row) return
    cancelUpgrade()
    preloadCovers(shown)
    const title = titleOf(row)
    const href = row.dataset.href ?? ''
    dialog.setAttribute('aria-label', `Vault card: ${title}`)
    const heading = dialog.querySelector<HTMLAnchorElement>('.vault-card-title')
    if (heading) {
      heading.textContent = title
      heading.href = href
    }
    const source = row.querySelector('.vault-entry-detail')
    dialog
      .querySelector('.vault-card-detail')
      ?.replaceChildren(...[...(source?.childNodes ?? [])].map(child => child.cloneNode(true)))
    const position = dialog.querySelector('.vault-card-position')
    if (position) position.textContent = `${shown + 1} / ${total}`
    if (previous) previous.disabled = shown <= 0
    if (next) next.disabled = shown >= total - 1
    if (!poster || !posterImage) return
    const thumbnail = row.dataset.cover
    const full = row.dataset.fullCover
    if (thumbnail) {
      delete poster.dataset.empty
      posterImage.src = thumbnail
    } else {
      poster.dataset.empty = ''
      posterImage.removeAttribute('src')
    }
    if (enter && !reduced.matches && typeof posterImage.animate === 'function') {
      entranceAnim?.cancel()
      entranceFlying = true
      const anim = posterImage.animate([{ transform: 'rotateY(6deg)' }, { transform: 'none' }], {
        duration: PULL_MS,
        easing: EASE,
      })
      entranceAnim = anim
      anim.onfinish = () => {
        if (entranceAnim === anim) entranceAnim = undefined
        entranceFlying = false
        flushPosterUpgrade()
      }
    }
    if (thumbnail && full && full !== thumbnail) {
      upgradeTimer = setTimeout(() => {
        upgradeTimer = undefined
        if (!dialog.open || cardIndex !== shown) return
        const original = new Image()
        original.decoding = 'async'
        activeOriginal = original
        const commit = () => {
          if (!dialog.open || cardIndex !== shown || activeOriginal !== original) return
          activeOriginal = undefined
          pendingPosterUrl = full
          flushPosterUpgrade()
        }
        original.onload = commit
        original.src = full
        if (typeof original.decode === 'function') {
          original.decode().then(commit, () => {})
        }
      }, UPGRADE_DELAY_MS)
    }
  }

  const clear = () => {
    cancelUpgrade()
    if (entranceAnim) {
      entranceAnim.onfinish = null
      entranceAnim.cancel()
      entranceAnim = undefined
    }
    entranceFlying = false
    posterImage?.removeAttribute('src')
    clearPreloads()
    cardIndex = -1
  }
  return { render, clear }
}
