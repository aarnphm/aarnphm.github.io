import type { BookcaseScene } from './scene'
import { removeArtwork } from './artwork'

export function bindTopics(s: BookcaseScene) {
  s.entryTopics = new Map(
    s.entries.map(entry => [entry, new Set<string>(JSON.parse(entry.dataset.topics ?? '[]'))]),
  )

  s.topicButtons = [...s.root.querySelectorAll<HTMLButtonElement>('.bookcase-topic')]

  s.showTopic = (next: string) => {
    s.topic = s.topicButtons.some(button => button.dataset.topic === next) ? next : ''
    for (const button of s.topicButtons) {
      button.setAttribute('aria-selected', String(button.dataset.topic === s.topic))
      button.tabIndex = button.dataset.topic === s.topic ? 0 : -1
    }
    for (const entry of s.entries) {
      entry.toggleAttribute(
        'data-topic-hidden',
        Boolean(s.topic) && !s.entryTopics.get(entry)?.has(s.topic),
      )
    }
    for (const group of s.groups) {
      const empty = !s.groupEntries
        .get(group)
        ?.entries.some(entry => !entry.hasAttribute('data-topic-hidden'))
      group.toggleAttribute('data-topic-hidden', empty)
    }
    const url = new URL(window.location.href)
    if (s.topic) url.searchParams.set('topic', s.topic)
    else url.searchParams.delete('topic')
    history.replaceState(history.state, '', url)
    s.refresh()
  }

  s.topicList = s.root.querySelector<HTMLElement>('.bookcase-topics')

  s.viewport.id ||= `bookcase-viewport-${crypto.randomUUID()}`

  for (const button of s.topicButtons) {
    button.setAttribute('aria-controls', s.viewport.id)
    button.addEventListener('click', () => s.showTopic(button.dataset.topic ?? ''), {
      signal: s.signal,
    })
  }

  s.nextTab = (key: string, active: number): number | undefined => {
    const n = s.topicButtons.length
    if (key === 'ArrowUp' || key === 'ArrowLeft') return (active - 1 + n) % n
    if (key === 'ArrowDown' || key === 'ArrowRight') return (active + 1) % n
    if (key === 'Home') return 0
    if (key === 'End') return n - 1
    // Repeating a category's initial cycles through its matching tabs.
    if (key.length !== 1) return undefined
    for (let step = 1; step <= n; step++) {
      const i = (active + step) % n
      if (s.topicButtons[i].dataset.shortcut === key.toLowerCase()) return i
    }
    return undefined
  }

  s.root.addEventListener(
    'keydown',
    event => {
      if (event.defaultPrevented || s.dialog.open) return
      if (event.ctrlKey || event.metaKey || event.altKey || event.isComposing) return
      if (!(event.target instanceof Element)) return
      if (event.target.closest('input, textarea, select, [contenteditable]')) return
      const inTopics = s.topicList?.contains(event.target)
      if (!inTopics && !/^\p{L}$/u.test(event.key)) return
      const active = s.topicButtons.findIndex(button => button.dataset.topic === s.topic)
      const next = s.nextTab(event.key, Math.max(0, active))
      if (next === undefined) return
      event.preventDefault()
      s.showTopic(s.topicButtons[next].dataset.topic ?? '')
      s.topicButtons[next].focus()
    },
    { signal: s.signal },
  )

  s.updateFit = () => {
    const previousFit = s.fitScale
    s.fitScale = Math.max(
      0.01,
      Math.min((s.viewportWidth - 32) / s.fitWidth, (s.viewportHeight - 72) / s.box.height),
    )
    s.minScale = Math.min(s.fitScale, (s.viewportWidth - 32) / s.box.width)
    if (s.fitted || s.camera.scale < s.minScale || previousFit === 1) s.fit()
    else s.schedule()
  }

  s.resize = () => {
    if (!s.root.isConnected) return
    const nextWidth = s.viewport.clientWidth
    const nextHeight = s.viewport.clientHeight
    if (nextWidth === s.viewportWidth && nextHeight === s.viewportHeight && s.painted.at > 0) return
    for (const view of s.aim ? [s.camera, s.aim] : [s.camera]) {
      view.x += (nextWidth - s.viewportWidth) / 2
      view.y += (nextHeight - s.viewportHeight) / 2
    }
    s.viewportWidth = nextWidth
    s.viewportHeight = nextHeight
    s.cameraDirty = true
    s.updateFit()
  }

  s.observer = new ResizeObserver(s.resize)

  s.observer.observe(s.root)

  s.observer.observe(s.viewport)

  window.addEventListener('resize', s.resize, { signal: s.signal })

  s.updateFit()

  s.initialTopic = new URL(window.location.href).searchParams.get('topic')

  if (s.initialTopic) {
    // A shared link opens on its tab without the label animation.
    if (s.topicList) s.topicList.dataset.motion = 'instant'
    s.showTopic(s.initialTopic)
    s.topicList?.getBoundingClientRect()
    if (s.topicList) delete s.topicList.dataset.motion
  }

  s.signal.addEventListener(
    'abort',
    () => {
      s.observer.disconnect()
      s.disposeEscape()
      if (s.frame) cancelAnimationFrame(s.frame)
      s.cancelWheel()
      if (s.scaleTimer !== undefined) clearTimeout(s.scaleTimer)
      if (s.dialog.open) s.dialog.close()
      s.dropLift()
      for (const image of s.residentCovers.values()) removeArtwork(image)
      for (const image of s.scene.querySelectorAll('image[data-bookcase-texture]'))
        image.removeAttribute('href')
      s.residentCovers.clear()
      s.preloaded.clear()
      if (s.prefetchTask !== undefined && s.idle) cancelIdleCallback(s.prefetchTask)
      else if (s.prefetchTask !== undefined) clearTimeout(s.prefetchTask)
      for (const url of s.prefetched.keys()) s.dropPrefetch(url)
      s.pan.before(s.world)
      s.pan.remove()
      s.scrollSpace.remove()
      s.world.style.left = '0px'
      s.world.style.top = '0px'
      s.viewport.scrollLeft = 0
      s.viewport.scrollTop = 0
      s.visibleEntries = undefined
      delete s.root.dataset.bookcaseReady
    },
    { once: true },
  )
}
