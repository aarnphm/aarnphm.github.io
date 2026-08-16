import { cleanupHydratedRoot } from './root-lifecycle'

export interface StreamVirtualSource {
  entryId: string
  load: () => Promise<HTMLElement>
  mount: (entry: HTMLElement) => void
  failure?: (error: unknown) => HTMLElement
}

interface StreamVirtualSlot {
  source: StreamVirtualSource
  node: HTMLElement
  mounted: boolean
  loading: boolean
  pointerActive: boolean
  statefulInteraction: boolean
  height: number | null
  measuredWidth: number | null
}

export interface StreamVirtualizer {
  registerMounted: (entry: HTMLElement, source: StreamVirtualSource) => void
  registerPlaceholder: (placeholder: HTMLElement, source: StreamVirtualSource) => void
  refresh: () => void
  destroy: () => void
}

const OVERSCAN_PX = 2400
const measuredEntries = new Map<string, { height: number; width: number }>()

const positiveSize = (value: number): number | null =>
  Number.isFinite(value) && value > 0 ? value : null

const copySlotState = (source: HTMLElement, target: HTMLElement): void => {
  for (const [key, value] of Object.entries(source.dataset)) {
    if (value !== undefined) target.dataset[key] = value
  }
  target.hidden = source.hidden
  target.classList.toggle('stream-entry-active', source.classList.contains('stream-entry-active'))
}

const targetEntryId = (): string | null => {
  const url = new URL(window.location.href)
  const queryTarget = url.searchParams.get('entry')?.trim()
  if (queryTarget) return queryTarget
  return decodeURIComponent(url.hash.slice(1)).trim() || null
}

const hasActiveMedia = (entry: HTMLElement): boolean =>
  Array.from(entry.querySelectorAll<HTMLMediaElement>('audio, video')).some(
    media => !media.paused && !media.ended,
  )

const hasInteractiveState = (entry: HTMLElement): boolean => {
  if (entry.classList.contains('stream-entry-active')) return true
  const selected = entry.querySelector<HTMLElement>('[data-selected-kind][data-selected-id]')
  return Boolean(selected?.dataset.selectedKind && selected.dataset.selectedId)
}

const shouldRemainMounted = (slot: StreamVirtualSlot): boolean => {
  if (slot.pointerActive || slot.statefulInteraction) return true
  const { node } = slot
  const active = document.activeElement
  if (active instanceof Node && node.contains(active)) return true
  const fullscreen = document.fullscreenElement
  if (fullscreen && node.contains(fullscreen)) return true
  if (hasActiveMedia(node) || hasInteractiveState(node)) return true
  const target = targetEntryId()
  return target === slot.source.entryId || target === node.id
}

const isNearViewport = (node: HTMLElement): boolean => {
  if (!node.isConnected || node.hidden) return false
  const bounds = node.getBoundingClientRect()
  return bounds.bottom >= -OVERSCAN_PX && bounds.top <= window.innerHeight + OVERSCAN_PX
}

const createSpacer = (slot: StreamVirtualSlot, height: number): HTMLLIElement => {
  const spacer = document.createElement('li')
  spacer.className = 'stream-entry-spacer'
  spacer.style.blockSize = `${height}px`
  spacer.setAttribute('aria-hidden', 'true')
  copySlotState(slot.node, spacer)
  const streamLink = slot.node.querySelector<HTMLElement>('[data-stream-link][data-stream-href]')
  if (streamLink?.dataset.streamHref) spacer.dataset.streamHref = streamLink.dataset.streamHref
  spacer.dataset.entryId = slot.source.entryId
  spacer.dataset.streamVirtual = 'spacer'
  return spacer
}

export const setupStreamVirtualizer = (
  feed: HTMLOListElement,
  signal: AbortSignal,
): StreamVirtualizer => {
  const slots = new Map<string, StreamVirtualSlot>()
  let destroyed = false
  let frame = 0
  let feedWidth = positiveSize(feed.getBoundingClientRect().width)

  const resizeObserver = new ResizeObserver(entries => {
    for (const observed of entries) {
      if (observed.target === feed) {
        const width = positiveSize(observed.contentRect.width)
        if (width !== null && feedWidth !== null && Math.abs(width - feedWidth) >= 1) {
          feedWidth = width
          for (const slot of slots.values()) {
            if (slot.measuredWidth === null || Math.abs(slot.measuredWidth - width) < 1) continue
            measuredEntries.delete(slot.source.entryId)
            slot.height = null
            slot.measuredWidth = null
          }
          scheduleSweep()
        } else if (width !== null) {
          feedWidth = width
        }
        continue
      }
      if (!(observed.target instanceof HTMLElement)) continue
      const entryId = observed.target.dataset.entryId
      if (!entryId) continue
      const slot = slots.get(entryId)
      if (!slot || !slot.mounted || slot.node !== observed.target) continue
      const height = positiveSize(observed.target.getBoundingClientRect().height)
      const width = positiveSize(feed.getBoundingClientRect().width) ?? feedWidth
      if (height === null || width === null) continue
      slot.height = height
      slot.measuredWidth = width
      measuredEntries.set(entryId, { height, width })
    }
  })

  const observer = new IntersectionObserver(() => scheduleSweep(), {
    rootMargin: `${OVERSCAN_PX}px 0px`,
  })

  const removeSlot = (slot: StreamVirtualSlot): void => {
    observer.unobserve(slot.node)
    resizeObserver.unobserve(slot.node)
    slots.delete(slot.source.entryId)
  }

  const failSlot = (slot: StreamVirtualSlot, error: unknown): void => {
    const replacement = slot.source.failure?.(error)
    if (!replacement) {
      console.error(error)
      slot.node.removeAttribute('aria-busy')
      slot.loading = false
      return
    }
    copySlotState(slot.node, replacement)
    replacement.dataset.entryId = slot.source.entryId
    slot.node.replaceWith(replacement)
    removeSlot(slot)
  }

  const mountSlot = (slot: StreamVirtualSlot): void => {
    if (destroyed || slot.mounted || slot.loading) return
    slot.loading = true
    slot.node.setAttribute('aria-busy', 'true')
    const placeholder = slot.node
    void slot.source
      .load()
      .then(entry => {
        if (destroyed || signal.aborted || slot.node !== placeholder || !placeholder.isConnected) {
          return
        }
        copySlotState(placeholder, entry)
        entry.dataset.entryId = slot.source.entryId
        delete entry.dataset.streamVirtual
        entry.removeAttribute('aria-busy')
        observer.unobserve(placeholder)
        placeholder.replaceWith(entry)
        slot.node = entry
        slot.mounted = true
        slot.loading = false
        slot.source.mount(entry)
        observer.observe(entry)
        resizeObserver.observe(entry)
        scheduleSweep()
      })
      .catch(error => {
        if (destroyed || signal.aborted || slot.node !== placeholder) return
        failSlot(slot, error)
      })
  }

  const unmountSlot = (slot: StreamVirtualSlot): void => {
    if (!slot.mounted || shouldRemainMounted(slot)) return
    const measured = positiveSize(slot.node.getBoundingClientRect().height)
    const cached = measuredEntries.get(slot.source.entryId)
    const height = measured ?? slot.height ?? cached?.height ?? null
    if (height === null) return
    const width =
      positiveSize(feed.getBoundingClientRect().width) ?? feedWidth ?? cached?.width ?? null
    if (measured !== null && width !== null) {
      slot.height = measured
      slot.measuredWidth = width
      measuredEntries.set(slot.source.entryId, { height: measured, width })
    }
    const entry = slot.node
    const spacer = createSpacer(slot, height)
    observer.unobserve(entry)
    resizeObserver.unobserve(entry)
    entry.replaceWith(spacer)
    slot.node = spacer
    slot.mounted = false
    cleanupHydratedRoot(entry)
    observer.observe(spacer)
  }

  const sweep = (): void => {
    frame = 0
    if (destroyed || signal.aborted) return
    for (const slot of slots.values()) {
      const near = isNearViewport(slot.node)
      if (slot.mounted) {
        if (!near) unmountSlot(slot)
      } else if (near) {
        mountSlot(slot)
      }
    }
  }

  function scheduleSweep(): void {
    if (destroyed || frame !== 0) return
    frame = window.requestAnimationFrame(sweep)
  }

  const register = (node: HTMLElement, source: StreamVirtualSource, mounted: boolean): void => {
    if (destroyed) return
    const existing = slots.get(source.entryId)
    if (existing) removeSlot(existing)
    node.dataset.entryId = source.entryId
    const cached = measuredEntries.get(source.entryId)
    const currentMeasurement =
      cached && (feedWidth === null || Math.abs(cached.width - feedWidth) < 1) ? cached : undefined
    const slot: StreamVirtualSlot = {
      source,
      node,
      mounted,
      loading: false,
      pointerActive: false,
      statefulInteraction: false,
      height: currentMeasurement?.height ?? null,
      measuredWidth: currentMeasurement?.width ?? null,
    }
    slots.set(source.entryId, slot)
    observer.observe(node)
    if (mounted) resizeObserver.observe(node)
    scheduleSweep()
  }

  const onPointerDown = (event: PointerEvent): void => {
    if (!(event.target instanceof Element)) return
    const owner = event.target.closest<HTMLElement>('[data-entry-id]')
    const entryId = owner?.dataset.entryId
    const slot = entryId ? slots.get(entryId) : undefined
    if (slot) slot.pointerActive = true
  }
  const releasePointer = (): void => {
    for (const slot of slots.values()) slot.pointerActive = false
    scheduleSweep()
  }
  const onStatefulInteraction = (event: Event): void => {
    if (!(event.target instanceof Element)) return
    const control = event.target.closest(
      'button, input, select, textarea, summary, [aria-expanded], [data-analysis-range], .tri-act-head, .tri-swim-mode',
    )
    const owner = control?.closest<HTMLElement>('[data-entry-id]')
    const entryId = owner?.dataset.entryId
    const slot = entryId ? slots.get(entryId) : undefined
    if (slot) slot.statefulInteraction = true
  }
  feed.addEventListener('pointerdown', onPointerDown, { passive: true })
  feed.addEventListener('click', onStatefulInteraction)
  feed.addEventListener('change', onStatefulInteraction)
  window.addEventListener('pointerup', releasePointer, { passive: true })
  window.addEventListener('pointercancel', releasePointer, { passive: true })
  window.addEventListener('resize', scheduleSweep, { passive: true })
  resizeObserver.observe(feed)

  const destroy = (): void => {
    if (destroyed) return
    destroyed = true
    signal.removeEventListener('abort', destroy)
    if (frame !== 0) window.cancelAnimationFrame(frame)
    observer.disconnect()
    resizeObserver.disconnect()
    feed.removeEventListener('pointerdown', onPointerDown)
    feed.removeEventListener('click', onStatefulInteraction)
    feed.removeEventListener('change', onStatefulInteraction)
    window.removeEventListener('pointerup', releasePointer)
    window.removeEventListener('pointercancel', releasePointer)
    window.removeEventListener('resize', scheduleSweep)
    for (const slot of slots.values()) {
      if (slot.mounted) cleanupHydratedRoot(slot.node)
    }
    slots.clear()
  }
  signal.addEventListener('abort', destroy, { once: true })

  return {
    registerMounted: (entry, source) => register(entry, source, true),
    registerPlaceholder: (placeholder, source) => register(placeholder, source, false),
    refresh: scheduleSweep,
    destroy,
  }
}
