import { playhtml } from 'playhtml'
import type { TrailVisitor } from './visitor-journey'

export type CampDestination = { value: () => number; onChange: (listener: () => void) => void }

export function anonymousVisitor(id: string) {
  return `visitor ${id.slice(0, 8)}`
}

export function connectCampPresence(
  scene: HTMLElement,
  host: HTMLElement,
  signal: AbortSignal,
  destination: CampDestination,
  onVisitors: (visitors: TrailVisitor[]) => void,
) {
  const document = scene.ownerDocument
  const view = document.defaultView
  const count = document.querySelector<HTMLElement>('[data-trail-visitors]')
  const connection = scene.querySelector<HTMLElement>('[data-camp-connection]')
  const cursors = scene.querySelector<HTMLElement>('.trail-shared-cursors')
  if (!view || !count || !connection || !cursors)
    throw new Error('Camp presence needs its controls')
  const room = playhtml.createPresenceRoom('camp-visitors-v1')
  const pointers = new Map<string, HTMLElement>()
  let pointer: { x: number; y: number; touch: boolean } | null = null
  let pointerTimer = 0
  let sentAt = -Infinity

  const renderVisitors = () => {
    const visitors: TrailVisitor[] = []
    for (const [id, peer] of room.presence.getPresences()) {
      if (peer.visiting !== true) continue
      const campsite = peer.campsite
      const camp =
        campsite &&
        typeof campsite === 'object' &&
        'camp' in campsite &&
        typeof campsite.camp === 'number' &&
        Number.isFinite(campsite.camp)
          ? Math.max(0, Math.min(2, Math.floor(campsite.camp)))
          : 0
      visitors.push({ id, own: peer.isMe, camp })
    }
    count.textContent = String(visitors.length)
    count.dataset.state = 'ready'
    count.setAttribute(
      'aria-label',
      `${visitors.length} ${visitors.length === 1 ? 'visitor' : 'visitors'} here`,
    )
    onVisitors(visitors.sort((a, b) => a.id.localeCompare(b.id)))
  }
  const renderPointers = () => {
    const shown = new Set<string>()
    if (!host.hidden)
      for (const [id, peer] of room.presence.getPresences()) {
        const pointer = peer.trailPointer
        if (
          peer.isMe ||
          peer.visiting !== true ||
          !pointer ||
          typeof pointer !== 'object' ||
          !('x' in pointer) ||
          !('y' in pointer) ||
          typeof pointer.x !== 'number' ||
          typeof pointer.y !== 'number' ||
          !Number.isFinite(pointer.x) ||
          !Number.isFinite(pointer.y)
        )
          continue
        shown.add(id)
        let cursor = pointers.get(id)
        if (!cursor) {
          cursor = document.createElement('div')
          cursor.className = 'trail-shared-cursor'
          cursor.dataset.visitorCursor = id
          cursor.innerHTML =
            '<svg viewBox="0 0 18 22"><path d="M2 2v15l4-4 4 7 3-2-4-7h6Z"/></svg><span></span>'
          const label = cursor.querySelector('span')
          if (label) label.textContent = anonymousVisitor(id)
          pointers.set(id, cursor)
          cursors.append(cursor)
        }
        cursor.style.left = `${Math.max(0, Math.min(1, pointer.x)) * view.innerWidth}px`
        cursor.style.top = `${Math.max(0, Math.min(1, pointer.y)) * view.innerHeight}px`
        cursor.dataset.flipped = String(pointer.x > 0.8)
        cursor.dataset.touch = String('touch' in pointer && pointer.touch === true)
      }
    for (const [id, cursor] of pointers)
      if (!shown.has(id)) {
        cursor.remove()
        pointers.delete(id)
      }
  }
  const publishPointer = () => {
    pointerTimer = 0
    if (signal.aborted || !pointer) return
    sentAt = view.performance.now()
    room.presence.setMyPresence('trailPointer', pointer)
  }
  document.addEventListener(
    'pointermove',
    event => {
      if (document.hidden) return
      pointer = {
        x: event.clientX / view.innerWidth,
        y: event.clientY / view.innerHeight,
        touch: event.pointerType === 'touch',
      }
      if (pointerTimer) return
      const delay = Math.max(0, 100 - (view.performance.now() - sentAt))
      pointerTimer = view.setTimeout(publishPointer, delay)
    },
    { passive: true, signal },
  )
  const clearPointer = () => {
    view.clearTimeout(pointerTimer)
    pointerTimer = 0
    pointer = null
    room.presence.setMyPresence('trailPointer', null)
  }
  document.addEventListener('pointerleave', clearPointer, { signal })
  const publish = () => {
    room.presence.setMyPresence('campsite', { camp: destination.value(), joined: true })
    room.presence.setMyPresence('visiting', true)
  }
  const clear = () => {
    clearPointer()
    room.presence.setMyPresence('visiting', null)
    room.presence.setMyPresence('campsite', null)
  }
  const unsubscribeVisitors = room.presence.onPresenceChange('visiting', () => {
    renderVisitors()
    renderPointers()
  })
  const unsubscribeCamp = room.presence.onPresenceChange('campsite', renderVisitors)
  const unsubscribePointer = room.presence.onPresenceChange('trailPointer', renderPointers)
  destination.onChange(() => {
    if (!signal.aborted) publish()
  })
  host.addEventListener('trail:show', publish, { signal })
  host.addEventListener('trail:hide', clearPointer, { signal })
  view.addEventListener('resize', renderPointers, { signal })
  view.addEventListener('pagehide', clear, { signal })
  view.addEventListener('pageshow', publish, { signal })
  signal.addEventListener(
    'abort',
    () => {
      clear()
      unsubscribeVisitors()
      unsubscribeCamp()
      unsubscribePointer()
      room.destroy()
      cursors.replaceChildren()
    },
    { once: true },
  )
  connection.textContent = 'visitors automatically walk to camp'
  publish()
}
