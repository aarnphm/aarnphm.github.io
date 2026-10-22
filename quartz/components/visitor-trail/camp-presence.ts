import { playhtml } from 'playhtml'
import type { TrailVisitor } from './visitor-journey'
import { sceneSettings } from './scene-settings'
import { localUtcOffset, OFFSET_PATTERN } from './visitor-locale'
import { anonymousVisitor } from './visitor-name'
import { renderVisitorRegister, type VisitorVisit } from './visitor-register'

const SETTLE_MS = 400
const SETTLE_MAX_MS = 2000

export function connectCampPresence(
  scene: HTMLElement,
  host: HTMLElement,
  signal: AbortSignal,
  destination: number,
  onVisitors: (visitors: TrailVisitor[]) => void,
  campCount: () => number = () => sceneSettings.camps.sites,
) {
  const document = scene.ownerDocument
  const view = document.defaultView
  const root = document.querySelector<HTMLElement>('.homepage-trail')
  const count = root?.querySelector<HTMLElement>('[data-trail-visitors]')
  const cursors = root?.querySelector<HTMLElement>('.trail-shared-cursors')
  if (!view || !root || !count || !cursors) throw new Error('Camp presence needs its controls')
  const room = playhtml.createPresenceRoom('camp-visitors-v1')
  const pointers = new Map<string, HTMLElement>()
  const activeVisitors = new Map<string, TrailVisitor>()
  const visits: VisitorVisit[] = []
  const arrived = Date.now()
  let pointer: { x: number; y: number; touch: boolean } | null = null
  let pointerTimer = 0
  let sentAt = -Infinity
  // The room's peers arrive over a moment after joining, while the first render sees only this
  // visitor, so the count would read 1 and then jump. It stays on the connecting dot until the
  // presence stream has been quiet for SETTLE_MS, or for at most SETTLE_MAX_MS.
  const joined = view.performance.now()
  let settled = false
  let settleTimer = 0
  const settle = () => {
    settleTimer = 0
    settled = true
    renderVisitors()
  }

  const renderVisitors = () => {
    const visitors: TrailVisitor[] = []
    const now = Date.now()
    for (const [id, peer] of room.presence.getPresences()) {
      // A peer whose identity has not arrived is keyed by its connection; it is counted once its
      // identity folds it into the visitor it belongs to.
      if (!peer.isMe && !peer.playerIdentity) continue
      const previous = activeVisitors.get(id)
      // PlayHTML expires stamped fields in background tabs before their socket identity leaves.
      if (peer.visiting !== true && !(peer.visiting === undefined && previous)) continue
      const campsite = peer.campsite
      const camp =
        campsite &&
        typeof campsite === 'object' &&
        'camp' in campsite &&
        typeof campsite.camp === 'number' &&
        Number.isFinite(campsite.camp)
          ? Math.max(0, Math.min(campCount() - 1, Math.floor(campsite.camp)))
          : (previous?.camp ?? 0)
      visitors.push({ id, own: peer.isMe, camp })
      const joined =
        typeof peer.trailArrival === 'number' &&
        Number.isFinite(peer.trailArrival) &&
        peer.trailArrival >= 0 &&
        peer.trailArrival <= now + 300000
          ? peer.trailArrival
          : null
      const offset =
        typeof peer.trailOffset === 'string' && OFFSET_PATTERN.test(peer.trailOffset)
          ? peer.trailOffset
          : null
      const visit = visits.find(visit => visit.id === id && visit.left === null)
      if (visit && offset) visit.offset = offset
      if (!visit)
        visits.push({
          id,
          own: peer.isMe,
          arrived: joined ?? now,
          observed: joined === null,
          left: null,
          offset,
        })
      else if (joined !== null) {
        visit.arrived = visit.observed ? joined : Math.min(visit.arrived, joined)
        visit.observed = false
      }
    }
    const present = new Set(visitors.map(visitor => visitor.id))
    activeVisitors.clear()
    for (const visitor of visitors) activeVisitors.set(visitor.id, visitor)
    for (const visit of visits) if (visit.left === null && !present.has(visit.id)) visit.left = now
    const departed = visits
      .filter(visit => visit.left !== null)
      .sort((a, b) => (b.left ?? 0) - (a.left ?? 0))
    for (const visit of departed.slice(16)) visits.splice(visits.indexOf(visit), 1)
    renderVisitorRegister(root, visits)
    onVisitors(visitors.sort((a, b) => a.id.localeCompare(b.id)))
    if (!settled) {
      view.clearTimeout(settleTimer)
      const left = SETTLE_MAX_MS - (view.performance.now() - joined)
      settleTimer = view.setTimeout(settle, Math.max(0, Math.min(SETTLE_MS, left)))
      return
    }
    count.textContent = String(visitors.length)
    count.dataset.state = 'ready'
    count.setAttribute(
      'aria-label',
      `${visitors.length} ${visitors.length === 1 ? 'visitor' : 'visitors'} here`,
    )
  }
  const renderPointers = () => {
    const shown = new Set<string>()
    if (!document.hidden)
      for (const [id, peer] of room.presence.getPresences()) {
        const pointer = peer.trailPointer
        if (
          peer.isMe ||
          !activeVisitors.has(id) ||
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
          const label = document.createElement('span')
          label.textContent = anonymousVisitor(id)
          cursor.append(label)
          pointers.set(id, cursor)
          cursors.append(cursor)
        }
        cursor.style.left = `${Math.max(0, Math.min(1, pointer.x)) * view.innerWidth}px`
        cursor.style.top = `${Math.max(0, Math.min(1, pointer.y)) * view.innerHeight}px`
        cursor.dataset.flipped = String(pointer.x > 0.8)
        cursor.dataset.raised = String(pointer.y > 0.85)
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
      if (document.hidden || !view.innerWidth || !view.innerHeight) return
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
    room.presence.setMyPresence('trailArrival', arrived)
    // Only the UTC offset is shared; zone and country stay in this browser.
    room.presence.setMyPresence('trailOffset', localUtcOffset(view))
    room.presence.setMyPresence('campsite', { camp: destination, joined: true })
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
  const unsubscribeArrival = room.presence.onPresenceChange('trailArrival', renderVisitors)
  const unsubscribeOffset = room.presence.onPresenceChange('trailOffset', renderVisitors)
  // A disconnected identity may have no live fields left to trigger a channel subscription.
  const membershipTimer = view.setInterval(() => {
    const peers = room.presence.getPresences()
    if ([...activeVisitors.keys()].some(id => !peers.has(id))) {
      renderVisitors()
      renderPointers()
    }
  }, 1000)
  host.addEventListener('trail:show', publish, { signal })
  host.addEventListener(
    'trail:destination',
    event => {
      const next = 'detail' in event ? event.detail : null
      if (typeof next !== 'number' || !Number.isFinite(next)) return
      destination = Math.max(0, Math.min(campCount() - 1, Math.floor(next)))
      room.presence.setMyPresence('campsite', { camp: destination, joined: true })
    },
    { signal },
  )
  document.addEventListener(
    'visibilitychange',
    () => {
      if (document.hidden) clearPointer()
      else {
        renderVisitors()
        renderPointers()
      }
    },
    { signal },
  )
  view.addEventListener('resize', renderPointers, { signal })
  view.addEventListener('pagehide', clear, { signal })
  view.addEventListener('pageshow', publish, { signal })
  signal.addEventListener(
    'abort',
    () => {
      clear()
      view.clearTimeout(settleTimer)
      view.clearInterval(membershipTimer)
      unsubscribeVisitors()
      unsubscribeCamp()
      unsubscribePointer()
      unsubscribeArrival()
      unsubscribeOffset()
      room.destroy()
      cursors.replaceChildren()
    },
    { once: true },
  )
  publish()
}
