import { playhtml } from 'playhtml'
import type { CampVisitor } from './camp-community'

export type CampDestination = { value: () => number; onChange: (listener: () => void) => void }

export function connectCampPresence(
  scene: HTMLElement,
  signal: AbortSignal,
  destination: CampDestination,
  onVisitors: (visitors: CampVisitor[]) => void,
) {
  const document = scene.ownerDocument
  const count = document.querySelector<HTMLElement>('[data-trail-visitors]')
  const join = scene.querySelector<HTMLButtonElement>('[data-camp-join]')
  const connection = scene.querySelector<HTMLElement>('[data-camp-connection]')
  if (!count || !join || !connection) throw new Error('Camp presence needs its controls')
  const room = playhtml.createPresenceRoom('camp-visitors-v1')
  let joined = false
  room.presence.setMyPresence('visiting', true)
  const unsubscribeVisitors = room.presence.onPresenceChange('visiting', peers => {
    const total = new Set([...peers].filter(([, peer]) => peer.visiting === true).map(([id]) => id))
      .size
    count.textContent = String(total)
    count.dataset.state = 'ready'
    count.setAttribute('aria-label', `${total} ${total === 1 ? 'visitor' : 'visitors'} here`)
  })
  const publish = () =>
    room.presence.setMyPresence(
      'campsite',
      joined ? { camp: destination.value(), joined: true } : null,
    )
  const unsubscribeCamp = room.presence.onPresenceChange('campsite', peers => {
    const visitors: CampVisitor[] = []
    for (const [id, peer] of peers) {
      const value = peer.campsite
      if (
        !value ||
        typeof value !== 'object' ||
        !('joined' in value) ||
        value.joined !== true ||
        !('camp' in value) ||
        typeof value.camp !== 'number' ||
        !Number.isFinite(value.camp)
      )
        continue
      visitors.push({ id, own: peer.isMe, camp: Math.max(0, Math.min(2, Math.floor(value.camp))) })
    }
    onVisitors(visitors.sort((a, b) => a.id.localeCompare(b.id)))
  })
  join.disabled = false
  connection.textContent = 'visitors can join this camp'
  join.addEventListener(
    'click',
    () => {
      joined = !joined
      join.setAttribute('aria-pressed', String(joined))
      join.textContent = joined ? 'leave camp' : 'join camp'
      publish()
    },
    { signal },
  )
  destination.onChange(() => {
    if (!signal.aborted) publish()
  })
  const clear = () => {
    room.presence.setMyPresence('visiting', null)
    room.presence.setMyPresence('campsite', null)
  }
  document.defaultView?.addEventListener('pagehide', clear, { signal })
  document.defaultView?.addEventListener(
    'pageshow',
    () => {
      room.presence.setMyPresence('visiting', true)
      publish()
    },
    { signal },
  )
  signal.addEventListener(
    'abort',
    () => {
      clear()
      unsubscribeVisitors()
      unsubscribeCamp()
      room.destroy()
    },
    { once: true },
  )
}
