import { heart, petSvg, sprout } from '../../../util/mascot'
import { createPetGaze } from '../pet-gaze'
import {
  center,
  friendCenter,
  friendScale,
  friendState,
  friendTimers,
  height,
  later,
  petState,
  reducedMotion,
  type Point,
} from './actor-model'
import { friendPlay, play, renderFriend, say, showFriendBubble } from './appearance'
import { watch } from './attention'
import { friendPage, friendWatch } from './friend-motion'

const friendLines = [
  ['salut !', 'salut, toi.'],
  ['ça va ?', 'ça va bien.'],
  ['on lit ?', 'toujours.'],
  ['tu restes ?', 'je reste.'],
  ['merci d’être là.', 'avec plaisir.'],
] as const

// A heart in pen, rising from between the two and fading.
export const heartUp = (at: Point) => {
  if (reducedMotion()) return
  const el = document.createElement('div')
  el.className = 'garden-pet-heart'
  el.setAttribute('aria-hidden', 'true')
  el.innerHTML = `<svg viewBox="-0.7 -0.7 1.4 1.4">${heart(0.07, 'var(--dark)')}</svg>`
  el.style.left = `${(at.x + window.scrollX - 11).toFixed(1)}px`
  el.style.top = `${(at.y + window.scrollY - 11).toFixed(1)}px`
  document.body.append(el)
  el.addEventListener('animationend', () => el.remove(), { once: true })
}

// Lean in toward each other and back.
const nudge = (el: Element | null | undefined, dx: number, ms: number) =>
  el?.animate(
    [
      { translate: '0 0' },
      { translate: `${dx}px 0`, offset: 0.3 },
      { translate: `${dx}px 0`, offset: 0.7 },
      { translate: '0 0' },
    ],
    { duration: ms, easing: 'ease-in-out' },
  )

export const bounce = (el: Element | null | undefined, delay = 0) =>
  el?.animate([{ translate: '0 0' }, { translate: '0 -8px', offset: 0.4 }, { translate: '0 0' }], {
    duration: 360,
    delay,
    easing: 'cubic-bezier(0.3, 0.6, 0.5, 1)',
  })

// One of their moments together, when both rest and the reader is not busy with either.
function meet() {
  friendState.meetTimer = undefined
  if (performance.now() < petState.scrollQuietUntil) return scheduleMeet()
  if (document.querySelector('dialog:modal')) return scheduleMeet()
  const fc = friendCenter()
  const busy =
    !friendState.friend ||
    friendState.friend.hidden ||
    !fc ||
    !petState.root ||
    petState.root.hidden ||
    !petState.doc ||
    petState.hop ||
    friendState.friendHop ||
    petState.carry ||
    petState.speaking ||
    petState.reminding ||
    petState.hovering ||
    friendState.friendHovering ||
    document.hidden ||
    reducedMotion()
  if (busy) return scheduleMeet()
  const mc = center()
  // Face each other: -1 faces right, as drawn mirrored.
  petState.face = fc.x > mc.x ? -1 : 1
  friendState.friendFace = -petState.face
  petState.lookAt = fc
  petState.gaze = ''
  watch()
  friendWatch()
  const toward = fc.x > mc.x ? 1 : -1
  const pick = Math.random()
  // Apart, on different edges, they can only call over to each other.
  const close = Math.hypot(fc.x - mc.x, fc.y - mc.y) < 2.5 * petState.width
  const meeting = !close || pick >= 0.7 ? 'chat' : pick < 0.4 ? 'hug' : 'five'
  if (petState.root) petState.root.dataset.meeting = meeting
  let ms = 1900
  if (meeting === 'hug') {
    play('happy', 1800)
    friendPlay('happy', 1800)
    nudge(petState.svg, toward * 5, 1600)
    nudge(friendState.friendSvg, -toward * 5, 1600)
    later(
      () =>
        heartUp({ x: (mc.x + fc.x) / 2, y: Math.min(mc.y, fc.y) - height(friendScale()) * 0.45 }),
      380,
    )
  } else if (meeting === 'five') {
    play('cheer', 1100)
    friendPlay('cheer', 1100)
    bounce(petState.svg)
    bounce(friendState.friendSvg, 120)
    ms = 1300
  } else {
    const [first, reply] = friendLines[Math.floor(Math.random() * friendLines.length)]
    const friendFirst = Math.random() < 0.5
    if (friendFirst) showFriendBubble(first, 1500)
    else say(first, 1500)
    later(() => (friendFirst ? say(reply, 1500) : showFriendBubble(reply, 1500)), 1300)
    ms = 2900
  }
  later(() => {
    if (petState.root) delete petState.root.dataset.meeting
    petState.lookAt = null
    petState.gaze = ''
    watch()
    scheduleMeet()
  }, ms)
}

function scheduleMeet() {
  window.clearTimeout(friendState.meetTimer)
  if (!friendState.friend) return
  friendState.meetTimer = window.setTimeout(meet, 6000 + Math.random() * 8000)
}

export function spawnFriend() {
  if (friendState.friend || !friendPage()) return
  friendState.friendMarkup ||= petSvg(
    'var(--pet-friend-fill)',
    'var(--dark)',
    undefined,
    19,
  ).replace(
    '<g class="pet-body">',
    `<g class="pet-body"><g data-part="sprout">${sprout((0.08 * 0.042) / 0.055, 'var(--dark)')}</g>`,
  )
  const el = document.createElement('div')
  el.className = 'garden-pet garden-pet-friend'
  el.style.transformOrigin = '0 0'
  el.dataset.petWidget = ''
  el.hidden = true
  document.body.appendChild(el)
  friendState.friend = el
  friendState.friendSvg = null
  return () => {
    el.innerHTML = `<button type="button" class="garden-pet-button" aria-label="the pet's friend">${friendState.friendMarkup}</button><div class="garden-pet-bubble" role="status"><span class="garden-pet-bubble-text"></span></div>`
    friendState.friendSvg = el.querySelector('svg')
    friendState.friendGaze = friendState.friendSvg ? createPetGaze(friendState.friendSvg) : null
    renderFriend()
    requestAnimationFrame(() => el.classList.add('is-visible'))
    scheduleMeet()
  }
}

export function removeFriend() {
  friendState.friendHop?.cancel()
  friendState.friendHop = null
  for (const timer of friendTimers) window.clearTimeout(timer)
  friendTimers.clear()
  window.clearTimeout(friendState.meetTimer)
  friendState.meetTimer = undefined
  friendState.chat = null
  friendState.friend?.remove()
  friendState.friend = null
  friendState.friendSvg = null
  friendState.friendGaze = null
  friendState.friendDoc = null
  friendState.friendPerch = null
  friendState.friendHovering = false
  friendState.friendOverride = null
}
