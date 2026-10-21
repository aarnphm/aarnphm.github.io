import { petSvg } from '../../../util/mascot'
import { readPetSpot } from '../../../util/pet-spot'
import { createPetGaze } from '../pet-gaze'
import { center, friendState, petState, scheduleWander, viewWidth } from './actor-model'
import { armQuietTimer, showBubble } from './appearance'
import { hideReminder, scheduleReminder } from './attention'
import { friendFollow, friendPage } from './friend-motion'
import { removeFriend, spawnFriend } from './friend-social'
import { clearGround } from './ground-trace'
import { relocate, savePosition } from './perch-motion'
import { clearTrailActor } from './trail-pose'

export const petsEnabledKey = 'garden:pets-enabled:v1'

// On unless the reader turned pets off from the palette.
export const readPetsEnabled = () => localStorage.getItem(petsEnabledKey) !== 'false'

const spawn = () => {
  petState.markup ||= petSvg()
  const el = document.createElement('div')
  el.className = 'garden-pet'
  el.style.transformOrigin = '0 0'
  el.dataset.petWidget = ''
  document.body.appendChild(el)
  petState.root = el
  petState.svg = null
  petState.gazeArt = null
  // The pet is absolutely positioned in document coordinates, so the window's scroll carries it
  // along its edge for free. The containing block may sit off the document origin; measure it once.
  const rect = el.getBoundingClientRect()
  petState.width = rect.width || 72
  petState.unit = petState.width / viewWidth
  petState.origin = { x: rect.left + window.scrollX, y: rect.top + window.scrollY }
  // Placement reads the page before the SVG's thousands of nodes need style and layout.
  return () => {
    el.innerHTML = `<button type="button" class="garden-pet-button" aria-label="garden pet">${petState.markup}</button><div class="garden-pet-bubble" role="status"><span class="garden-pet-bubble-text"></span><button type="button" class="garden-pet-close" aria-label="dismiss reminder">×</button></div>`
    petState.svg = el.querySelector('svg')
    petState.gazeArt = petState.svg ? createPetGaze(petState.svg) : null
    showBubble()
    requestAnimationFrame(() => el.classList.add('is-visible'))
  }
}

const despawn = () => {
  clearTrailActor()
  petState.hop?.cancel()
  petState.hop = null
  petState.landAfterJet = false
  petState.trailWalk = null
  petState.trailJourney = null
  petState.afterMove = null
  if (petState.doc || petState.carry) savePosition()
  petState.carry = null
  for (const timer of [
    petState.stateTimer,
    petState.dwellTimer,
    petState.scrollTimer,
    petState.wanderTimer,
    petState.reminderTimer,
  ])
    window.clearTimeout(timer)
  petState.reminderTimer = undefined
  petState.resizeObserver?.disconnect()
  hideReminder()
  clearGround()
  petState.root?.remove()
  petState.root = null
  petState.svg = null
  petState.gazeArt = null
  petState.doc = null
  petState.perch = null
  removeFriend()
}

export const syncPets = () => {
  let placed = false
  const enabled = readPetsEnabled()
  document.body.dataset.pets = enabled ? 'on' : 'off'
  // The 404 scene paints its own pet into the ruin, starting from where this one stood.
  if (!enabled || document.querySelector('.nf-scene')) {
    despawn()
    return
  }
  // The friend lives on the friendship page alone; the morph drops it with the pet.
  if (friendState.friend && (!friendState.friend.isConnected || !friendPage())) removeFriend()
  // The SPA morphs <body> on navigation, which drops the pet and the box it held. It comes back
  // near where it was on screen and drops onto the nearest edge of the new page.
  if (!petState.root || !petState.root.isConnected) {
    clearTrailActor()
    const before = ((petState.doc || petState.carry) && !petState.hop
      ? center()
      : readPetSpot()) ?? { x: window.innerWidth * 0.8, y: window.innerHeight * 0.7 }
    petState.hop?.cancel()
    petState.hop = null
    petState.doc = null
    petState.carry = null
    petState.perch = null
    petState.trailWalk = null
    petState.trailJourney = null
    petState.afterMove = null
    const mount = spawn()
    const mountFriend = spawnFriend()
    relocate(before, null)
    placed = true
    mount()
    mountFriend?.()
  } else if (petState.perch?.el && !petState.perch.el.isConnected) {
    relocate(center(), null)
    placed = true
  }
  if (friendPage() && !friendState.friend) {
    const mountFriend = spawnFriend()
    if (petState.perch && petState.doc && !petState.hop)
      friendFollow(petState.perch, petState.doc, 'appear')
    mountFriend?.()
  }
  if (petState.lastMove === 0) petState.lastMove = performance.now()
  armQuietTimer()
  scheduleWander()
  scheduleReminder()
  return placed
}
