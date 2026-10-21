import { trailScene } from '../../util/trail-scene'
import {
  center,
  friendCenter,
  friendScale,
  friendState,
  height,
  later,
  petState,
  railY,
  readerMode,
  reducedMotion,
  type Point,
} from './pets/actor-model'
import {
  friendPlay,
  loadWeather,
  play,
  render,
  renderFriend,
  showBubble,
  syncStorm,
} from './pets/appearance'
import {
  hideReminder,
  onPointerDown,
  onPointerMove,
  onSettle,
  onTriWindows,
  watch,
} from './pets/attention'
import { bounce, heartUp } from './pets/friend-social'
import { recheckPerch, relocate, savePosition } from './pets/perch-motion'
import { petsEnabledKey, readPetsEnabled, syncPets } from './pets/presence'
import { endCarry, onViewportWidthChange, startCarry } from './pets/scroll-carry'
import { driveKeys, onDriveKey, onDriveKeyUp } from './pets/trail-drive'
import { watchTrail } from './pets/trail-lifecycle'

const scrollTimeline = typeof CSS !== 'undefined' && CSS.supports('animation-timeline', 'scroll()')

document.addEventListener('petstoggle', event => {
  const enabled = event.detail.enabled ?? !readPetsEnabled()
  localStorage.setItem(petsEnabledKey, enabled ? 'true' : 'false')
  syncPets()
  const toast: CustomEventMap['toast'] = new CustomEvent('toast', {
    detail: { message: `pets ${enabled ? 'on' : 'off'}` },
  })
  document.dispatchEvent(toast)
})

document.addEventListener('nav', () => {
  // Reading ready can update layout. Do it before syncPets inserts the SVG artwork.
  const fontsReady = document.fonts?.ready
  const controller = new AbortController()
  const { signal } = controller
  petState.scrollQuietUntil = 0
  const deferGeometry = () => {
    petState.scrollQuietUntil = performance.now() + 500
  }
  document.addEventListener('wheel', deferGeometry, { passive: true, signal })
  document.addEventListener('scroll', deferGeometry, { capture: true, passive: true, signal })
  let viewportWidth = window.innerWidth
  document.addEventListener('animationstart', syncStorm, { signal })
  petState.trailSpaceSignal = signal
  petState.trailSpaces = new WeakMap()
  const placed = syncPets()
  document.addEventListener('keydown', onDriveKey, { signal })
  document.addEventListener('keyup', onDriveKeyUp, { signal })
  window.addEventListener('blur', () => driveKeys.clear(), { signal })
  void loadWeather(signal)
  const scene = document.querySelector<HTMLElement>('.homepage-trail-scene')
  const trailNav = { trailWanted: false }
  const { queueTrail, clearClock, closeTrail } = watchTrail(trailNav, scene, placed, signal)
  document.addEventListener('pointermove', onPointerMove, { passive: true, signal })
  document.addEventListener('pointerdown', onPointerDown, { passive: true, signal })
  document.addEventListener(
    'click',
    event => {
      const hit = event.target as Element | null
      // The friend cheers; the pet turns to it and waves back.
      if (
        hit &&
        friendState.friend?.contains(hit) &&
        hit.closest('.garden-pet-button') &&
        !friendState.friendHop
      ) {
        const fc = friendCenter()
        friendPlay('cheer', 1400)
        bounce(friendState.friendSvg)
        if (fc && petState.root && petState.doc && !petState.hop && !petState.carry) {
          petState.lookAt = fc
          petState.gaze = ''
          watch()
          play('wave', 1400)
          heartUp({ x: fc.x, y: fc.y - height(friendScale()) * 0.5 })
          later(() => {
            petState.lookAt = null
            petState.gaze = ''
            watch()
          }, 1400)
        }
        return
      }
      if (!petState.root || !hit || !petState.root.contains(hit)) return
      if (!hit.closest('.garden-pet-button')) return
      // A click on the pet dismisses the reminder it is holding up.
      if (petState.reminding) hideReminder()
      if (!petState.hop) play('cheer', 1400)
    },
    { signal },
  )
  document.addEventListener(
    'pointerover',
    event => {
      if (event.pointerType === 'touch') return
      const overFriend = !!friendState.friend
        ?.querySelector('.garden-pet-button')
        ?.contains(event.target as Node)
      if (overFriend !== friendState.friendHovering) {
        friendState.friendHovering = overFriend
        renderFriend()
      }
      const over = !!petState.root
        ?.querySelector('.garden-pet-button')
        ?.contains(event.target as Node)
      if (over === petState.hovering) return
      petState.hovering = over
      render()
    },
    { passive: true, signal },
  )
  window.addEventListener(
    'scroll',
    () => {
      window.clearTimeout(petState.scrollTimer)
      if (reducedMotion()) {
        clearClock()
        petState.scrollTimer = window.setTimeout(onSettle, 220)
        return
      }
      if (petState.trailWalk || petState.trailJourney) trailNav.trailWanted = true
      petState.landAfterJet = false
      if (!petState.carry) startCarry()
      else if (petState.root && !scrollTimeline)
        petState.root.style.translate = `0 ${railY(petState.carry).toFixed(1)}px`
      petState.scrollTimer = window.setTimeout(() => {
        endCarry()
        if (trailNav.trailWanted) queueTrail()
      }, 200)
    },
    { passive: true, signal },
  )
  window.addEventListener(
    'resize',
    () => {
      if (viewportWidth !== window.innerWidth) {
        viewportWidth = window.innerWidth
        onViewportWidthChange()
      }
      clearClock()
      window.clearTimeout(petState.scrollTimer)
      petState.scrollTimer = window.setTimeout(() => {
        if (petState.carry) endCarry()
        else onSettle()
        if (trailNav.trailWanted) queueTrail()
      }, 220)
    },
    { passive: true, signal },
  )
  window.addEventListener('pagehide', () => (petState.doc || petState.carry) && savePosition(), {
    signal,
  })
  // Glasses going on or coming off play their push-on or lift-off once; render keeps them on.
  let readingTimer: number | undefined
  const swapGlasses = (on: boolean) => {
    if (!petState.svg) return
    clearTimeout(readingTimer)
    const change = on ? 'data-reading-on' : 'data-reading-off'
    petState.svg.removeAttribute('data-reading-on')
    petState.svg.removeAttribute('data-reading-off')
    petState.svg.setAttribute(change, '')
    readingTimer = window.setTimeout(() => petState.svg?.removeAttribute(change), 420)
  }
  // While the French voice loads and says a phrase, the pet reads it off the page in its reading
  // glasses, eyes on the phrase, then says it in its bubble.
  let readLook: Point | null = null
  const onSpeech = (event: CustomEventMap['speech']) => {
    const { state, text, lang, phrase } = event.detail
    const was = !!petState.speaking
    petState.speaking = state === 'idle' ? null : { text, lang, playing: state === 'playing' }
    if (!!petState.speaking !== was && !readerMode()) swapGlasses(!!petState.speaking)
    if (petState.lookAt === readLook) petState.lookAt = null
    const rect = petState.speaking ? phrase?.getBoundingClientRect() : undefined
    readLook = rect ? { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 } : null
    if (readLook) petState.lookAt = readLook
    petState.gaze = ''
    watch()
    showBubble()
  }
  document.addEventListener('speech', onSpeech)
  // Presenting puts the slides in full screen, where only that element's subtree shows, so the
  // pet moves in with it and back out after.
  const onFullscreen = () => {
    if (!petState.root || document.body.dataset.slides !== 'true') return
    const host =
      document.fullscreenElement instanceof HTMLElement ? document.fullscreenElement : document.body
    if (petState.root.parentElement === host) return
    petState.hop?.cancel()
    petState.hop = null
    host.append(petState.root)
    petState.root.style.transform = 'none'
    const rect = petState.root.getBoundingClientRect()
    petState.origin = { x: rect.left + window.scrollX, y: rect.top + window.scrollY }
    petState.doc = null
    petState.perch = null
    // The window keeps its old size for a frame or two after the event (27 ms leaving full screen
    // in Chromium), so place the pet once the size has held for a few frames.
    let size = ''
    let still = 0
    const start = performance.now()
    const place = () => {
      const now = `${window.innerWidth}x${window.innerHeight}`
      still = now === size ? still + 1 : 0
      size = now
      if (still < 4 && performance.now() - start < 600) requestAnimationFrame(place)
      else relocate(center(), null)
    }
    requestAnimationFrame(place)
  }
  document.addEventListener('fullscreenchange', onFullscreen, { signal })
  // Reader mode puts its glasses on (render reads the mode).
  let reflowTimer: number | undefined
  const onReaderMode = (event: CustomEventMap['readermodechange']) => {
    if (!petState.svg) return
    clearTimeout(reflowTimer)
    // Only the toggle animates; a pet spawned on a new page is already wearing them, and a pet
    // reading a phrase aloud keeps them on.
    const on = event.detail.mode === 'on'
    if (!petState.speaking && (on || petState.svg.hasAttribute('data-reading'))) swapGlasses(on)
    render()
    // The text reflows into the freed width, under the pet as often as not; once it has settled,
    // a pet whose edge went away or now covers text finds another.
    reflowTimer = window.setTimeout(recheckPerch, 300)
  }
  document.addEventListener('readermodechange', onReaderMode)
  // Content that lands after the pet has perched (a course page's hydrated rows, the lecture
  // workspace bar) or text that reflows once the web fonts load can end up under it. For the page's
  // first seconds, the perch is re-checked once the page goes quiet; the pet's own nodes don't count.
  let lateTimer: number | undefined
  let latePending = false
  // A pet still hopping onto its first edge is checked once it lands.
  const recheckSoon = (ms = 400) => {
    clearTimeout(lateTimer)
    lateTimer = window.setTimeout(() => {
      const drawing = trailScene(document)
      if (
        drawing &&
        (drawing.dataset.trailOpening !== 'complete' || drawing.dataset.trailReveal !== undefined)
      ) {
        latePending = true
        return
      }
      latePending = false
      if (petState.hop) recheckSoon(200)
      else recheckPerch()
    }, ms)
  }
  const resumeRecheck = () => latePending && recheckSoon(0)
  scene?.addEventListener('trail:prepared', resumeRecheck, { signal })
  scene?.addEventListener('trail:reveal-end', resumeRecheck, { signal })
  scene?.addEventListener('trail:hide', resumeRecheck, { signal })
  const late = new MutationObserver(records => {
    const outside = records.some(r => {
      const el = r.target instanceof Element ? r.target : r.target.parentElement
      return !el?.closest('.garden-pet, .pet-ground, .garden-pet-heart')
    })
    if (outside) recheckSoon()
  })
  late.observe(document.body, { childList: true, subtree: true })
  const lateStop = window.setTimeout(() => late.disconnect(), 5000)
  void fontsReady?.then(() => {
    if (!signal.aborted) recheckSoon()
  })
  // Each slide brings its own text, which may now sit under the presenting pet (or clear it).
  let slideTimer: number | undefined
  const onSlide = () => {
    if (!petState.root?.parentElement?.classList.contains('is-presenting')) return
    clearTimeout(slideTimer)
    // Past the slide's enter transition, so the text is measured where it rests.
    slideTimer = window.setTimeout(() => relocate(center(), null), 450)
  }
  document.addEventListener('slidechange', onSlide)
  document.addEventListener('contentdecrypted', onSlide)
  const windows = new MutationObserver(onTriWindows)
  for (const tri of document.querySelectorAll('.triathlon'))
    windows.observe(tri, { attributes: true, attributeFilter: ['class'] })
  window.addCleanup(() => {
    closeTrail()
    controller.abort()
    if (petState.trailSpaceSignal === signal) {
      petState.trailSpaceSignal = undefined
      petState.trailSpaces = new WeakMap()
    }
    document.removeEventListener('speech', onSpeech)
    document.removeEventListener('readermodechange', onReaderMode)
    clearTimeout(readingTimer)
    clearTimeout(reflowTimer)
    late.disconnect()
    clearTimeout(lateStop)
    clearTimeout(lateTimer)
    document.removeEventListener('slidechange', onSlide)
    document.removeEventListener('contentdecrypted', onSlide)
    clearTimeout(slideTimer)
    petState.speaking = null
    readLook = null
    windows.disconnect()
    window.clearTimeout(petState.windowTimer)
    petState.lookAt = null
  })
})
