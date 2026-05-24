const stickerSrcs = [
  '/static/landing/rocky-monomyth/sticker.webp',
  '/static/landing/rocky-monomyth/sticker-computer.webp',
  '/static/landing/rocky-monomyth/sticker-wave.webp',
  '/static/landing/rocky-monomyth/sticker-emote.webp',
]
const landingPositionKey = 'garden:landing-pet-position:v2'
const petsEnabledKey = 'garden:pets-enabled:v1'
const hydrationReminderIntervalMs = 45 * 60 * 1000
const hydrationReminderVisibleMs = 22 * 1000
const edgePadding = 12
const defaultPetsEnabled = Number('__QUARTZ_PETS_DEFAULT_ENABLED__') === 1
const landingStickerWidth = 168
const landingStickerHeight = 178

type Point = { x: number; y: number }
type DragState = {
  pointerId: number
  startX: number
  startY: number
  originX: number
  originY: number
}

const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max)
const randomBetween = (min: number, max: number) =>
  max <= min ? min : min + Math.random() * (max - min)
const readPositiveInteger = (value: string | undefined, fallback: number) => {
  const parsed = Number(value)
  return Number.isInteger(parsed) && parsed > 0 ? parsed : fallback
}

const readPoint = (key: string): Point | null => {
  const raw = localStorage.getItem(key)
  if (!raw) return null

  const [x, y] = raw.split(',').map(Number)
  return Number.isFinite(x) && Number.isFinite(y) ? { x, y } : null
}

const writePoint = (key: string, point: Point) =>
  localStorage.setItem(key, `${Math.round(point.x)},${Math.round(point.y)}`)

const readPetsEnabled = () => {
  const stored = localStorage.getItem(petsEnabledKey)
  if (stored) return stored !== 'false'
  return defaultPetsEnabled
}

const writePetsEnabled = (enabled: boolean) =>
  localStorage.setItem(petsEnabledKey, enabled ? 'true' : 'false')

const notifyPetsEnabled = (enabled: boolean) => {
  const event: CustomEventMap['toast'] = new CustomEvent('toast', {
    detail: { message: `pets ${enabled ? 'on' : 'off'}` },
  })
  document.dispatchEvent(event)
}

const clampPoint = (element: HTMLElement, point: Point): Point => {
  const rect = element.getBoundingClientRect()
  const maxX = Math.max(edgePadding, window.innerWidth - rect.width - edgePadding)
  const maxY = Math.max(edgePadding, window.innerHeight - rect.height - edgePadding)

  return { x: clamp(point.x, edgePadding, maxX), y: clamp(point.y, edgePadding, maxY) }
}

const setPosition = (element: HTMLElement, point: Point) => {
  const next = clampPoint(element, point)
  element.style.position = 'fixed'
  element.style.left = `${Math.round(next.x)}px`
  element.style.top = `${Math.round(next.y)}px`
  element.style.right = 'auto'
  element.style.bottom = 'auto'
}

const persistPosition = (element: HTMLElement, key: string) => {
  const rect = element.getBoundingClientRect()
  writePoint(key, clampPoint(element, { x: rect.left, y: rect.top }))
}

const makeDraggable = (element: HTMLElement, storageKey?: string) => {
  if (element.dataset.petDraggable === 'true') return

  element.dataset.petDraggable = 'true'
  element.querySelectorAll('img').forEach(img => {
    img.draggable = false
  })

  const saved = storageKey ? readPoint(storageKey) : null
  if (saved) setPosition(element, saved)

  let dragState: DragState | null = null

  const startDrag = (event: PointerEvent) => {
    if (event.pointerType === 'mouse' && event.button !== 0) return
    if (event.target instanceof Element && event.target.closest('[data-pet-close]')) return

    const rect = element.getBoundingClientRect()
    setPosition(element, { x: rect.left, y: rect.top })
    dragState = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      originX: rect.left,
      originY: rect.top,
    }

    element.classList.add('is-dragging')
    element.setPointerCapture?.(event.pointerId)
    event.preventDefault()
  }

  const moveDrag = (event: PointerEvent) => {
    if (!dragState || event.pointerId !== dragState.pointerId) return

    setPosition(element, {
      x: dragState.originX + event.clientX - dragState.startX,
      y: dragState.originY + event.clientY - dragState.startY,
    })
  }

  const endDrag = (event: PointerEvent) => {
    if (!dragState || event.pointerId !== dragState.pointerId) return

    if (element.hasPointerCapture?.(event.pointerId)) {
      element.releasePointerCapture(event.pointerId)
    }

    dragState = null
    element.classList.remove('is-dragging')
    if (storageKey) persistPosition(element, storageKey)
  }

  const nudge = (event: KeyboardEvent) => {
    const directions: Record<string, readonly [number, number]> = {
      ArrowLeft: [-1, 0],
      ArrowRight: [1, 0],
      ArrowUp: [0, -1],
      ArrowDown: [0, 1],
    }
    const direction = directions[event.key]
    if (!direction) return

    event.preventDefault()
    const rect = element.getBoundingClientRect()
    const distance = event.shiftKey ? 48 : 16
    setPosition(element, {
      x: rect.left + direction[0] * distance,
      y: rect.top + direction[1] * distance,
    })
    if (storageKey) persistPosition(element, storageKey)
  }

  const clampOnResize = () => {
    const rect = element.getBoundingClientRect()
    setPosition(element, { x: rect.left, y: rect.top })
    if (storageKey) persistPosition(element, storageKey)
  }

  element.addEventListener('pointerdown', startDrag)
  element.addEventListener('pointermove', moveDrag)
  element.addEventListener('pointerup', endDrag)
  element.addEventListener('pointercancel', endDrag)
  element.addEventListener('keydown', nudge)
  window.addEventListener('resize', clampOnResize)

  window.addCleanup?.(() => {
    element.removeEventListener('pointerdown', startDrag)
    element.removeEventListener('pointermove', moveDrag)
    element.removeEventListener('pointerup', endDrag)
    element.removeEventListener('pointercancel', endDrag)
    element.removeEventListener('keydown', nudge)
    window.removeEventListener('resize', clampOnResize)
    delete element.dataset.petDraggable
  })
}

const placeHydrationPet = (element: HTMLElement) => {
  const rect = element.getBoundingClientRect()
  const maxX = Math.max(edgePadding, window.innerWidth - rect.width - edgePadding)
  const lowerY = Math.min(
    Math.max(96, edgePadding),
    Math.max(edgePadding, window.innerHeight - rect.height - edgePadding),
  )
  const maxY = Math.max(lowerY, window.innerHeight - rect.height - edgePadding)

  setPosition(element, { x: randomBetween(edgePadding, maxX), y: randomBetween(lowerY, maxY) })
}

const removeHydrationPets = () => {
  document.querySelectorAll<HTMLElement>('.hydration-pet').forEach(element => element.remove())
}

const ensureLandingPetImage = (element: HTMLElement) => {
  const existing = element.querySelector<HTMLImageElement>('img')
  if (existing) {
    existing.draggable = false
    existing.dataset.noPopover = 'true'
    return
  }

  const img = document.createElement('img')
  img.src = element.dataset.petSrc ?? stickerSrcs[0]
  img.alt = element.dataset.petAlt ?? ''
  img.width = readPositiveInteger(element.dataset.petWidth, landingStickerWidth)
  img.height = readPositiveInteger(element.dataset.petHeight, landingStickerHeight)
  img.loading = 'eager'
  img.decoding = 'async'
  img.draggable = false
  img.dataset.noPopover = 'true'
  element.append(img)
}

const removeLandingPetImage = (element: HTMLElement) => {
  element.querySelector('img')?.remove()
}

const spawnHydrationPet = () => {
  if (!readPetsEnabled()) return
  if (document.querySelector('.hydration-pet')) return

  const element = document.createElement('figure')
  element.className = 'hydration-pet'
  element.dataset.petWidget = 'true'
  element.tabIndex = 0
  element.setAttribute('role', 'group')
  element.setAttribute('aria-label', 'draggable rocky water reminder')

  const img = document.createElement('img')
  img.src = stickerSrcs[Math.floor(Math.random() * stickerSrcs.length)]
  img.alt = ''
  img.width = 168
  img.height = 178
  img.decoding = 'async'
  img.draggable = false
  img.dataset.noPopover = 'true'

  const bubble = document.createElement('figcaption')
  bubble.className = 'hydration-pet-bubble'
  bubble.setAttribute('role', 'status')

  const message = document.createElement('span')
  message.textContent = 'drink water'

  const close = document.createElement('button')
  close.className = 'hydration-pet-close'
  close.type = 'button'
  close.dataset.petClose = 'true'
  close.setAttribute('aria-label', 'dismiss drink water reminder')
  close.textContent = '×'

  bubble.append(message, close)
  element.append(img, bubble)
  document.body.append(element)
  placeHydrationPet(element)
  makeDraggable(element)

  const hideReminder = () => {
    element.classList.remove('is-reminding')
  }

  close.addEventListener('click', hideReminder)
  requestAnimationFrame(() => element.classList.add('is-visible'))

  window.addCleanup?.(() => {
    close.removeEventListener('click', hideReminder)
    element.remove()
  })
}

let hydrationReminderTimer: number | undefined

const shouldRenderHydrationPet = () => {
  const slug = document.body.dataset.slug ?? ''
  return readPetsEnabled() && Boolean(slug) && slug !== 'index' && slug !== '404'
}

const ensureHydrationPet = () => {
  if (!shouldRenderHydrationPet()) return null
  let element = document.querySelector<HTMLElement>('.hydration-pet')
  if (!element) {
    spawnHydrationPet()
    element = document.querySelector<HTMLElement>('.hydration-pet')
  }
  return element
}

const showHydrationReminder = () => {
  const element = ensureHydrationPet()
  if (!element) return

  element.classList.add('is-reminding')
  window.setTimeout(() => element.classList.remove('is-reminding'), hydrationReminderVisibleMs)
}

const clearHydrationReminder = () => {
  if (hydrationReminderTimer === undefined) return
  window.clearTimeout(hydrationReminderTimer)
  hydrationReminderTimer = undefined
}

const scheduleHydrationReminder = () => {
  if (hydrationReminderTimer !== undefined || !readPetsEnabled()) return

  hydrationReminderTimer = window.setTimeout(() => {
    hydrationReminderTimer = undefined
    showHydrationReminder()
    scheduleHydrationReminder()
  }, hydrationReminderIntervalMs)
}

const syncPets = () => {
  const enabled = readPetsEnabled()
  document.body.dataset.pets = enabled ? 'on' : 'off'

  const landingPet = document.querySelector<HTMLElement>('[data-pet-home]')
  if (landingPet) {
    if (enabled) {
      ensureLandingPetImage(landingPet)
      landingPet.hidden = false
      makeDraggable(landingPet, landingPositionKey)
    } else {
      landingPet.hidden = true
      removeLandingPetImage(landingPet)
    }
  }

  if (!enabled) {
    clearHydrationReminder()
    removeHydrationPets()
    return
  }

  ensureHydrationPet()
  scheduleHydrationReminder()
}

const setPetsEnabled = (enabled: boolean) => {
  writePetsEnabled(enabled)
  syncPets()
  notifyPetsEnabled(enabled)
}

document.addEventListener('petstoggle', event => {
  setPetsEnabled(event.detail.enabled ?? !readPetsEnabled())
})

document.addEventListener('nav', () => {
  syncPets()
})
