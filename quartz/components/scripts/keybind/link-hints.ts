const HINT_CHARS = 'asdfghjklzxcvbnm'

type Hint = { element: HTMLElement; key: string; marker: HTMLElement }

function hintKeys(count: number): string[] {
  let width = 1
  while (width < 4 && count > HINT_CHARS.length ** width) width++
  return Array.from({ length: Math.min(count, HINT_CHARS.length ** width) }, (_, index) => {
    let key = ''
    for (let digit = 0; digit < width; digit++) {
      key = HINT_CHARS[index % HINT_CHARS.length] + key
      index = Math.floor(index / HINT_CHARS.length)
    }
    return key
  })
}

function isVisible(element: HTMLElement): boolean {
  if (
    element.closest('[inert], [aria-hidden="true"]') ||
    element.matches(':disabled, [aria-disabled="true"]') ||
    !element.checkVisibility({ visibilityProperty: true })
  )
    return false
  const rect = element.getBoundingClientRect()
  return (
    rect.width > 0 &&
    rect.height > 0 &&
    rect.top < innerHeight &&
    rect.bottom > 0 &&
    rect.left < innerWidth &&
    rect.right > 0 &&
    (element.classList.contains('clipboard-button') || getComputedStyle(element).opacity !== '0')
  )
}

function hintRoot(): Element | null {
  const arena = document.querySelector<HTMLElement>('#arena-modal.active')
  if (arena && getComputedStyle(arena).display !== 'none') {
    const content = arena.querySelector('.arena-modal-content')
    if (content) return content
  }
  const reader = document.querySelector<HTMLElement>('.reader.active')
  if (reader && getComputedStyle(reader).display !== 'none') {
    const content = reader.querySelector('.reader-content')
    if (content) return content
  }
  return document.querySelector('#quartz-root')
}

function clickableElements(): HTMLElement[] {
  const root = hintRoot()
  if (!root) return []
  const elements = new Set<HTMLElement>()
  for (const selector of [
    'a[href]',
    'button',
    'span[role="button"]',
    'div[role="button"]',
    '[onclick]',
    'input[type=submit]',
    'input[type=button]',
    'label',
    '.clipboard-button',
  ]) {
    for (const candidate of root.querySelectorAll<HTMLElement>(selector)) {
      if (candidate.closest('#toc-vertical')) continue
      const toggle = candidate.closest<HTMLElement>('[data-collapse-toggle]')
      // A heading gets one action: its authored title link, or its section toggle.
      const element = toggle
        ? (Array.from(
            toggle.querySelectorAll<HTMLAnchorElement>('a[href]:not([data-role="anchor"])'),
          ).find(isVisible) ?? toggle)
        : candidate
      if (isVisible(element)) elements.add(element)
    }
  }
  return [...elements]
}

export function createLinkHints(signal: AbortSignal) {
  let hints: Hint[] = []
  let prefix = ''
  const exiting = new Set<HTMLElement>()

  const remove = (marker: HTMLElement) => {
    marker.remove()
    exiting.delete(marker)
  }

  const dismiss = () => {
    const previous = hints
    hints = []
    prefix = ''
    for (const { marker } of previous) {
      marker.className = 'link-hint-marker-exiting'
      const animations = marker.getAnimations()
      if (animations.length === 0) remove(marker)
      else {
        exiting.add(marker)
        void Promise.allSettled(animations.map(animation => animation.finished)).then(() =>
          remove(marker),
        )
      }
    }
  }

  const dispose = () => {
    for (const { marker } of hints) marker.remove()
    for (const marker of exiting) marker.remove()
    exiting.clear()
    hints = []
    prefix = ''
  }

  const show = () => {
    dispose()
    if (signal.aborted) return
    const elements = clickableElements()
    const keys = hintKeys(elements.length)
    hints = elements.flatMap((element, index) => {
      const key = keys[index]
      if (!key) return []
      const marker = document.createElement('kbd')
      marker.className = 'link-hint-marker'
      marker.textContent = key
      marker.setAttribute('aria-hidden', 'true')
      const rect = element.getBoundingClientRect()
      marker.style.top = `${Math.max(0, rect.top)}px`
      marker.style.left = `${Math.max(0, rect.left)}px`
      document.body.append(marker)
      return [{ element, key, marker }]
    })
    const sizes = hints.map(({ marker }) => ({
      width: marker.offsetWidth,
      height: marker.offsetHeight,
    }))
    hints.forEach(({ marker }, index) => {
      marker.style.left = `${Math.min(parseFloat(marker.style.left), innerWidth - sizes[index].width)}px`
      marker.style.top = `${Math.min(parseFloat(marker.style.top), innerHeight - sizes[index].height)}px`
    })
  }

  const handleKeydown = (event: KeyboardEvent): boolean => {
    if (hints.length === 0) return false
    if (event.key === 'Escape') {
      event.preventDefault()
      dismiss()
    } else if (
      event.key.length === 1 &&
      HINT_CHARS.includes(event.key.toLowerCase()) &&
      !event.ctrlKey &&
      !event.metaKey &&
      !event.altKey
    ) {
      event.preventDefault()
      if (event.repeat) return true
      prefix += event.key.toLowerCase()
      const matches = hints.filter(hint => hint.key.startsWith(prefix))
      if (matches.length === 0) dismiss()
      else if (matches.length === 1 && matches[0].key === prefix) {
        const { element } = matches[0]
        dismiss()
        if (element.isConnected) element.click()
      } else {
        for (const { key, marker } of hints) {
          marker.hidden = !key.startsWith(prefix)
          if (marker.hidden) continue
          const typed = document.createElement('span')
          typed.className = 'link-hint-typed'
          typed.textContent = key.slice(0, prefix.length)
          marker.replaceChildren(typed, document.createTextNode(key.slice(prefix.length)))
        }
      }
    }
    return true
  }

  window.addEventListener('scroll', dismiss, { passive: true, capture: true, signal })
  window.addEventListener('resize', dismiss, { signal })
  document.addEventListener('pointerdown', dismiss, { signal })
  document.addEventListener('focusin', dismiss, { signal })
  window.addEventListener('blur', dismiss, { signal })
  signal.addEventListener('abort', dispose, { once: true })

  return { show, handleKeydown }
}
