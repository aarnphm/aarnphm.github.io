import { registerEscapeHandler } from './util'

type Browser = 'Safari' | 'Chrome' | 'Firefox' | 'Edge' | 'Opera' | 'Other'

const detectBrowser = (): Browser => {
  const userAgent = window.navigator.userAgent.toLowerCase()

  if (userAgent.includes('safari') && !userAgent.includes('chrome')) {
    return 'Safari'
  } else if (userAgent.includes('edg')) {
    return 'Edge'
  } else if (userAgent.includes('firefox')) {
    return 'Firefox'
  } else if (userAgent.includes('opr') || userAgent.includes('opera')) {
    return 'Opera'
  } else if (userAgent.includes('chrome')) {
    return 'Chrome'
  }
  return 'Other'
}

const isMacOS = (): boolean => {
  return window.navigator.userAgent.toLowerCase().includes('mac')
}

document.addEventListener('nav', async () => {
  const keybind = document.getElementsByClassName('keybind')[0] as HTMLDivElement | null
  if (!keybind) return

  const container = keybind.querySelector<HTMLDivElement>('#shortcut-container')
  const shortcutKey = keybind.querySelector('#shortcut-key') as HTMLElement

  const showContainer = () => {
    container?.classList.add('active')
    container!.style.visibility = 'visible'
  }

  const hideContainer = () => {
    container?.classList.remove('active')
    container!.style.visibility = 'hidden'
  }

  async function shortcutHandler(e: HTMLElementEventMap['keydown']) {
    if (!shortcutKey) return
    for (const binding of JSON.parse(shortcutKey.dataset.mapping as string)) {
      const [, key] = binding.split('--')

      if (e.key === key && (e.ctrlKey || e.metaKey)) {
        e.preventDefault()
        const containerOpen = container?.classList.contains('active')
        if (containerOpen) {
          hideContainer()
        } else {
          showContainer()
        }
        break
      }
    }
  }

  document.addEventListener('keydown', shortcutHandler)
  window.addCleanup(() => document.removeEventListener('keydown', shortcutHandler))
  registerEscapeHandler(keybind, hideContainer)
})

type MapAction = string | (() => void)
const _mapping: Map<string, MapAction> = new Map()
_mapping.set('\\', '/')
_mapping.set('j', '/curius')

const aliases: Record<string, { mac: string; def: string }> = {
  recherche: { mac: '/', def: 'k' },
  graphique: { mac: ';', def: 'g' },
}

// Scroll amount in pixels per key press
const SCROLL_AMOUNT_SMALL = 120 // for j/k - larger jumps
const SCROLL_AMOUNT_LARGE = 360 // for ctrl-e/y

// Vimium detection storage key
const VIMIUM_DISMISSED_KEY = 'vimium-warning-dismissed'

// Function to detect if Vimium or similar vim extensions are installed
async function detectVimiumExtension(): Promise<boolean> {
  // Chrome extensions inject content after page load, so we need to wait
  await new Promise(resolve => setTimeout(resolve, 500))

  // Check for Vimium-specific DOM elements that get injected
  const vimiumSelectors = [
    'div[class*="vimium"]',
    'div[id*="vimium"]',
    '.vimiumHintMarker',
    '#vimiumFlash',
    '#vimiumHUD',
    'iframe[id*="vimium"]',
    'div[style*="vimium"]',
  ]

  for (const selector of vimiumSelectors) {
    if (document.querySelector(selector)) {
      return true
    }
  }

  // Check for injected scripts or styles
  const scripts = Array.from(document.querySelectorAll('script'))
  const hasVimiumScript = scripts.some(
    script =>
      script.src.includes('vimium') ||
      script.textContent?.includes('vimium') ||
      script.id?.includes('vimium'),
  )

  const styles = Array.from(document.querySelectorAll("style, link[rel='stylesheet']"))
  const hasVimiumStyle = styles.some(
    style =>
      (style as HTMLLinkElement).href?.includes('vimium') ||
      style.textContent?.includes('vimium') ||
      style.id?.includes('vimium'),
  )

  return hasVimiumScript || hasVimiumStyle
}

// Show persistent toast warning about Vimium
async function showVimiumWarning() {
  const dismissed = localStorage.getItem(VIMIUM_DISMISSED_KEY)
  if (dismissed === 'true') return

  const isVimiumDetected = await detectVimiumExtension()
  if (!isVimiumDetected) return

  // Use the toast event system with a long duration (30 seconds)
  const event: CustomEventMap['toast'] = new CustomEvent('toast', {
    detail: {
      message: "vimium détecté. désactivez l'extension pour une navigation optimale.",
      containerId: 'vimium-toast-container',
      durationMs: 30000, // 30 seconds
    },
  })
  document.dispatchEvent(event)

  // Mark as dismissed after showing (won't show again this session)
  localStorage.setItem(VIMIUM_DISMISSED_KEY, 'true')
}

document.addEventListener('nav', () => {
  const container = document.getElementById('shortcut-container') as HTMLDivElement | null
  if (!container) return

  // Show Vimium warning on initial load
  showVimiumWarning()

  // Close shortcut modal on ESC
  const closeShortcutModal = () => {
    container.classList.remove('active')
  }

  const handleEscapeKey = (e: KeyboardEvent) => {
    if (e.key === 'Escape' && container.classList.contains('active')) {
      e.preventDefault()
      closeShortcutModal()
    }
  }

  document.addEventListener('keydown', handleEscapeKey)
  window.addCleanup(() => document.removeEventListener('keydown', handleEscapeKey))

  // Close modal when clicking outside
  const handleClickOutside = (e: MouseEvent) => {
    const shortcutSpace = container.querySelector('#shortcut-space')
    if (
      container.classList.contains('active') &&
      shortcutSpace &&
      !shortcutSpace.contains(e.target as Node)
    ) {
      closeShortcutModal()
    }
  }

  document.addEventListener('click', handleClickOutside)
  window.addCleanup(() => document.removeEventListener('click', handleClickOutside))

  // Helper to check if we should ignore the target
  const shouldIgnoreTarget = (el: EventTarget | null) => {
    if (!el || !(el instanceof Element)) return false
    const tag = el.tagName.toLowerCase()

    // Check if headings modal is open
    const headingsModal = document.querySelector('.headings-modal-container') as HTMLElement
    const isHeadingsModalOpen = headingsModal && headingsModal.style.display === 'flex'

    return (
      tag === 'input' ||
      tag === 'textarea' ||
      (el as HTMLElement).isContentEditable ||
      el.closest('.search .search-container') !== null ||
      isHeadingsModalOpen
    )
  }

  const isPaletteActive = () => {
    const palette = document.getElementById('palette-container')
    return palette?.classList.contains('active') ?? false
  }

  // Character set used for link hint markers
  const HINT_CHARS = 'asdfghjklzxcvbnm'

  // Track 'g' key for 'gg' sequence
  let waitingForSecondG = false
  let ggTimeout: number | null = null

  const suppressPaletteBindings = (e: KeyboardEvent) => {
    if (!isPaletteActive()) return
    if (e.key !== 'g' && e.key !== 'h') return
    waitingForSecondG = false
    if (ggTimeout) {
      clearTimeout(ggTimeout)
      ggTimeout = null
    }
    e.stopImmediatePropagation()
  }
  document.addEventListener('keydown', suppressPaletteBindings, true)
  window.addCleanup(() => document.removeEventListener('keydown', suppressPaletteBindings, true))

  // Link hints mode state
  let hintsActive = false
  let hintElements: Array<{ element: HTMLElement; hint: string }> = []
  let hintMarkers: HTMLElement[] = []
  let typedHint = ''

  // Generate hint strings (up to 4 chars)
  function generateHints(count: number): string[] {
    const chars = HINT_CHARS
    const hints: string[] = []

    if (count <= chars.length) {
      // Single character hints
      for (let i = 0; i < count; i++) {
        hints.push(chars[i])
      }
    } else if (count <= chars.length * chars.length) {
      // Two character hints
      for (let i = 0; i < chars.length && hints.length < count; i++) {
        for (let j = 0; j < chars.length && hints.length < count; j++) {
          hints.push(chars[i] + chars[j])
        }
      }
    } else if (count <= chars.length * chars.length * chars.length) {
      // Three character hints
      for (let i = 0; i < chars.length && hints.length < count; i++) {
        for (let j = 0; j < chars.length && hints.length < count; j++) {
          for (let k = 0; k < chars.length && hints.length < count; k++) {
            hints.push(chars[i] + chars[j] + chars[k])
          }
        }
      }
    } else {
      // Four character hints
      for (let i = 0; i < chars.length && hints.length < count; i++) {
        for (let j = 0; j < chars.length && hints.length < count; j++) {
          for (let k = 0; k < chars.length && hints.length < count; k++) {
            for (let l = 0; l < chars.length && hints.length < count; l++) {
              hints.push(chars[i] + chars[j] + chars[k] + chars[l])
            }
          }
        }
      }
    }

    return hints
  }

  // Check if element is visible and clickable
  function isElementVisible(el: HTMLElement): boolean {
    // Special case: clipboard buttons can be invisible but still interactable
    const isClipboardButton = el.classList.contains('clipboard-button')
    if (isClipboardButton) {
      return true
    }

    const rect = el.getBoundingClientRect()
    const style = window.getComputedStyle(el)

    return (
      rect.width > 0 &&
      rect.height > 0 &&
      rect.top < window.innerHeight &&
      rect.bottom > 0 &&
      rect.left < window.innerWidth &&
      rect.right > 0 &&
      style.visibility !== 'hidden' &&
      style.display !== 'none' &&
      style.opacity !== '0'
    )
  }

  // Collect all clickable elements
  function collectClickableElements(): HTMLElement[] {
    // Check for various active modals and scope to them
    let rootElement: Element | null = null

    // Check arena modal
    const arenaModal = document.querySelector('#arena-modal') as HTMLElement | null
    if (
      arenaModal &&
      arenaModal.classList.contains('active') &&
      window.getComputedStyle(arenaModal).display !== 'none'
    ) {
      rootElement = arenaModal.querySelector('.arena-modal-content')
    }

    // Check reader mode
    if (!rootElement) {
      const readerView = document.querySelector('.reader') as HTMLElement | null
      if (
        readerView &&
        readerView.classList.contains('active') &&
        window.getComputedStyle(readerView).display !== 'none'
      ) {
        rootElement = readerView.querySelector('.reader-content')
      }
    }

    // Check headings modal
    if (!rootElement) {
      const headingsModal = document.querySelector(
        '.headings-modal-container',
      ) as HTMLElement | null
      if (headingsModal && window.getComputedStyle(headingsModal).display === 'flex') {
        rootElement = headingsModal.querySelector('.headings-list')
      }
    }

    // Default to quartz root
    if (!rootElement) {
      rootElement = document.querySelector('#quartz-root')
    }

    if (!rootElement) return []

    const selectors = [
      'a[href]',
      'button',
      'span[role="button"]',
      'div[role="button"]',
      '[onclick]',
      'input[type=submit]',
      'input[type=button]',
      'label',
      '.clipboard-button',
    ]

    const elements: HTMLElement[] = []
    const seen = new Set<HTMLElement>()

    selectors.forEach(selector => {
      rootElement.querySelectorAll(selector).forEach(el => {
        const htmlEl = el as HTMLElement
        if (!seen.has(htmlEl) && isElementVisible(htmlEl)) {
          elements.push(htmlEl)
          seen.add(htmlEl)
        }
      })
    })

    return elements
  }

  // Show hint markers
  function showHints() {
    const elements = collectClickableElements()
    if (elements.length === 0) return

    const hints = generateHints(elements.length)
    hintElements = elements.map((el, i) => ({ element: el, hint: hints[i] }))
    typedHint = ''

    hintElements.forEach(({ element, hint }) => {
      const rect = element.getBoundingClientRect()
      const marker = document.createElement('div')
      marker.className = 'link-hint-marker'
      marker.textContent = hint
      marker.style.top = `${rect.top}px`
      marker.style.left = `${rect.left}px`

      document.body.appendChild(marker)
      hintMarkers.push(marker)
    })

    hintsActive = true
  }

  // Clear hint markers
  function clearHints() {
    hintMarkers.forEach(marker => marker.remove())
    hintMarkers = []
    hintElements = []
    typedHint = ''
    hintsActive = false
  }

  // Handle hint typing
  function handleHintKey(key: string) {
    typedHint += key.toLowerCase()

    // Find matching hints
    const matches = hintElements.filter(({ hint }) => hint.startsWith(typedHint))

    if (matches.length === 0) {
      // No matches - reset
      clearHints()
      return
    }

    if (matches.length === 1 && matches[0].hint === typedHint) {
      // Exact match - click the element
      const { element } = matches[0]
      clearHints()
      element.click()
      return
    }

    // Update markers to show only matching hints
    hintMarkers.forEach((marker, i) => {
      const { hint } = hintElements[i]
      if (hint.startsWith(typedHint)) {
        // Highlight typed portion
        const typed = hint.slice(0, typedHint.length)
        const remaining = hint.slice(typedHint.length)
        marker.innerHTML = `<span style="opacity: 0.5">${typed}</span>${remaining}`
      } else {
        marker.style.display = 'none'
      }
    })
  }

  // Vim navigation handler
  function vimNavigationHandler(e: KeyboardEvent) {
    if (shouldIgnoreTarget(e.target)) return
    if (isPaletteActive()) return

    // Handle hint mode
    if (hintsActive) {
      if (e.key === 'Escape') {
        e.preventDefault()
        clearHints()
        return
      }

      // Check if key is a hint character
      if (
        e.key.length === 1 &&
        HINT_CHARS.includes(e.key.toLowerCase()) &&
        !e.ctrlKey &&
        !e.metaKey
      ) {
        e.preventDefault()
        handleHintKey(e.key)
        return
      }
      return
    }

    if (e.ctrlKey || e.metaKey || e.altKey) return

    switch (e.key) {
      case 'f':
        e.preventDefault()
        showHints()
        break

      case 'j':
        e.preventDefault()
        window.scrollBy({ top: SCROLL_AMOUNT_SMALL, behavior: 'smooth' })
        break

      case 'k':
        e.preventDefault()
        window.scrollBy({ top: -SCROLL_AMOUNT_SMALL, behavior: 'smooth' })
        break

      case 'G':
        e.preventDefault()
        window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' })
        break

      case 'g':
        if (waitingForSecondG) {
          // Second 'g' pressed - scroll to top
          e.preventDefault()
          window.scrollTo({ top: 0, behavior: 'smooth' })
          waitingForSecondG = false
          if (ggTimeout) clearTimeout(ggTimeout)
        } else {
          // First 'g' pressed - wait for second
          waitingForSecondG = true
          ggTimeout = window.setTimeout(() => {
            waitingForSecondG = false
          }, 1000)
        }
        break

      case 'H':
        e.preventDefault()
        window.scrollTo({ top: window.scrollY, behavior: 'auto' })
        break

      case 'M':
        e.preventDefault()
        const middleY = window.scrollY + window.innerHeight / 2 - window.innerHeight / 2
        window.scrollTo({ top: middleY, behavior: 'auto' })
        break

      case 'L':
        e.preventDefault()
        const bottomY = window.scrollY + window.innerHeight - window.innerHeight / 2
        window.scrollTo({ top: bottomY, behavior: 'auto' })
        break
    }
  }

  // Ctrl-based scroll handler
  function ctrlScrollHandler(e: KeyboardEvent) {
    if (shouldIgnoreTarget(e.target)) return
    if (isPaletteActive()) return
    if (!e.ctrlKey || e.metaKey) return

    const halfPage = window.innerHeight / 2

    switch (e.key) {
      case 'd':
        e.preventDefault()
        window.scrollBy({ top: halfPage, behavior: 'instant' })
        break

      case 'u':
        e.preventDefault()
        window.scrollBy({ top: -halfPage, behavior: 'instant' })
        break

      case 'e':
        e.preventDefault()
        window.scrollBy({ top: SCROLL_AMOUNT_LARGE, behavior: 'instant' })
        break

      case 'y':
        e.preventDefault()
        window.scrollBy({ top: -SCROLL_AMOUNT_LARGE, behavior: 'instant' })
        break
    }
  }

  document.addEventListener('keydown', vimNavigationHandler)
  document.addEventListener('keydown', ctrlScrollHandler)

  // Clear hints on scroll or resize
  const clearHintsOnScroll = () => {
    if (hintsActive) clearHints()
  }
  window.addEventListener('scroll', clearHintsOnScroll, { passive: true })
  window.addEventListener('resize', clearHintsOnScroll)

  window.addCleanup(() => {
    document.removeEventListener('keydown', vimNavigationHandler)
    document.removeEventListener('keydown', ctrlScrollHandler)
    window.removeEventListener('scroll', clearHintsOnScroll)
    window.removeEventListener('resize', clearHintsOnScroll)
    clearHints()
    if (ggTimeout) clearTimeout(ggTimeout)
  })

  const shortcuts = container.querySelectorAll(
    'ul[id="shortcut-list"] > li > div[id="shortcuts"]',
  ) as NodeListOf<HTMLElement>
  for (const short of shortcuts) {
    const binding = short.dataset.key as string
    const span = short.dataset.value as string

    // Check if binding has a prefix (cmd, ctrl, etc.)
    const hasPrefix = binding.includes('--')

    if (hasPrefix) {
      let [prefix, key] = binding.split('--')
      const spanAliases = aliases[span]
      prefix = isMacOS() ? '⌘' : '⌃'
      const browser = detectBrowser()
      if (spanAliases) {
        const { mac, def } = spanAliases
        key = browser === 'Safari' ? mac : def
      }
      short.innerHTML = `
<kbd class="clickable">${prefix} ${key}</kbd>
<span>${span}</span>
`
    } else {
      // Plain keybinding without modifier
      short.innerHTML = `
<kbd class="clickable">${binding}</kbd>
<span>${span}</span>
`
    }
  }

  function shortcutHandler(e: HTMLElementEventMap['keydown']) {
    if (_mapping.get(e.key) !== undefined && (e.ctrlKey || e.metaKey)) {
      e.preventDefault()
      const action = _mapping.get(e.key)
      if (typeof action === 'function') {
        action()
      } else if (typeof action === 'string') {
        container?.classList.toggle('active', false)
        if (window.location.pathname === action) return
        window.spaNavigate(new URL(action, window.location.toString()))
      }
    }
  }

  document.addEventListener('keydown', shortcutHandler)
  window.addCleanup(() => document.removeEventListener('keydown', shortcutHandler))
})
