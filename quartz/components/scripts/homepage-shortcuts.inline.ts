import { homepageShortcutKeys } from './homepage-shortcuts'
import { currentNavSignal } from './nav-lifecycle'

type Shortcut = { link: HTMLAnchorElement; key: string; hint: HTMLElement }
let mountedSignal: AbortSignal | undefined

document.addEventListener('nav', () => {
  if (document.body.dataset.slug !== 'index') return
  const board = document.querySelector<HTMLElement>('.landing .homepage-board')
  if (!board) return
  const signal = currentNavSignal()
  if (mountedSignal === signal) return
  mountedSignal = signal

  let shortcuts: Shortcut[] = []
  let prefix = ''
  let timer: number | undefined
  let shiftHeld = false
  let hintMode = false
  let enabled = localStorage.getItem('homepage-link-keys') !== 'off'

  const help = document.createElement('div')
  help.className = 'homepage-shortcut-help'
  help.lang = 'en'
  const toggle = document.createElement('button')
  toggle.type = 'button'
  const instruction = document.createElement('span')
  instruction.textContent = 'press f or hold ⇧ for home link keys'
  help.append(toggle, instruction)
  document.getElementById('shortcut-space')?.append(help)

  const blocked = () =>
    !enabled ||
    !board.isConnected ||
    document.body.dataset.slug !== 'index' ||
    document.body.dataset.trailOpen === 'true' ||
    document.hidden ||
    !board.checkVisibility({ visibilityProperty: true }) ||
    !!document.querySelector(
      'dialog[open], [popover]:popover-open, #palette-container.active, .search-container.active, #shortcut-container.active, .link-hint-marker',
    )

  const editing = (target: EventTarget | null) =>
    target instanceof Element &&
    (target.closest(
      'input, textarea, select, [role="textbox"], [role="combobox"], [role="listbox"], .cm-editor',
    ) !== null ||
      (target instanceof HTMLElement && target.isContentEditable))

  const paint = () => {
    board.toggleAttribute(
      'data-home-keys-visible',
      enabled && (hintMode || shiftHeld || prefix !== ''),
    )
    for (const shortcut of shortcuts) {
      shortcut.hint.hidden = prefix !== '' && !shortcut.key.startsWith(prefix)
      shortcut.link.toggleAttribute('data-home-key-match', prefix !== '' && shortcut.key === prefix)
    }
  }

  const reset = () => {
    window.clearTimeout(timer)
    timer = undefined
    prefix = ''
    shiftHeld = false
    hintMode = false
    paint()
  }

  const scan = () => {
    for (const { link, hint } of shortcuts) {
      hint.remove()
      delete link.dataset.homeShortcut
      link.removeAttribute('data-home-key-match')
    }
    // querySelectorAll preserves pre-order DFS, including nested emphasis and footer links.
    const links = Array.from(board.querySelectorAll<HTMLAnchorElement>('a[href]')).filter(
      link =>
        link.checkVisibility({ visibilityProperty: true }) &&
        !link.closest('[inert], [aria-hidden="true"]'),
    )
    const keys = homepageShortcutKeys(
      links.map(link => {
        const label = link.cloneNode(true)
        if (!(label instanceof Element)) return ''
        label.querySelectorAll('svg, [aria-hidden="true"]').forEach(icon => icon.remove())
        return label.textContent?.trim() || link.getAttribute('aria-label') || ''
      }),
    )
    shortcuts = links.flatMap((link, index) => {
      const key = keys[index]
      if (!key) return []
      const hint = document.createElement('kbd')
      hint.className = 'homepage-link-key'
      hint.textContent = key
      hint.setAttribute('aria-hidden', 'true')
      link.dataset.homeShortcut = key
      link.append(hint)
      return [{ link, key, hint }]
    })
    paint()
  }

  const follow = (shortcut: Shortcut | undefined) => {
    const canFollow =
      shortcut && !blocked() && !editing(document.activeElement) && shortcut.link.isConnected
    reset()
    if (canFollow) shortcut.link.click()
  }

  const updateToggle = () => {
    toggle.textContent = `link keys ${enabled ? 'on' : 'off'}`
    toggle.setAttribute('aria-pressed', String(enabled))
    toggle.title = 'Turn homepage link keys on or off'
    instruction.hidden = !enabled
  }
  toggle.addEventListener(
    'click',
    () => {
      enabled = !enabled
      localStorage.setItem('homepage-link-keys', enabled ? 'on' : 'off')
      reset()
      updateToggle()
    },
    { signal },
  )

  const keydown = (event: KeyboardEvent) => {
    if (event.defaultPrevented || event.isComposing || editing(event.target) || blocked()) {
      reset()
      return
    }
    if (event.ctrlKey || event.metaKey || event.altKey) {
      reset()
      return
    }
    if (event.key === 'f' && !hintMode) {
      event.preventDefault()
      event.stopImmediatePropagation()
      if (event.repeat) return
      reset()
      scan()
      hintMode = true
      paint()
      return
    }
    if (event.key === 'Shift') {
      if (!prefix && !hintMode) scan()
      shiftHeld = true
      paint()
      return
    }
    if (event.key === 'Escape') {
      if (prefix || hintMode) {
        event.preventDefault()
        event.stopImmediatePropagation()
      }
      reset()
      return
    }
    if (event.key === 'Enter' && prefix) {
      event.preventDefault()
      event.stopImmediatePropagation()
      follow(shortcuts.find(shortcut => shortcut.key === prefix))
      return
    }
    const key = hintMode ? event.key.toUpperCase() : event.key
    if (!/^[A-Z]$/.test(key)) {
      reset()
      return
    }
    if (hintMode) {
      event.preventDefault()
      event.stopImmediatePropagation()
    }
    if (!prefix && !hintMode) scan()
    const candidate = prefix + key
    const matches = shortcuts.filter(shortcut => shortcut.key.startsWith(candidate))
    if (matches.length === 0) {
      reset()
      return
    }
    event.preventDefault()
    event.stopImmediatePropagation()
    if (event.repeat) return
    window.clearTimeout(timer)
    prefix = candidate
    shiftHeld = event.shiftKey
    paint()
    const exact = matches.find(shortcut => shortcut.key === prefix)
    if (exact && matches.length === 1) follow(exact)
    else timer = window.setTimeout(() => follow(exact), 700)
  }

  document.addEventListener('keydown', keydown, { capture: true, signal })
  document.addEventListener(
    'keyup',
    event => {
      if (event.key !== 'Shift') return
      shiftHeld = false
      paint()
    },
    { signal },
  )
  document.addEventListener('pointerdown', reset, { signal })
  document.addEventListener('focusin', reset, { signal })
  document.addEventListener('visibilitychange', reset, { signal })
  window.addEventListener('blur', reset, { signal })
  window.addEventListener('scroll', reset, { passive: true, signal })
  window.addEventListener('resize', reset, { signal })
  window.addCleanup(() => {
    reset()
    for (const { link, hint } of shortcuts) {
      hint.remove()
      delete link.dataset.homeShortcut
      link.removeAttribute('data-home-key-match')
    }
    help.remove()
    board.removeAttribute('data-home-keys-visible')
    if (mountedSignal === signal) mountedSignal = undefined
  })
  scan()
  updateToggle()
})
