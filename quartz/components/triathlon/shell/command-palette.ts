import type { TriathlonContext } from '../runtime/context'
import { el } from '../runtime/dom'
import { nextTriMapStyle } from '../runtime/preferences'
import { toggleTriMapStyle } from '../runtime/preferences'
import { toggleTriPanelsFullscreen } from '../runtime/preferences'
import { toggleTriPowerFilter } from '../runtime/preferences'
import { toggleTriUnit } from '../runtime/preferences'

export const TRI_PAGES: { path: string; label: string; hint: string }[] = [
  { path: '/triathlon', label: 'triathlon', hint: 'overview' },
  { path: '/triathlon/tools', label: 'tools', hint: 'gears' },
  { path: '/triathlon/calc', label: 'calculator', hint: 'race calc' },
  { path: '/triathlon/analytics', label: 'analytics', hint: 'charts' },
  { path: '/triathlon/maps', label: 'maps', hint: 'routes' },
  { path: '/triathlon/training', label: 'training', hint: 'plans' },
  { path: '/triathlon/feed', label: 'feed', hint: 'all activities' },
  { path: '/triathlon/on', label: 'on', hint: 'by date' },
]

export type SearchShortcut = { view: string; openClass?: string; search: string }

export const TRI_SEARCH_SHORTCUTS: SearchShortcut[] = [
  { view: 'analytics', openClass: 'tri-analytics-open', search: '.tri-analytics .tri-ana-search' },
  { view: 'maps', openClass: 'tri-map-open', search: '.tri-map .tri-map-search' },
  {
    view: 'training',
    openClass: 'tri-training-open',
    search: '.tri-training .tri-training-search',
  },
  { view: 'feed', search: '.tri-feed .tri-feed-search' },
]

export const isEditable = (el: HTMLElement): boolean => {
  const tag = el.tagName.toLowerCase()
  return (
    tag === 'input' ||
    tag === 'textarea' ||
    tag === 'select' ||
    el.isContentEditable ||
    el.closest('.search-container') !== null
  )
}

export const currentSearchShortcut = (root: HTMLElement): SearchShortcut | undefined => {
  const subView = root.dataset.triView
  if (subView) return TRI_SEARCH_SHORTCUTS.find(shortcut => shortcut.view === subView)
  return TRI_SEARCH_SHORTCUTS.find(
    shortcut => shortcut.openClass && root.classList.contains(shortcut.openClass),
  )
}

export const toggleSearchFocus = (root: HTMLElement, target: HTMLElement | null): boolean => {
  const shortcut = currentSearchShortcut(root)
  if (!shortcut) return false
  const search = root.querySelector<HTMLInputElement>(shortcut.search)
  if (!search) return false
  if (target && isEditable(target) && target !== search) return false
  if (document.activeElement === search) search.blur()
  else {
    search.focus()
    search.select()
  }
  return true
}

export const blurFocusedPanelSearch = (root: HTMLElement): boolean => {
  const shortcut = currentSearchShortcut(root)
  if (!shortcut?.openClass) return false
  const search = root.querySelector<HTMLInputElement>(shortcut.search)
  if (!search || document.activeElement !== search) return false
  search.blur()
  return true
}

export const mapDetailMetricTabForKey = (
  root: HTMLElement,
  key: string,
): HTMLButtonElement | null => {
  if (key.length !== 1) return null
  const tablist = root.querySelector<HTMLElement>('.tri-map--detail .tri-map-tablist')
  if (!tablist) return null
  const shortcut = key.toLowerCase()
  return (
    Array.from(tablist.querySelectorAll<HTMLButtonElement>('.tri-map-tab')).find(
      tab => tab.dataset.shortcut === shortcut,
    ) ?? null
  )
}

export const setupCommandPalette = (root: HTMLElement, context: TriathlonContext): (() => void) => {
  const trigger = root.querySelector<HTMLButtonElement>('.tri-cmdk-trigger')
  const overlay = el('div', 'tri-cmdk', undefined, {
    id: 'tri-command-palette',
    'aria-hidden': 'true',
  })
  const box = el('div', 'tri-cmdk-box', undefined, {
    role: 'dialog',
    'aria-label': 'command palette',
  })
  const input = el('input', 'tri-cmdk-input', undefined, {
    type: 'text',
    placeholder: 'go to page · toggle units...',
    'aria-label': 'command',
    autocomplete: 'off',
    spellcheck: 'false',
  }) as HTMLInputElement
  const list = el('div', 'tri-cmdk-list', undefined, { role: 'listbox' })
  box.append(input, list)
  overlay.appendChild(box)
  root.appendChild(overlay)

  interface Cmd {
    label: () => string
    hint: string
    keys: string
    run: () => void
  }
  const navTo = (path: string) => (): void => {
    close()
    const url = new URL(path, window.location.toString())
    if (window.spaNavigate) window.spaNavigate(url)
    else window.location.href = url.toString()
  }
  const cmds: Cmd[] = [
    ...TRI_PAGES.map(p => ({
      label: () => `${p.label}`,
      hint: p.hint,
      keys: `go ${p.label} ${p.path}`,
      run: navTo(p.path),
    })),
    {
      label: () =>
        context.presentation.distance === 'imperial' ? 'imperial → metric' : 'metric → imperial',
      hint: 'units',
      keys: 'toggle units km mi miles kg lb imperial metric pace distance speed weight',
      run: () => {
        toggleTriUnit(context.preferences)
        render()
      },
    },
    {
      label: () =>
        context.formatter.text(
          context.presentation.powerSamples === 'exclude-zero'
            ? 'power averages · zeros excluded'
            : 'power averages · zeros included',
        ),
      hint: 'power',
      keys: 'power watts zero zeros include exclude coasting freewheel downhill traffic stop',
      run: () => {
        toggleTriPowerFilter(context.preferences)
        render()
      },
    },
    {
      label: () =>
        context.presentation.locale === 'fr' ? 'langue · english' : 'language · français',
      hint: 'locale',
      keys: 'language langue locale english french francais français en fr i18n',
      run: () => {
        context.preferences.update({ locale: context.presentation.locale === 'fr' ? 'en' : 'fr' })
        render()
      },
    },
    {
      label: () => {
        const next = nextTriMapStyle()
        return `map style · ${next === 'mono' ? 'monochrome' : next}`
      },
      hint: 'map',
      keys: 'map style roads streets monochrome mono satellite imagery mapbox route road',
      run: () => {
        toggleTriMapStyle()
        render()
      },
    },
    {
      label: () =>
        context.formatter.text(
          root.classList.contains('tri-panels-fullscreen')
            ? 'panels · windowed'
            : 'panels · full screen',
        ),
      hint: 'layout',
      keys: 'toggle panels fullscreen full screen windowed desktop mobile analytics map training layout',
      run: () => {
        toggleTriPanelsFullscreen(root)
        render()
      },
    },
  ]

  let items: Cmd[] = cmds
  let sel = 0
  let isOpen = false

  const paint = (): void => {
    const rows = list.querySelectorAll<HTMLElement>('.tri-cmdk-row')
    rows.forEach((r, i) => {
      r.classList.toggle('tri-cmdk-row--on', i === sel)
      r.setAttribute('aria-selected', String(i === sel))
    })
    rows[sel]?.scrollIntoView({ block: 'nearest' })
  }
  const render = (): void => {
    const q = input.value.trim().toLowerCase()
    items = q
      ? cmds.filter(c => `${c.label()} ${c.hint} ${c.keys}`.toLowerCase().includes(q))
      : cmds
    if (sel >= items.length) sel = Math.max(0, items.length - 1)
    list.replaceChildren(
      ...items.map((c, i) => {
        const row = el(
          'div',
          i === sel ? 'tri-cmdk-row tri-cmdk-row--on' : 'tri-cmdk-row',
          undefined,
          { role: 'option', 'aria-selected': String(i === sel) },
        )
        row.append(
          el('span', 'tri-cmdk-row-label', c.label()),
          el('span', 'tri-cmdk-row-hint', c.hint),
        )
        row.addEventListener('mousemove', () => {
          if (sel !== i) {
            sel = i
            paint()
          }
        })
        row.addEventListener('click', () => c.run())
        return row
      }),
    )
    if (!items.length)
      list.appendChild(el('div', 'tri-cmdk-empty', context.formatter.text('no commands')))
  }
  const openPalette = (): void => {
    if (isOpen) return
    isOpen = true
    input.value = ''
    sel = 0
    render()
    overlay.classList.add('tri-cmdk--on')
    overlay.setAttribute('aria-hidden', 'false')
    trigger?.setAttribute('aria-expanded', 'true')
    input.focus()
  }
  function close(): void {
    if (!isOpen) return
    isOpen = false
    overlay.classList.remove('tri-cmdk--on')
    overlay.setAttribute('aria-hidden', 'true')
    trigger?.setAttribute('aria-expanded', 'false')
    input.blur()
  }

  const togglePalette = (): void => {
    if (isOpen) close()
    else openPalette()
  }

  const onInput = (): void => {
    sel = 0
    render()
  }
  const onInputKey = (e: KeyboardEvent): void => {
    if (e.key === 'Escape') {
      e.preventDefault()
      close()
    } else if (e.key === 'Enter') {
      e.preventDefault()
      items[sel]?.run()
    } else if (e.key === 'ArrowDown' || (e.ctrlKey && e.key.toLowerCase() === 'n')) {
      e.preventDefault()
      if (items.length) sel = (sel + 1) % items.length
      paint()
    } else if (e.key === 'ArrowUp' || (e.ctrlKey && e.key.toLowerCase() === 'p')) {
      e.preventDefault()
      if (items.length) sel = (sel - 1 + items.length) % items.length
      paint()
    }
  }
  const onDocKey = (e: KeyboardEvent): void => {
    if ((e.ctrlKey || e.metaKey) && !e.altKey && !e.shiftKey && e.key.toLowerCase() === 'k') {
      e.preventDefault()
      e.stopImmediatePropagation()
      if (toggleSearchFocus(root, null) || currentSearchShortcut(root)) return
      if (root.matches('.tri-analytics-open, .tri-map-open, .tri-training-open, .tri-calc-open'))
        return
      togglePalette()
    }
  }
  const onScrim = (e: MouseEvent): void => {
    if (e.target === overlay) close()
  }
  input.addEventListener('input', onInput)
  input.addEventListener('keydown', onInputKey)
  overlay.addEventListener('mousedown', onScrim)
  trigger?.addEventListener('click', togglePalette)
  document.addEventListener('keydown', onDocKey, true)
  return () => {
    document.removeEventListener('keydown', onDocKey, true)
    trigger?.removeEventListener('click', togglePalette)
    overlay.remove()
  }
}
