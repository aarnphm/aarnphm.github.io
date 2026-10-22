import { autoUpdate, computePosition, flip, offset, shift, size } from '@floating-ui/dom'

export interface SelectOption<T> {
  value: T
  label: string
  /** A second line under the label in the menu; the trigger shows only the label. */
  detail?: string
}

export interface SelectGroup<T> {
  /** Shown as a heading above the group; an unlabeled group renders its options in place. */
  label?: string
  options: SelectOption<T>[]
}

export interface SelectOptions<T> {
  id: string
  /** Names the listbox; the trigger reads as "label: chosen option". */
  label: string
  groups: SelectGroup<T>[]
  selected: T
  onSelect: (value: T) => void
  /** Extra classes on the root, for a host's layout rules. */
  className?: string
  /** Use the top layer when an ancestor clips overflowing menus. */
  floating?: boolean
}

export interface Select<T> {
  element: HTMLElement
  trigger: HTMLButtonElement
  /** Marks `value` as chosen without calling `onSelect`. */
  setSelected: (value: T) => void
  mount: () => () => void
}

const SVGNS = 'http://www.w3.org/2000/svg'

const span = (className: string, text?: string): HTMLSpanElement => {
  const node = document.createElement('span')
  node.className = className
  if (text !== undefined) node.textContent = text
  return node
}

const chevron = (): SVGElement => {
  const svg = document.createElementNS(SVGNS, 'svg')
  svg.setAttribute('class', 'g-select-chevron')
  svg.setAttribute('viewBox', '0 0 16 16')
  svg.setAttribute('fill', 'none')
  svg.setAttribute('aria-hidden', 'true')
  svg.setAttribute('focusable', 'false')
  const path = document.createElementNS(SVGNS, 'path')
  path.setAttribute('d', 'm4 6 4 4 4-4')
  path.setAttribute('stroke', 'currentColor')
  path.setAttribute('stroke-width', '1.4')
  path.setAttribute('stroke-linecap', 'round')
  path.setAttribute('stroke-linejoin', 'round')
  svg.append(path)
  return svg
}

const TYPEAHEAD_RESET_MS = 700

/** A single-choice listbox behind a trigger button. */
export const buildSelect = <T>(options: SelectOptions<T>): Select<T> => {
  const root = document.createElement('div')
  root.className = options.className ? `g-select ${options.className}` : 'g-select'
  const trigger = document.createElement('button')
  trigger.type = 'button'
  trigger.className = 'g-select-trigger'
  trigger.setAttribute('aria-haspopup', 'listbox')
  trigger.setAttribute('aria-expanded', 'false')
  trigger.setAttribute('aria-controls', options.id)
  const value = span('g-select-value')
  trigger.append(value, chevron())
  const menu = document.createElement('div')
  menu.className = 'g-select-menu'
  menu.id = options.id
  menu.setAttribute('role', 'listbox')
  menu.setAttribute('aria-label', options.label)
  menu.hidden = true
  if (options.floating) menu.setAttribute('popover', 'manual')

  const flatOptions: SelectOption<T>[] =
    options.groups.length === 1
      ? options.groups[0].options
      : options.groups.flatMap(group => group.options)
  const items: { button: HTMLButtonElement; option: SelectOption<T>; lowerLabel: string }[] = []
  let selectedIndex = -1
  let focusedIndex = -1
  let menuMaterialized = false

  const ensureMenuMaterialized = (): void => {
    if (menuMaterialized) return
    menuMaterialized = true
    const fragment = document.createDocumentFragment()
    const addOption = (parent: ParentNode, option: SelectOption<T>): void => {
      const idx = items.length
      const isSelected = idx === selectedIndex
      const button = document.createElement('button')
      button.type = 'button'
      button.className = 'g-select-option'
      button.dataset.selectIndex = String(idx)
      button.setAttribute('role', 'option')
      button.setAttribute('aria-selected', isSelected ? 'true' : 'false')
      button.tabIndex = isSelected ? 0 : -1
      const check = span('g-select-check', '✓')
      check.setAttribute('aria-hidden', 'true')
      const label = span('g-select-label', option.label)
      if (option.detail) {
        const text = span('g-select-text')
        text.append(label, span('g-select-detail', option.detail))
        button.append(check, text)
      } else button.append(check, label)
      items.push({ button, option, lowerLabel: option.label.toLocaleLowerCase() })
      parent.append(button)
    }
    for (const group of options.groups) {
      if (!group.label) {
        for (const option of group.options) addOption(fragment, option)
        continue
      }
      const node = document.createElement('div')
      node.className = 'g-select-group'
      node.setAttribute('role', 'group')
      node.setAttribute('aria-label', group.label)
      const heading = span('g-select-heading', group.label)
      heading.setAttribute('aria-hidden', 'true')
      node.append(heading)
      for (const option of group.options) addOption(node, option)
      fragment.append(node)
    }
    focusedIndex = selectedIndex
    menu.append(fragment)
  }

  root.append(trigger, menu)

  const setSelected = (selected: T): void => {
    if (flatOptions.length === 0) return
    let nextIndex = flatOptions.findIndex(option => option.value === selected)
    if (nextIndex === -1) nextIndex = 0
    const chosen = flatOptions[nextIndex]
    const prevSelected = selectedIndex
    selectedIndex = nextIndex
    value.textContent = chosen.label
    const currentLabel = menu.getAttribute('aria-label') || options.label
    trigger.setAttribute('aria-label', `${currentLabel}: ${chosen.label}`)
    if (items.length > 0) {
      if (prevSelected >= 0 && prevSelected !== nextIndex && items[prevSelected]) {
        items[prevSelected].button.setAttribute('aria-selected', 'false')
        items[prevSelected].button.tabIndex = -1
      }
      if (focusedIndex >= 0 && focusedIndex !== nextIndex && items[focusedIndex]) {
        items[focusedIndex].button.tabIndex = -1
      }
      const nextItem = items[nextIndex]
      if (nextItem) {
        nextItem.button.setAttribute('aria-selected', 'true')
        nextItem.button.tabIndex = 0
      }
      focusedIndex = nextIndex
    }
  }
  setSelected(options.selected)

  let stopPositioning: (() => void) | undefined
  const close = (restoreFocus = false): void => {
    stopPositioning?.()
    stopPositioning = undefined
    if (menu.matches(':popover-open')) menu.hidePopover()
    menu.hidden = true
    trigger.setAttribute('aria-expanded', 'false')
    if (restoreFocus) trigger.focus({ preventScroll: true })
  }
  const focusItem = (index: number, scroll = true): void => {
    const target = items[index]
    if (!target) return
    if (focusedIndex !== index) {
      if (focusedIndex >= 0 && items[focusedIndex]) {
        items[focusedIndex].button.tabIndex = -1
      }
      target.button.tabIndex = 0
      focusedIndex = index
    }
    target.button.focus({ preventScroll: true })
    if (scroll) target.button.scrollIntoView({ block: 'nearest' })
  }
  const open = (): void => {
    ensureMenuMaterialized()
    menu.hidden = false
    trigger.setAttribute('aria-expanded', 'true')
    if (options.floating && !menu.matches(':popover-open')) {
      menu.showPopover()
      stopPositioning = autoUpdate(trigger, menu, async () => {
        const { x, y } = await computePosition(trigger, menu, {
          strategy: 'fixed',
          placement: 'bottom-start',
          middleware: [
            offset(3),
            flip({ padding: 8 }),
            shift({ padding: 8 }),
            size({
              padding: 8,
              apply: ({ availableHeight, rects }) => {
                menu.style.width = `${rects.reference.width}px`
                menu.style.maxHeight = `min(22rem, ${Math.max(0, availableHeight)}px)`
              },
            }),
          ],
        })
        if (menu.hidden) return
        menu.style.left = `${x}px`
        menu.style.top = `${y}px`
      })
    }
    const initialIndex = selectedIndex >= 0 ? selectedIndex : 0
    focusItem(initialIndex, false)
    if (initialIndex > 0) {
      requestAnimationFrame(() => {
        if (!menu.hidden) items[initialIndex]?.button.scrollIntoView({ block: 'nearest' })
      })
    }
  }

  // Listeners read the root's document and skip realm-bound instanceof checks: the homepage trail
  // builds this control inside its iframe and mounts it in the parent page.
  const onClick = (event: MouseEvent): void => {
    const target = event.target as Node | null
    if (!target) return
    if (trigger.contains(target)) {
      if (menu.hidden) open()
      else close()
      return
    }
    const optionButton = (target as Element).closest?.(
      '.g-select-option',
    ) as HTMLButtonElement | null
    if (!optionButton || !menu.contains(optionButton)) return
    const idx = Number(optionButton.dataset.selectIndex)
    const chosen = items[idx]
    if (!chosen) return
    setSelected(chosen.option.value)
    options.onSelect(chosen.option.value)
    close(true)
  }
  let search = ''
  let lastSearch = 0
  const onKeydown = (event: KeyboardEvent): void => {
    if (event.key === 'Escape' && !menu.hidden) {
      event.preventDefault()
      event.stopPropagation()
      close(true)
      return
    }
    if (event.target === trigger && (event.key === 'ArrowDown' || event.key === 'ArrowUp')) {
      event.preventDefault()
      open()
      return
    }
    if (menu.hidden || items.length === 0) return
    const targetIndex = Number((event.target as HTMLElement | null)?.dataset?.selectIndex)
    const index =
      Number.isInteger(targetIndex) && targetIndex >= 0 ? targetIndex : Math.max(0, focusedIndex)
    const next =
      event.key === 'Home'
        ? 0
        : event.key === 'End'
          ? items.length - 1
          : event.key === 'ArrowDown'
            ? Math.min(items.length - 1, index + 1)
            : event.key === 'ArrowUp'
              ? Math.max(0, index - 1)
              : -1
    if (next >= 0) {
      event.preventDefault()
      focusItem(next)
      return
    }
    if (
      event.key.length !== 1 ||
      event.key === ' ' ||
      event.ctrlKey ||
      event.metaKey ||
      event.altKey
    )
      return
    // Typeahead: letters typed within a short pause extend the prefix; the search wraps.
    const now = performance.now()
    search = `${now - lastSearch < TYPEAHEAD_RESET_MS ? search : ''}${event.key.toLocaleLowerCase()}`
    lastSearch = now
    const count = items.length
    for (let offset = 1; offset <= count; offset++) {
      const candidate = (index + offset) % count
      if (items[candidate].lowerLabel.startsWith(search)) {
        event.preventDefault()
        focusItem(candidate)
        return
      }
    }
  }
  const onFocusout = (event: FocusEvent): void => {
    const next = event.relatedTarget as Node | null
    if (!next || !root.contains(next)) close()
  }
  const onPointerdown = (event: PointerEvent): void => {
    if (!menu.hidden && !event.composedPath().includes(root)) close()
  }

  return {
    element: root,
    trigger,
    setSelected,
    mount: () => {
      const host = root.ownerDocument
      root.addEventListener('click', onClick)
      root.addEventListener('keydown', onKeydown)
      root.addEventListener('focusout', onFocusout)
      host.addEventListener('pointerdown', onPointerdown)
      return () => {
        close()
        root.removeEventListener('click', onClick)
        root.removeEventListener('keydown', onKeydown)
        root.removeEventListener('focusout', onFocusout)
        host.removeEventListener('pointerdown', onPointerdown)
      }
    },
  }
}
