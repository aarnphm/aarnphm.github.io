export interface SelectOption<T> {
  value: T
  label: string
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

  const items: { button: HTMLButtonElement; option: SelectOption<T> }[] = []
  const addOption = (parent: HTMLElement, option: SelectOption<T>): void => {
    const button = document.createElement('button')
    button.type = 'button'
    button.className = 'g-select-option'
    button.setAttribute('role', 'option')
    button.append(span('g-select-check', '✓'), span('g-select-label', option.label))
    button.firstElementChild?.setAttribute('aria-hidden', 'true')
    items.push({ button, option })
    parent.append(button)
  }
  for (const group of options.groups) {
    if (!group.label) {
      for (const option of group.options) addOption(menu, option)
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
    menu.append(node)
  }
  root.append(trigger, menu)

  const setSelected = (selected: T): void => {
    const chosen = items.find(item => item.option.value === selected) ?? items[0]
    if (!chosen) return
    value.textContent = chosen.option.label
    trigger.setAttribute('aria-label', `${options.label}: ${chosen.option.label}`)
    for (const item of items) {
      item.button.setAttribute('aria-selected', String(item === chosen))
      item.button.tabIndex = item === chosen ? 0 : -1
    }
  }
  setSelected(options.selected)

  const close = (restoreFocus = false): void => {
    menu.hidden = true
    trigger.setAttribute('aria-expanded', 'false')
    if (restoreFocus) trigger.focus({ preventScroll: true })
  }
  const focusItem = (index: number): void => {
    const target = items[index]
    if (!target) return
    for (const item of items) item.button.tabIndex = item === target ? 0 : -1
    target.button.focus({ preventScroll: true })
    target.button.scrollIntoView({ block: 'nearest' })
  }
  const open = (): void => {
    menu.hidden = false
    trigger.setAttribute('aria-expanded', 'true')
    focusItem(items.findIndex(item => item.button.getAttribute('aria-selected') === 'true'))
  }

  const onClick = (event: MouseEvent): void => {
    const target = event.target
    if (!(target instanceof Node)) return
    if (trigger.contains(target)) {
      if (menu.hidden) open()
      else close()
      return
    }
    const chosen = items.find(item => item.button.contains(target))
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
    if (menu.hidden) return
    const index = items.findIndex(item => item.button === document.activeElement)
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
    const match = items
      .map((_, offset) => (index + offset + 1) % items.length)
      .find(candidate => items[candidate].option.label.toLocaleLowerCase().startsWith(search))
    if (match === undefined) return
    event.preventDefault()
    focusItem(match)
  }
  const onFocusout = (event: FocusEvent): void => {
    if (!(event.relatedTarget instanceof Node) || !root.contains(event.relatedTarget)) close()
  }
  const onPointerdown = (event: PointerEvent): void => {
    if (!menu.hidden && !event.composedPath().includes(root)) close()
  }

  return {
    element: root,
    trigger,
    setSelected,
    mount: () => {
      root.addEventListener('click', onClick)
      root.addEventListener('keydown', onKeydown)
      root.addEventListener('focusout', onFocusout)
      document.addEventListener('pointerdown', onPointerdown)
      return () => {
        root.removeEventListener('click', onClick)
        root.removeEventListener('keydown', onKeydown)
        root.removeEventListener('focusout', onFocusout)
        document.removeEventListener('pointerdown', onPointerdown)
      }
    },
  }
}
