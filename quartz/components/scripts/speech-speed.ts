import { autoUpdate, computePosition, flip, offset, shift } from '@floating-ui/dom'

export type SpeechSpeed = '1' | '0.8'

type SpeechPicker = { element: HTMLElement; close: (restoreFocus?: boolean) => void }

export function createSpeechSpeed(
  initial: SpeechSpeed,
  select: (value: SpeechSpeed) => void,
  signal: AbortSignal,
): SpeechPicker {
  return createSpeechPicker({
    id: 'speech-speed-options',
    label: 'Vitesse',
    menuLabel: 'Vitesse de lecture',
    choices: [
      ['1', '1×'],
      ['0.8', '0,8×'],
    ],
    initial,
    select,
    signal,
  })
}

export function createSpeechPicker<T extends string>({
  id,
  label: labelText,
  menuLabel,
  choices,
  initial,
  select,
  signal,
}: {
  id: string
  label: string
  menuLabel: string
  choices: [T, string][]
  initial: T
  select: (value: T) => void
  signal: AbortSignal
}): SpeechPicker {
  const picker = document.createElement('div')
  picker.className = 'speech-speed'
  const label = document.createElement('span')
  label.textContent = labelText
  const trigger = document.createElement('button')
  trigger.type = 'button'
  trigger.className = 'speech-speed-trigger'
  trigger.setAttribute('aria-haspopup', 'listbox')
  trigger.setAttribute('aria-expanded', 'false')
  trigger.setAttribute('aria-controls', id)
  const value = document.createElement('span')
  const chevron = document.createElementNS('http://www.w3.org/2000/svg', 'svg')
  chevron.setAttribute('viewBox', '0 0 16 16')
  chevron.setAttribute('aria-hidden', 'true')
  const path = document.createElementNS('http://www.w3.org/2000/svg', 'path')
  path.setAttribute('d', 'm4 6 4 4 4-4')
  chevron.append(path)
  trigger.append(value, chevron)
  const menu = document.createElement('div')
  menu.id = id
  menu.className = 'speech-speed-menu'
  menu.setAttribute('role', 'listbox')
  menu.setAttribute('aria-label', menuLabel)
  menu.popover = 'manual'
  const options: { button: HTMLButtonElement; value: T; label: string }[] = []
  let stopPositioning: (() => void) | undefined
  const close = (restoreFocus = false) => {
    stopPositioning?.()
    stopPositioning = undefined
    menu.hidePopover()
    trigger.setAttribute('aria-expanded', 'false')
    if (restoreFocus) trigger.focus({ preventScroll: true })
  }
  const update = (selected: T) => {
    for (const option of options) {
      const active = option.value === selected
      option.button.setAttribute('aria-selected', String(active))
      option.button.tabIndex = active ? 0 : -1
      if (active) {
        value.textContent = option.label
        trigger.setAttribute('aria-label', `${menuLabel} : ${option.label}`)
      }
    }
  }
  for (const [choice, text] of choices) {
    const button = document.createElement('button')
    button.type = 'button'
    button.className = 'speech-speed-option'
    button.setAttribute('role', 'option')
    const check = document.createElement('span')
    check.className = 'speech-speed-check'
    check.textContent = '✓'
    check.setAttribute('aria-hidden', 'true')
    button.append(check, document.createTextNode(text))
    button.addEventListener(
      'click',
      () => {
        update(choice)
        select(choice)
        close(true)
      },
      { signal },
    )
    options.push({ button, value: choice, label: text })
    menu.append(button)
  }
  update(initial)
  const focusOption = (index: number) => {
    const option = options[index]
    if (!option) return
    for (const candidate of options) candidate.button.tabIndex = candidate === option ? 0 : -1
    option.button.focus({ preventScroll: true })
  }
  const open = () => {
    if (menu.matches(':popover-open')) return
    menu.style.visibility = 'hidden'
    menu.showPopover()
    trigger.setAttribute('aria-expanded', 'true')
    let focusPending = true
    const position = async () => {
      const { x, y } = await computePosition(trigger, menu, {
        placement: 'bottom-end',
        strategy: 'fixed',
        middleware: [offset(3), flip(), shift({ padding: 12 })],
      })
      if (!menu.matches(':popover-open') || signal.aborted) return
      Object.assign(menu.style, { left: `${x}px`, top: `${y}px`, visibility: 'visible' })
      if (focusPending) {
        focusPending = false
        focusOption(
          options.findIndex(option => option.button.getAttribute('aria-selected') === 'true'),
        )
      }
    }
    stopPositioning = autoUpdate(trigger, menu, position)
  }
  trigger.addEventListener('click', () => (menu.matches(':popover-open') ? close() : open()), {
    signal,
  })
  picker.addEventListener(
    'keydown',
    event => {
      if (event.key === 'Escape' && menu.matches(':popover-open')) {
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
      if (!menu.matches(':popover-open')) return
      if (event.key === 'Tab') {
        close()
        return
      }
      const current = options.findIndex(option => option.button === document.activeElement)
      const next =
        event.key === 'Home' || event.key === '1'
          ? 0
          : event.key === 'End' || event.key === '0'
            ? options.length - 1
            : event.key === 'ArrowDown'
              ? Math.min(options.length - 1, current + 1)
              : event.key === 'ArrowUp'
                ? Math.max(0, current - 1)
                : -1
      if (next >= 0) {
        event.preventDefault()
        focusOption(next)
      }
    },
    { signal },
  )
  picker.addEventListener(
    'focusout',
    event => {
      if (!(event.relatedTarget instanceof Node) || !picker.contains(event.relatedTarget)) close()
    },
    { signal },
  )
  document.addEventListener(
    'pointerdown',
    event => {
      if (!event.composedPath().includes(picker)) close()
    },
    { signal },
  )
  signal.addEventListener('abort', () => close(), { once: true })
  picker.append(label, trigger, menu)
  return { element: picker, close }
}
