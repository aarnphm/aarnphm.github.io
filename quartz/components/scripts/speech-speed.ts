export type SpeechSpeed = '1' | '0.8'

export function createSpeechSpeed(
  initial: SpeechSpeed,
  select: (value: SpeechSpeed) => void,
  signal: AbortSignal,
): { element: HTMLElement; close: (restoreFocus?: boolean) => void } {
  const picker = document.createElement('div')
  picker.className = 'speech-speed'
  const label = document.createElement('span')
  label.textContent = 'Vitesse'
  const trigger = document.createElement('button')
  trigger.type = 'button'
  trigger.className = 'speech-speed-trigger'
  trigger.setAttribute('aria-haspopup', 'listbox')
  trigger.setAttribute('aria-expanded', 'false')
  trigger.setAttribute('aria-controls', 'speech-speed-options')
  const value = document.createElement('span')
  const chevron = document.createElementNS('http://www.w3.org/2000/svg', 'svg')
  chevron.setAttribute('viewBox', '0 0 16 16')
  chevron.setAttribute('aria-hidden', 'true')
  const path = document.createElementNS('http://www.w3.org/2000/svg', 'path')
  path.setAttribute('d', 'm4 6 4 4 4-4')
  chevron.append(path)
  trigger.append(value, chevron)
  const menu = document.createElement('div')
  menu.id = 'speech-speed-options'
  menu.className = 'speech-speed-menu'
  menu.setAttribute('role', 'listbox')
  menu.setAttribute('aria-label', 'Vitesse de lecture')
  menu.hidden = true
  const options: { button: HTMLButtonElement; value: SpeechSpeed; label: string }[] = []
  const close = (restoreFocus = false) => {
    menu.hidden = true
    trigger.setAttribute('aria-expanded', 'false')
    if (restoreFocus) trigger.focus({ preventScroll: true })
  }
  const update = (selected: SpeechSpeed) => {
    for (const option of options) {
      const active = option.value === selected
      option.button.setAttribute('aria-selected', String(active))
      option.button.tabIndex = active ? 0 : -1
      if (active) {
        value.textContent = option.label
        trigger.setAttribute('aria-label', `Vitesse de lecture : ${option.label}`)
      }
    }
  }
  for (const [speed, text] of [
    ['1', '1×'],
    ['0.8', '0,8×'],
  ]) {
    if (speed !== '1' && speed !== '0.8') continue
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
        update(speed)
        select(speed)
        close(true)
      },
      { signal },
    )
    options.push({ button, value: speed, label: text })
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
    menu.hidden = false
    trigger.setAttribute('aria-expanded', 'true')
    focusOption(options.findIndex(option => option.button.getAttribute('aria-selected') === 'true'))
  }
  trigger.addEventListener('click', () => (menu.hidden ? open() : close()), { signal })
  picker.addEventListener(
    'keydown',
    event => {
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
  picker.append(label, trigger, menu)
  return { element: picker, close }
}
