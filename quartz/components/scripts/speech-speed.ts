import { buildSelect } from '../controls/select'

export type SpeechSpeed = '1' | '0.8'

type SpeechPicker = { element: HTMLElement; close: () => void }

export function createSpeechSpeed(
  initial: SpeechSpeed,
  select: (value: SpeechSpeed) => void,
  signal: AbortSignal,
): SpeechPicker {
  return createSpeechPicker({
    id: 'speech-speed-options',
    label: 'vitesse',
    menuLabel: 'vitesse de lecture',
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
  label,
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
  picker.className = 'speech-picker'
  const caption = document.createElement('span')
  caption.textContent = label
  // The panel paints with containment, so the menu takes the top layer.
  const control = buildSelect<T>({
    id,
    label: menuLabel,
    groups: [{ options: choices.map(([value, text]) => ({ value, label: text })) }],
    selected: initial,
    onSelect: select,
    className: 'speech-select',
    floating: true,
  })
  signal.addEventListener('abort', control.mount(), { once: true })
  picker.append(caption, control.element)
  return {
    element: picker,
    close: () => {
      if (control.trigger.getAttribute('aria-expanded') === 'true') control.trigger.click()
    },
  }
}
