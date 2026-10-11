import { defaultForestSettings, forestLimits, type ForestSettings } from './scenery'

const fields: { key: keyof ForestSettings; label: string; name: string; suffix: string }[] = [
  { key: 'density', label: 'trees', name: 'Tree count', suffix: '' },
  { key: 'size', label: 'size', name: 'Tree size', suffix: '%' },
  { key: 'depth', label: 'branches', name: 'Branch depth', suffix: '' },
  { key: 'angle', label: 'angle', name: 'Branch angle', suffix: '°' },
  { key: 'curl', label: 'curl', name: 'Branch curl', suffix: '°' },
  { key: 'branching', label: 'branch chance', name: 'Branching probability', suffix: '%' },
  { key: 'variation', label: 'variation', name: 'Tree variation', suffix: '%' },
  { key: 'dapple', label: 'dapple', name: 'Canopy dapple', suffix: '%' },
]

const rangeProgress = (key: keyof ForestSettings, value: number) => {
  const { min, max } = forestLimits[key]
  return `${((value - min) / (max - min)) * 100}%`
}

export function forestControlsMarkup(id: string) {
  return `
    <button class="trail-tree-toggle" type="button" aria-label="Tree settings" aria-expanded="false" aria-controls="${id}-forest">
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M11 2h2v2h2v2h2v2h-2v2h3v2h2v2h-4v2h5v3H3v-3h5v-2H4v-2h2v-2h3V8H7V6h2V4h2ZM11 19h2v4h-2Z"/></svg>
    </button>
    <div class="trail-tree-panel" id="${id}-forest" role="group" aria-label="Tree generation" hidden>
      <div class="trail-tree-panel-head"><span>grove</span><span>l-system</span><button class="trail-tree-close" type="button" aria-label="Close tree settings"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="m4 4 8 8M12 4l-8 8"/></svg></button></div>
      <div class="trail-tree-controls">
        ${fields
          .map(({ key, label, name, suffix }) => {
            const { min, max, step } = forestLimits[key]
            return `<div class="trail-tree-control">
            <label for="${id}-${key}">${label}<output for="${id}-${key}" data-forest-output="${key}">${defaultForestSettings[key]}${suffix}</output></label>
            <input type="range" id="${id}-${key}" aria-label="${name}" aria-valuetext="${defaultForestSettings[key]}${suffix}" min="${min}" max="${max}" step="${step}" value="${defaultForestSettings[key]}" data-forest-field="${key}" style="--trail-control-progress: ${rangeProgress(key, defaultForestSettings[key])}"/>
          </div>`
          })
          .join('')}
      </div>
    </div>`
}

export function mountForestControls(
  root: HTMLElement,
  signal: AbortSignal,
  onChange: (settings: ForestSettings) => void,
  onOpen: () => void,
) {
  const toggle = root.querySelector<HTMLButtonElement>('.trail-tree-toggle')
  const panel = root.querySelector<HTMLElement>('.trail-tree-panel')
  const dismiss = root.querySelector<HTMLButtonElement>('.trail-tree-close')
  if (!toggle || !panel || !dismiss) throw new Error('Tree controls are incomplete')
  const settings = { ...defaultForestSettings }
  const close = () => {
    panel.hidden = true
    toggle.setAttribute('aria-expanded', 'false')
  }
  const dismissPanel = () => {
    close()
    toggle.focus({ preventScroll: true })
  }
  dismiss.addEventListener('click', dismissPanel, { signal })
  panel.addEventListener(
    'keydown',
    event => {
      if (event.key !== 'Escape') return
      event.preventDefault()
      event.stopPropagation()
      dismissPanel()
    },
    { signal },
  )
  toggle.addEventListener(
    'click',
    () => {
      if (!panel.hidden) {
        close()
        return
      }
      panel.hidden = false
      toggle.setAttribute('aria-expanded', 'true')
      onOpen()
    },
    { signal },
  )
  for (const { key, suffix } of fields) {
    const input = panel.querySelector<HTMLInputElement>(`[data-forest-field="${key}"]`)
    const output = panel.querySelector<HTMLOutputElement>(`[data-forest-output="${key}"]`)
    if (!input || !output) throw new Error('Tree slider is incomplete')
    input.addEventListener(
      'input',
      () => {
        const { min, max } = forestLimits[key]
        settings[key] = Math.max(min, Math.min(max, Math.round(input.valueAsNumber)))
        output.value = `${settings[key]}${suffix}`
        input.setAttribute('aria-valuetext', output.value)
        input.style.setProperty('--trail-control-progress', rangeProgress(key, settings[key]))
        onChange({ ...settings })
      },
      { signal },
    )
    input.addEventListener(
      'keydown',
      event => {
        if (
          [
            'ArrowLeft',
            'ArrowRight',
            'ArrowUp',
            'ArrowDown',
            'Home',
            'End',
            'PageUp',
            'PageDown',
          ].includes(event.key)
        ) {
          event.stopPropagation()
        }
      },
      { signal },
    )
  }
  return { toggle, close, isOpen: () => !panel.hidden }
}
