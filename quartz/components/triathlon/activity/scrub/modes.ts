import { SHIFTING_SETTINGS, type ShiftingSetting } from '../../../../util/triathlon-card'

export type CyclingChartMode = 'distance' | 'power'
export type RunMetricMode =
  | 'vertical-oscillation'
  | 'vertical-ratio'
  | 'ground-contact-time'
  | 'ground-contact-balance'
  | 'step-speed-loss'
  | 'step-speed-loss-percent'

const isRunMetricMode = (value: string | undefined): value is RunMetricMode =>
  value === 'vertical-oscillation' ||
  value === 'vertical-ratio' ||
  value === 'ground-contact-time' ||
  value === 'ground-contact-balance' ||
  value === 'step-speed-loss' ||
  value === 'step-speed-loss-percent'

export const runMetricMode = (element: HTMLElement | null): RunMetricMode | null => {
  const value = element?.dataset.runMetricMode
  return isRunMetricMode(value) ? value : null
}

export const setRunMetricMode = (chart: HTMLElement, mode: RunMetricMode): void => {
  if (!chart.querySelector(`[data-run-metric-pane="${mode}"]`)) return
  chart.dataset.runMetricMode = mode
  const activity = chart.closest<HTMLElement>('.tri-act')
  activity?.classList.remove('tri-act--scrub')
  for (const pane of chart.querySelectorAll<HTMLElement>('[data-run-metric-pane]')) {
    const visible = pane.dataset.runMetricPane === mode
    pane.hidden = !visible
    pane.setAttribute('aria-hidden', String(!visible))
    pane.classList.remove('tri-elev-wrap--read')
  }
  for (const option of chart.querySelectorAll<HTMLButtonElement>('.tri-run-metric-mode'))
    option.setAttribute('aria-pressed', String(option.dataset.runMetricMode === mode))
}

export const cyclingChartMode = (element: HTMLElement | null): CyclingChartMode => {
  const activity = element?.closest<HTMLElement>('.tri-act')
  return (activity?.dataset.cyclingChartMode ?? element?.dataset.cyclingChartMode) === 'power'
    ? 'power'
    : 'distance'
}

export const setCyclingChartMode = (chart: HTMLElement, mode: CyclingChartMode): void => {
  const activity = chart.closest<HTMLElement>('.tri-act')
  const charts = activity
    ? Array.from(activity.querySelectorAll<HTMLElement>('.tri-cycling-mode-chart'))
    : [chart]
  if (activity) {
    activity.dataset.cyclingChartMode = mode
    activity.classList.remove('tri-act--scrub')
  }
  for (const sibling of charts) {
    sibling.dataset.cyclingChartMode = mode
    sibling.classList.remove('tri-elev-wrap--read')
    sibling.classList.remove('tri-chart--hover')
    const wattsReadout = sibling.querySelector<HTMLElement>('.tri-cycling-watts-readout')
    if (wattsReadout) wattsReadout.hidden = true
    sibling
      .querySelector('.tri-cycling-watts-heat-cell--active')
      ?.classList.remove('tri-cycling-watts-heat-cell--active')
    for (const option of sibling.querySelectorAll<HTMLButtonElement>('.tri-cycling-chart-mode'))
      option.setAttribute('aria-pressed', String(option.dataset.cyclingChartMode === mode))
    for (const pane of sibling.querySelectorAll<HTMLElement>('.tri-cycling-chart-pane')) {
      const visible = pane.dataset.cyclingChartMode === mode
      pane.hidden = !visible
      pane.setAttribute('aria-hidden', String(!visible))
    }
  }
}

const shiftDatasetKey = (setting: ShiftingSetting): string =>
  `shift${setting[0].toUpperCase()}${setting.slice(1)}`

const shiftOption = (setting: ShiftingSetting, option: string | undefined): string | null =>
  SHIFTING_SETTINGS.find(entry => entry.setting === setting)?.options.find(
    candidate => candidate === option,
  ) ?? null

// Settings live on the activity so a profile re-render (zoom, metric switch) keeps them.
const setShiftSetting = (chart: HTMLElement, setting: ShiftingSetting, option: string) => {
  if (!shiftOption(setting, option)) return
  const activity = chart.closest<HTMLElement>('.tri-act')
  const key = shiftDatasetKey(setting)
  if (activity) activity.dataset[key] = option
  const charts = activity
    ? Array.from(activity.querySelectorAll<HTMLElement>('.tri-shift-chart'))
    : [chart]
  for (const sibling of charts) {
    sibling.dataset[key] = option
    if (setting === 'view') sibling.classList.remove('tri-elev-wrap--read', 'tri-chart--hover')
    for (const item of sibling.querySelectorAll<HTMLElement>(
      `.tri-shift-settings-option[data-shift-setting="${setting}"]`,
    ))
      item.setAttribute('aria-checked', String(item.dataset.shiftOption === option))
  }
}

export const restoreShiftSettings = (root: HTMLElement): void => {
  const chart = root.querySelector<HTMLElement>('.tri-shift-chart')
  const activity = chart?.closest<HTMLElement>('.tri-act')
  if (!chart || !activity) return
  for (const { setting } of SHIFTING_SETTINGS) {
    const option = shiftOption(setting, activity.dataset[shiftDatasetKey(setting)])
    if (option) setShiftSetting(chart, setting, option)
  }
}

const shiftMenuItems = (menu: HTMLElement): HTMLButtonElement[] =>
  Array.from(menu.querySelectorAll<HTMLButtonElement>('.tri-shift-settings-option'))

const closeShiftMenu = (picker: HTMLElement, restoreFocus = false): void => {
  const trigger = picker.querySelector<HTMLButtonElement>('.tri-shift-settings-trigger')
  const menu = picker.querySelector<HTMLElement>('.tri-shift-settings-menu')
  if (!trigger || !menu || menu.hidden) return
  menu.hidden = true
  trigger.setAttribute('aria-expanded', 'false')
  if (restoreFocus) trigger.focus({ preventScroll: true })
}

const openShiftMenu = (picker: HTMLElement): void => {
  const trigger = picker.querySelector<HTMLButtonElement>('.tri-shift-settings-trigger')
  const menu = picker.querySelector<HTMLElement>('.tri-shift-settings-menu')
  if (!trigger || !menu) return
  menu.hidden = false
  trigger.setAttribute('aria-expanded', 'true')
  const items = shiftMenuItems(menu)
  ;(items.find(item => item.getAttribute('aria-checked') === 'true') ?? items[0])?.focus({
    preventScroll: true,
  })
}

export const setCyclingChartModeFromTarget = (target: EventTarget | null): boolean => {
  if (!(target instanceof Element)) return false
  const button = target.closest<HTMLButtonElement>('.tri-cycling-chart-mode')
  const chart = button?.closest<HTMLElement>('.tri-cycling-mode-chart')
  if (!button || !chart) return false
  setCyclingChartMode(chart, button.dataset.cyclingChartMode === 'power' ? 'power' : 'distance')
  return true
}

export const setRunMetricModeFromTarget = (target: EventTarget | null): boolean => {
  if (!(target instanceof Element)) return false
  const button = target.closest<HTMLButtonElement>('.tri-run-metric-mode')
  const chart = button?.closest<HTMLElement>('.tri-run-metric-chart')
  const mode = runMetricMode(button)
  if (!button || !chart || !mode) return false
  setRunMetricMode(chart, mode)
  return true
}

export const shiftSettingsFromTarget = (target: EventTarget | null): boolean => {
  if (!(target instanceof Element)) return false
  const picker = target.closest<HTMLElement>('.tri-shift-settings')
  const chart = picker?.closest<HTMLElement>('.tri-shift-chart')
  if (!picker || !chart) return false
  if (target.closest('.tri-shift-settings-trigger')) {
    const menu = picker.querySelector<HTMLElement>('.tri-shift-settings-menu')
    if (menu?.hidden) openShiftMenu(picker)
    else closeShiftMenu(picker)
    return true
  }
  const item = target.closest<HTMLButtonElement>('.tri-shift-settings-option')
  const setting = SHIFTING_SETTINGS.find(entry => entry.setting === item?.dataset.shiftSetting)
  if (!item || !setting || !item.dataset.shiftOption) return true
  setShiftSetting(chart, setting.setting, item.dataset.shiftOption)
  // Group and value only shape the histogram, so picking one shows it; labels apply to both views.
  if (setting.setting === 'group' || setting.setting === 'value')
    setShiftSetting(chart, 'view', 'hist')
  return true
}

export const onShiftMenuKey = (event: KeyboardEvent): void => {
  if (!(event.target instanceof Element)) return
  const picker = event.target.closest<HTMLElement>('.tri-shift-settings')
  const menu = picker?.querySelector<HTMLElement>('.tri-shift-settings-menu')
  if (!picker || !menu) return
  if (event.key === 'Escape' && !menu.hidden) {
    event.preventDefault()
    event.stopPropagation()
    closeShiftMenu(picker, true)
    return
  }
  if (event.target.closest('.tri-shift-settings-trigger')) {
    if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return
    event.preventDefault()
    openShiftMenu(picker)
    return
  }
  const items = shiftMenuItems(menu)
  const current = items.indexOf(event.target as HTMLButtonElement)
  if (current < 0) return
  const next =
    event.key === 'ArrowDown'
      ? (current + 1) % items.length
      : event.key === 'ArrowUp'
        ? (current - 1 + items.length) % items.length
        : event.key === 'Home'
          ? 0
          : event.key === 'End'
            ? items.length - 1
            : -1
  if (next < 0) {
    if (event.key === 'Tab') closeShiftMenu(picker)
    return
  }
  event.preventDefault()
  items[next].focus({ preventScroll: true })
}

export const closeShiftMenusOutside = (scope: HTMLElement, target: EventTarget | null): void => {
  for (const picker of scope.querySelectorAll<HTMLElement>('.tri-shift-settings'))
    if (!(target instanceof Node && picker.contains(target))) closeShiftMenu(picker)
}

export const onModeKey = (event: KeyboardEvent): void => {
  if (event.key !== 'Enter' && event.key !== ' ') return
  if (setRunMetricModeFromTarget(event.target)) {
    event.preventDefault()
    return
  }
  if (!setCyclingChartModeFromTarget(event.target)) return
  event.preventDefault()
}
