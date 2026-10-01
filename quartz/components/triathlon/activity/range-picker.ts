import type { ActivityAnalysisRange, StravaActivityDetail } from '../../../plugins/stores/strava'
import { cyclingWorkoutLaps, zoneClock } from '../../../util/triathlon-card'
import { el, svg } from '../runtime/dom'

export const buildActivityRangePicker = (
  activity: StravaActivityDetail,
  text: (key: string) => string,
  select: (range: ActivityAnalysisRange | null) => void,
): { element: HTMLElement; dispose: () => void } => {
  const picker = el('div', 'tri-lab-date-picker tri-workspace-range')
  const trigger = document.createElement('button')
  trigger.type = 'button'
  trigger.className = 'tri-lab-date-trigger tri-workspace-range-trigger'
  trigger.setAttribute('aria-label', text('activity range'))
  trigger.setAttribute('aria-haspopup', 'listbox')
  trigger.setAttribute('aria-expanded', 'false')
  const value = el('span', 'tri-workspace-range-value', text('entire activity'))
  const chevron = svg('svg', {
    class: 'tri-lab-date-chevron',
    viewBox: '0 0 16 16',
    fill: 'none',
    'aria-hidden': 'true',
  })
  chevron.append(svg('path', { d: 'm4 6 4 4 4-4', stroke: 'currentColor', 'stroke-width': 1.4 }))
  trigger.append(value, chevron)
  const menu = el('div', 'tri-lab-date-menu tri-workspace-range-menu', undefined, {
    id: `tri-workspace-ranges-${activity.id}`,
    role: 'listbox',
    'aria-label': text('activity range'),
  })
  menu.hidden = true
  trigger.setAttribute('aria-controls', menu.id)
  const options: {
    button: HTMLButtonElement
    range: ActivityAnalysisRange | null
    label: string
  }[] = []
  const lapNumbers = new Map(cyclingWorkoutLaps(activity).map(lap => [lap.range.id, lap.index]))
  const addOption = (parent: HTMLElement, range: ActivityAnalysisRange | null): void => {
    const number = range?.kind === 'lap' ? lapNumbers.get(range.id) : undefined
    const name = number === undefined ? range?.label : `${text('lap')} ${number}`
    const label = range ? `${name} · ${zoneClock(range.durationS)}` : text('entire activity')
    const button = document.createElement('button')
    button.type = 'button'
    button.className = 'tri-lab-date-option tri-workspace-range-option'
    button.setAttribute('role', 'option')
    button.setAttribute('aria-selected', String(range === null))
    button.tabIndex = range === null ? 0 : -1
    button.append(
      el('span', 'tri-lab-date-check', '✓', { 'aria-hidden': 'true' }),
      el('span', 'tri-workspace-range-value', label),
    )
    options.push({ button, range, label })
    parent.append(button)
  }
  addOption(menu, null)
  for (const [kind, label] of [
    ['lap', 'Laps'],
    ['climb', 'Summit Freeride'],
    ['segment', 'Segments'],
  ]) {
    const ranges = activity.analysisRanges.filter(
      range => range.kind === kind && range.endElapsedS > range.startElapsedS,
    )
    if (!ranges.length) continue
    const group = el('div', 'tri-workspace-range-group', undefined, {
      role: 'group',
      'aria-label': text(label),
    })
    group.append(el('span', 'tri-workspace-range-heading', text(label), { 'aria-hidden': 'true' }))
    for (const range of ranges) addOption(group, range)
    menu.append(group)
  }
  picker.append(trigger, menu)
  const close = (restoreFocus = false): void => {
    menu.hidden = true
    trigger.setAttribute('aria-expanded', 'false')
    if (restoreFocus) trigger.focus({ preventScroll: true })
  }
  const focusOption = (index: number): void => {
    const option = options[index]
    if (!option) return
    for (const candidate of options) candidate.button.tabIndex = candidate === option ? 0 : -1
    option.button.focus({ preventScroll: true })
    option.button.scrollIntoView({ block: 'nearest' })
  }
  const open = (): void => {
    menu.hidden = false
    trigger.setAttribute('aria-expanded', 'true')
    focusOption(options.findIndex(option => option.button.getAttribute('aria-selected') === 'true'))
  }
  const onClick = (event: MouseEvent): void => {
    const target = event.target
    if (!(target instanceof Node)) return
    if (trigger.contains(target)) {
      if (menu.hidden) open()
      else close()
      return
    }
    const option = options.find(option => option.button.contains(target))
    if (!option) return
    value.textContent = option.label
    for (const candidate of options)
      candidate.button.setAttribute('aria-selected', String(candidate === option))
    select(option.range)
    close(true)
  }
  let search = ''
  let lastSearch = 0
  const onKey = (event: KeyboardEvent): void => {
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
    const index = options.findIndex(option => option.button === document.activeElement)
    const next =
      event.key === 'Home'
        ? 0
        : event.key === 'End'
          ? options.length - 1
          : event.key === 'ArrowDown'
            ? Math.min(options.length - 1, index + 1)
            : event.key === 'ArrowUp'
              ? Math.max(0, index - 1)
              : -1
    if (next >= 0) {
      event.preventDefault()
      focusOption(next)
    } else if (
      event.key.length === 1 &&
      event.key !== ' ' &&
      !event.ctrlKey &&
      !event.metaKey &&
      !event.altKey
    ) {
      const now = performance.now()
      search = `${now - lastSearch < 700 ? search : ''}${event.key.toLocaleLowerCase()}`
      lastSearch = now
      const matches = options.map((_, offset) => (index + offset + 1) % options.length)
      const match = matches.find(candidate =>
        options[candidate].label.toLocaleLowerCase().startsWith(search),
      )
      if (match !== undefined) {
        event.preventDefault()
        focusOption(match)
      }
    }
  }
  const onFocusout = (event: FocusEvent): void => {
    if (!(event.relatedTarget instanceof Node) || !picker.contains(event.relatedTarget)) close()
  }
  const onPointer = (event: PointerEvent): void => {
    if (!event.composedPath().includes(picker)) close()
  }
  picker.addEventListener('click', onClick)
  picker.addEventListener('keydown', onKey)
  picker.addEventListener('focusout', onFocusout)
  document.addEventListener('pointerdown', onPointer)
  return {
    element: picker,
    dispose: () => {
      picker.removeEventListener('click', onClick)
      picker.removeEventListener('keydown', onKey)
      picker.removeEventListener('focusout', onFocusout)
      document.removeEventListener('pointerdown', onPointer)
    },
  }
}
