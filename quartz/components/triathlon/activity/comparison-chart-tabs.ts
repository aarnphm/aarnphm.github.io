import type { TriathlonPresentation } from '../../../util/triathlon-presentation'
import { triText } from '../../../util/triathlon-i18n'
import { el } from '../runtime/dom'
import { nextMapMetricShortcutIndex } from '../shell/command-palette'

const chartShortLabels: Readonly<Record<string, string>> = {
  elevation: 'E',
  speed: 'S',
  hr: 'HR',
  power: 'W',
  cadence: 'C',
  respiration: 'R',
  temperature: 'T',
  'skin-temperature': 'ST',
  'stride-length': 'SL',
  'ground-contact-time': 'GCT',
  'vertical-oscillation': 'VO',
  'swim-pace': 'P',
  'stroke-rate': 'SR',
  'gear-ratio-distribution': 'GR',
  'power-distribution': 'PD',
  'power-curve': 'CP',
  'hr-zones': 'HZ',
  'power-zones': 'PZ',
}

export const mountComparisonChartTabs = (
  comparison: HTMLElement | SVGElement,
  presentation: TriathlonPresentation,
  charts: readonly HTMLElement[],
  onChange: (chart: HTMLElement | null) => void,
): (() => void) => {
  const viewport = comparison.querySelector<HTMLElement>('.tri-compare-charts-viewport')
  const panel = viewport?.querySelector<HTMLElement>('.tri-compare-charts')
  if (!viewport || !panel || charts.length === 0) return () => {}

  const text = (key: string): string => triText(presentation.locale, key)
  const tablist = el('div', 'tri-map-tablist tri-compare-chart-tabs', undefined, {
    role: 'tablist',
    'aria-label': text('comparison charts'),
  })
  const choices = [
    { chart: null, label: text('all charts'), shortLabel: 'ALL' },
    ...charts.flatMap(chart => {
      const key = chart.dataset.compareChart
      const shortLabel = key ? chartShortLabels[key] : undefined
      const title = chart.querySelector<HTMLElement>('.tri-compare-title')
      const label = title?.textContent?.trim()
      return shortLabel && label && Number(chart.dataset.available) > 0
        ? [{ chart, label, shortLabel }]
        : []
    }),
  ]
  const id = `tri-compare-charts-${crypto.randomUUID()}`
  panel.id = id
  panel.setAttribute('role', 'tabpanel')
  const tabs = choices.map((choice, index) => {
    const tab = document.createElement('button')
    tab.className = 'tri-map-tab'
    tab.type = 'button'
    tab.id = `${id}-tab-${index}`
    tab.setAttribute('role', 'tab')
    tab.setAttribute('aria-label', choice.label)
    tab.setAttribute('aria-controls', id)
    tab.dataset.index = String(index)
    tab.dataset.shortcut = choice.shortLabel[0].toLowerCase()
    tab.style.setProperty('--tri-map-tab-shortcut-width', `${choice.shortLabel.length}ch`)
    tab.style.setProperty('--tri-map-tab-label-width', `${choice.label.length}ch`)
    tab.append(
      el('span', 'tri-map-tab-shortcut', choice.shortLabel, { 'aria-hidden': 'true' }),
      el('span', 'tri-map-tab-label', choice.label, { 'aria-hidden': 'true' }),
    )
    tablist.append(tab)
    return tab
  })
  let active = 0
  const select = (index: number, animate: boolean): void => {
    const choice = choices[index]
    const selected = tabs[index]
    if (!choice || !selected) return
    active = index
    if (!animate) tablist.dataset.motion = 'instant'
    for (const [i, tab] of tabs.entries()) {
      tab.setAttribute('aria-selected', String(i === index))
      tab.tabIndex = i === index ? 0 : -1
    }
    panel.setAttribute('aria-labelledby', selected.id)
    for (const chart of charts) {
      chart.hidden = choice.chart !== null && chart !== choice.chart
      chart.inert = chart.hidden
    }
    panel.scrollTop = 0
    onChange(choice.chart)
    if (!animate) {
      tablist.getBoundingClientRect()
      delete tablist.dataset.motion
    }
  }
  const onClick = (event: MouseEvent): void => {
    const tab =
      event.target instanceof Element
        ? event.target.closest<HTMLButtonElement>('.tri-map-tab[data-index]')
        : null
    if (!tab || !tablist.contains(tab)) return
    select(Number(tab.dataset.index), event.detail > 0)
  }
  const onKeyDown = (event: KeyboardEvent): void => {
    if (event.ctrlKey || event.metaKey || event.altKey || event.isComposing || event.repeat) return
    const next =
      event.key === 'ArrowLeft'
        ? (active - 1 + tabs.length) % tabs.length
        : event.key === 'ArrowRight'
          ? (active + 1) % tabs.length
          : event.key === 'Home'
            ? 0
            : event.key === 'End'
              ? tabs.length - 1
              : nextMapMetricShortcutIndex(
                  tabs.map(tab => tab.dataset.shortcut),
                  active,
                  event.key,
                )
    if (next < 0) return
    event.preventDefault()
    event.stopPropagation()
    select(next, false)
    tabs[next]?.focus({ preventScroll: true })
    tabs[next]?.scrollIntoView({ block: 'nearest', inline: 'nearest' })
  }
  viewport.prepend(tablist)
  select(0, false)
  tablist.addEventListener('click', onClick)
  tablist.addEventListener('keydown', onKeyDown)
  return () => {
    tablist.removeEventListener('click', onClick)
    tablist.removeEventListener('keydown', onKeyDown)
    tablist.remove()
    panel.removeAttribute('id')
    panel.removeAttribute('role')
    panel.removeAttribute('aria-labelledby')
    for (const chart of charts) {
      chart.hidden = false
      chart.inert = false
    }
  }
}
