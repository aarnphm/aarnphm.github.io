import { ACTIVITY_CARD_STATS } from '../../../util/press-kit-activity'
import { DAY_LAYOUTS } from '../press-kit-card/day-layout'
import { blockDisplayKeys, type TrainingCardBlock } from '../press-kit-day'
import { localizeActivity, shortMetricLabel } from '../press-kit-i18n'
import { runsToPlain } from '../press-kit-text'
import {
  $,
  blockDisplays,
  selectedActivities,
  selectedBlocks,
  selectedMetrics,
  session,
  shareLocale,
  show,
} from './session-state'
import { readingTime, subjectActivities, syncMetricIncludes } from './share-subject'

export function layoutMetricControls() {
  if (!session.dialog?.open) return
  const context = document.createElement('canvas').getContext('2d')
  if (!context) return
  const textWidth = (element: HTMLElement): number => {
    const style = getComputedStyle(element)
    context.font = `${style.fontWeight} ${style.fontSize} ${style.fontFamily}`
    return context.measureText(element.textContent ?? '').width
  }
  for (const metrics of session.dialog.querySelectorAll<HTMLElement>(
    '.press-kit-metric-group[open] .press-kit-data-grid',
  )) {
    if (!metrics.clientWidth) continue
    metrics.querySelector('.press-kit-data-filler')?.remove()
    const columnWidth = (metrics.clientWidth - parseFloat(getComputedStyle(metrics).columnGap)) / 2
    const items = Array.from(metrics.querySelectorAll<HTMLButtonElement>('[data-activity-metric]'))
      .map(button => {
        const label = button.querySelector<HTMLElement>('.press-kit-metric-label')
        const value = button.querySelector<HTMLElement>('small')
        if (!label || !value) return null
        const style = getComputedStyle(button)
        const inset =
          parseFloat(style.paddingInlineStart) +
          parseFloat(style.paddingInlineEnd) +
          parseFloat(style.borderInlineStartWidth) +
          parseFloat(style.borderInlineEndWidth)
        const available = columnWidth - inset
        const wide = textWidth(label) > available || textWidth(value) > available
        button.classList.toggle('press-kit-data-toggle--wide', wide)
        return { button, wide, index: Number(button.dataset.metricOrder ?? 0) }
      })
      .filter(item => item !== null)
      .sort((a, b) => Number(a.wide) - Number(b.wide) || a.index - b.index)
    if (items.some((item, index) => metrics.children[index] !== item.button)) {
      const focus = document.activeElement
      metrics.append(...items.map(item => item.button))
      if (focus instanceof HTMLElement && metrics.contains(focus))
        focus.focus({ preventScroll: true })
    }
    // An odd count of half-width toggles leaves a hole before the full-width rows; an empty
    // cell closes the table there.
    const narrow = items.filter(item => !item.wide)
    if (narrow.length % 2) {
      const filler = document.createElement('span')
      filler.className = 'press-kit-data-filler'
      filler.setAttribute('aria-hidden', 'true')
      narrow[narrow.length - 1].button.after(filler)
    }
  }
}

export function mountMetricControls() {
  const host = $('.press-kit-metrics')
  if (!host) return
  const expanded = new Set(
    Array.from(
      host.querySelectorAll<HTMLElement>('[data-metric-group][open]'),
      group => group.dataset.metricGroup,
    ),
  )
  host.replaceChildren()
  for (const source of subjectActivities()) {
    const activity = localizeActivity(source, shareLocale())
    const previous = selectedMetrics.get(activity.url)
    const defaults = new Set<string>(ACTIVITY_CARD_STATS)
    if (activity.sport === 'run') {
      defaults.delete('power')
      defaults.add('metric:cadence')
    }
    selectedMetrics.set(
      activity.url,
      new Set(
        activity.stats.filter(stat => (previous ?? defaults).has(stat.key)).map(stat => stat.key),
      ),
    )
    const group = document.createElement('details')
    group.className = 'press-kit-metric-group'
    group.dataset.metricGroup = activity.url
    group.open = expanded.has(activity.url)
    const summary = document.createElement('summary')
    summary.textContent = `mesures · ${activity.title}`
    group.append(summary)
    const actions = document.createElement('div')
    actions.className = 'press-kit-metric-actions'
    for (const [action, label] of [
      ['all', 'tout'],
      ['none', 'aucune'],
    ]) {
      const button = document.createElement('button')
      button.type = 'button'
      button.className = 'g-button'
      button.dataset.metricAction = action
      button.dataset.metricActivity = activity.url
      button.textContent = label
      actions.append(button)
    }
    group.append(actions)
    const metrics = document.createElement('div')
    metrics.className = 'press-kit-data-grid'
    for (const [index, stat] of activity.stats.entries()) {
      const button = document.createElement('button')
      button.type = 'button'
      button.className = 'press-kit-data-toggle'
      button.dataset.activityMetric = stat.key
      button.dataset.metricActivity = activity.url
      button.dataset.metricOrder = String(index)
      button.title = [stat.label, stat.value, stat.provenance].filter(Boolean).join(' · ')
      button.setAttribute('aria-label', `${activity.title} · ${stat.label}`)
      button.setAttribute('aria-pressed', String(selectedMetrics.get(activity.url)?.has(stat.key)))
      const label = document.createElement('span')
      label.className = 'press-kit-data-label'
      const labelText = document.createElement('span')
      labelText.className = 'press-kit-metric-label'
      labelText.textContent = shortMetricLabel(stat.label)
      label.append(labelText)
      button.append(label)
      const value = document.createElement('small')
      value.textContent = stat.value
      button.append(value)
      metrics.append(button)
    }
    group.append(metrics)
    host.append(group)
  }
  if (subjectActivities().length) syncMetricIncludes()
}

function displayControls(block: TrainingCardBlock): HTMLDetailsElement {
  const group = document.createElement('details')
  group.className = 'press-kit-block-display'
  group.dataset.blockOptions = block.id
  const summary = document.createElement('summary')
  summary.textContent = `affichage · ${block.kind === 'entry' ? 'entrée' : block.title}`
  group.append(summary)
  const options = document.createElement('div')
  options.className = 'press-kit-data-grid press-kit-data-grid--compact'
  blockDisplays.set(block.id, { title: show.title, date: show.date, metrics: show.metrics })
  for (const key of blockDisplayKeys(block)) {
    const button = document.createElement('button')
    button.type = 'button'
    button.className = 'press-kit-data-toggle'
    button.dataset.blockDisplay = key
    button.dataset.displayBlock = block.id
    const label = key === 'title' ? 'titre' : key === 'date' ? 'date' : 'mesures'
    button.setAttribute(
      'aria-label',
      `${block.kind === 'entry' ? 'entrée' : block.title} · ${label}`,
    )
    const text = document.createElement('span')
    text.className = 'press-kit-data-label'
    text.textContent = label
    button.append(text)
    options.append(button)
  }
  group.append(options)
  return group
}

// Built here, beside the block list it arranges; syncControls shows it for three or more blocks.
function dayLayoutControl(): HTMLElement {
  const setting = document.createElement('div')
  setting.className = 'press-kit-chart-setting press-kit-day-layout'
  setting.hidden = true
  const legend = document.createElement('span')
  legend.className = 'press-kit-legend'
  legend.id = 'press-kit-day-layout-label'
  legend.textContent = 'disposition'
  const segments = document.createElement('div')
  segments.className = 'press-kit-segments'
  segments.setAttribute('role', 'radiogroup')
  segments.setAttribute('aria-labelledby', legend.id)
  for (const [value, text] of DAY_LAYOUTS) {
    const label = document.createElement('label')
    label.className = 'press-kit-segment'
    const input = document.createElement('input')
    input.type = 'radio'
    input.name = 'press-kit-day-layout'
    input.value = value
    const caption = document.createElement('span')
    caption.textContent = text
    label.append(input, caption)
    segments.append(label)
  }
  const hint = document.createElement('p')
  hint.className = 'press-kit-hint'
  hint.hidden = true
  setting.append(legend, segments, hint)
  return setting
}

export function mountDayControls() {
  const host = $('.press-kit-day-blocks')
  if (!host) return
  if (!$('.press-kit-day-layout')) host.after(dayLayoutControl())
  host.replaceChildren()
  selectedBlocks.clear()
  selectedActivities.clear()
  blockDisplays.clear()
  const entryDisplay = $('.press-kit-entry-display')
  entryDisplay?.replaceChildren()
  if (session.subject?.day && entryDisplay) {
    const prose = runsToPlain(session.subject.excerpt).trim()
    entryDisplay.append(
      displayControls({
        id: 'entry',
        kind: 'entry',
        title: session.subject.entryTitle ?? 'entrée',
        url: session.subject.url,
        text: prose,
        date: session.subject.entryDate,
        reading: readingTime(prose),
        activities: [],
        rows: [],
      }),
    )
  }
  const blocks = session.subject?.day?.blocks ?? []
  const activities = session.subject?.day ? subjectActivities() : []
  if (activities.length) {
    const field = document.createElement('fieldset')
    field.className = 'press-kit-field'
    const legend = document.createElement('legend')
    legend.textContent = 'activités'
    const list = document.createElement('div')
    list.className = 'press-kit-data-grid press-kit-block-list'
    field.append(legend, list)
    host.append(field)
    for (const activity of activities) {
      selectedActivities.add(activity.url)
      const button = document.createElement('button')
      button.type = 'button'
      button.className = 'press-kit-data-toggle'
      button.dataset.dayActivity = activity.url
      button.setAttribute('aria-pressed', 'true')
      const label = document.createElement('span')
      label.className = 'press-kit-data-label'
      label.textContent = activity.title
      button.append(label)
      list.append(button)
      const block = blocks.find(
        block => block.kind === 'activity' && block.activities[0]?.url === activity.url,
      )
      if (block) list.append(displayControls(block))
    }
  }
  for (const block of blocks) {
    const selected = block.kind !== 'memo'
    if (selected) selectedBlocks.add(block.id)
    if (block.kind === 'activity') continue
    const button = document.createElement('button')
    button.type = 'button'
    button.className = 'press-kit-data-toggle'
    button.dataset.dayBlock = block.id
    button.setAttribute('aria-pressed', String(selected))
    const label = document.createElement('span')
    label.className = 'press-kit-data-label'
    label.textContent = block.title
    button.append(label)
    host.append(button)
    host.append(displayControls(block))
  }
}
