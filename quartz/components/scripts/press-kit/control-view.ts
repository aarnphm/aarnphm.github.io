import type { CardModel } from '../press-kit-card'
import {
  ACTIVITY_CARD_STATS,
  isActivityCardChartKey,
  OVERLAY_OPTIONS,
  type OverlayKey,
} from '../../../util/press-kit-activity'
import { triLabel } from '../../../util/triathlon-i18n'
import { buildSelect } from '../../controls/select'
import { FORMATS } from '../press-kit-card'
import { offersDayLayout, STACK_LIMIT, stacks } from '../press-kit-card/day-layout'
import { blockDisplayKeys, isBlockDisplayKey } from '../press-kit-day'
import { localizeActivity } from '../press-kit-i18n'
import { cardDate, runsToPlain, runsToText } from '../press-kit-text'
import { defaultNote } from './link-sharing'
import { schedule } from './picture-sharing'
import {
  $,
  blockDisplays,
  selectedActivities,
  selectedMetrics,
  session,
  shareLocale,
  show,
  type PressKitTab,
} from './session-state'
import {
  activeActivities,
  activeDayBlocks,
  availableTabs,
  citationAllowed,
  headingText,
  titleText,
} from './share-subject'
import { propositionsText } from './tractatus'
import { layoutMetricControls, mountMetricControls } from './training-controls'

function anchorGroups() {
  const options: { value: string; label: string }[] = [
    {
      value: 'page',
      label:
        session.subject?.kind === 'activity'
          ? 'activité entière'
          : session.subject?.kind === 'day'
            ? 'journée entière'
            : session.subject?.kind === 'entry'
              ? 'entrée entière'
              : 'page entière',
    },
  ]
  if (session.quote.fragment) options.push({ value: 'passage', label: 'passage sélectionné' })
  const headings = (session.subject?.headings ?? []).map(heading => ({
    value: `#${heading.id}`,
    label: `${'  '.repeat(Math.max(0, Number(heading.tagName[1]) - 2))}${headingText(heading)}`,
  }))
  const blocks =
    session.subject?.day?.blocks.map(block => ({
      value: `block:${block.id}`,
      label: block.title,
    })) ?? []
  return [
    { options },
    ...(blocks.length ? [{ label: 'blocs', options: blocks }] : []),
    ...(headings.length > 0 ? [{ label: 'sections', options: headings }] : []),
  ]
}

export function mountAnchor(signal: AbortSignal) {
  const host = $('.press-kit-anchor')
  if (!host) return
  session.unmountAnchor?.()
  host.replaceChildren()
  session.anchorSelect = buildSelect<string>({
    id: 'press-kit-anchor-menu',
    label: 'ancre',
    groups: anchorGroups(),
    selected: session.anchor,
    className: 'press-kit-select',
    onSelect: value => {
      session.anchor = value
      if (!session.noteEdited) $<HTMLTextAreaElement>('#press-kit-note')!.value = defaultNote()
      schedule()
    },
  })
  host.append(session.anchorSelect.element)
  const unmount = session.anchorSelect.mount()
  session.unmountAnchor = unmount
  signal.addEventListener('abort', unmount, { once: true })
}

export function mountOverlay(signal: AbortSignal) {
  const host = $('.press-kit-overlay')
  if (!host) return
  session.overlaySelect = buildSelect<OverlayKey>({
    id: 'press-kit-overlay-menu',
    label: 'superposition',
    groups: [{ options: OVERLAY_OPTIONS.map(([value, label]) => ({ value, label })) }],
    selected: session.overlay,
    className: 'press-kit-select',
    onSelect: value => {
      session.overlay = value
      schedule()
    },
  })
  host.replaceChildren(session.overlaySelect.element)
  signal.addEventListener('abort', session.overlaySelect.mount(), { once: true })
}

const HELP: Record<'image' | 'links', string> = {
  image: '{1}{2}{3} onglets · {f} format · {t} thème · {d} télécharger · {c} copier · {esc} fermer',
  links: '{1}{2}{3} onglets · {esc} fermer',
}

function setHelp() {
  const status = $('.press-kit-help')
  if (!status) return
  const template = HELP[session.tab === 'links' ? 'links' : 'image'].replace(
    '{3}',
    citationAllowed() ? '{3}' : '',
  )
  status.replaceChildren(
    ...template.split(/\{([^}]+)\}/).map((part, index) => {
      if (index % 2 === 0) return part
      const kbd = document.createElement('kbd')
      kbd.textContent = part
      return kbd
    }),
  )
}

export function syncControls() {
  if (!session.dialog) return
  session.dialog.dataset.tab = session.tab
  session.dialog.dataset.subject = session.subject?.kind ?? 'page'
  session.dialog.dataset.training = String(
    session.subject?.kind === 'activity' || session.subject?.kind === 'day',
  )
  session.dialog.dataset.transparent = String(show.transparent)
  session.dialog.dataset.citation = String(citationAllowed())
  session.dialog.querySelectorAll<HTMLElement>('[data-tractatus-only]').forEach(element => {
    element.hidden = !session.quote.propositions?.length
  })
  session.dialog.querySelectorAll<HTMLButtonElement>('.press-kit-tab').forEach(button => {
    const selected = button.dataset.tab === session.tab
    button.setAttribute('aria-selected', String(selected))
    button.tabIndex = selected ? 0 : -1
  })
  $('#press-kit-panel')?.setAttribute('aria-labelledby', `press-kit-tab-${session.tab}`)
  session.dialog
    .querySelectorAll<HTMLInputElement>('input[name="press-kit-format"]')
    .forEach(input => (input.checked = input.value === session.format))
  session.dialog
    .querySelectorAll<HTMLInputElement>('input[name="press-kit-theme"]')
    .forEach(input => (input.checked = input.value === session.theme))
  session.overlaySelect?.setSelected(session.overlay)
  session.dialog
    .querySelectorAll<HTMLInputElement>('input[name="press-kit-language"]')
    .forEach(input => (input.checked = input.value === shareLocale()))
  const activities = activeActivities()
  const blocks = activeDayBlocks()
  session.dialog.querySelectorAll<HTMLButtonElement>('[data-include]').forEach(button => {
    if (isIncludeKey(button.name)) {
      const key = button.name
      const values = isBlockDisplayKey(key)
        ? Array.from(blockDisplays.entries())
            .filter(([id]) =>
              id === 'entry'
                ? show.entry
                : blocks.some(block => block.id === id && blockDisplayKeys(block).includes(key)),
            )
            .map(([, display]) => display[key])
        : []
      button.setAttribute(
        'aria-pressed',
        values.length && values.some(Boolean) && values.some(value => !value)
          ? 'mixed'
          : String(values.length ? values.every(Boolean) : show[key]),
      )
    }
    if (button.closest('[data-activity-only]')) {
      button.disabled =
        button.name === 'map'
          ? !activities.some(activity => activity.route.length > 0)
          : ACTIVITY_CARD_STATS.some(key => key === button.name) &&
            !activities.some(activity => activity.stats.some(stat => stat.key === button.name))
    }
    if (isActivityCardChartKey(button.name)) {
      const available = activities.some(activity =>
        activity.charts.some(chart => chart.key === button.name),
      )
      button.disabled = !available
      button.hidden = !available
    }
    if (button.name === 'entry')
      button.disabled = !runsToPlain(session.subject?.excerpt ?? []).trim()
  })
  session.dialog.querySelectorAll<HTMLButtonElement>('[data-block-display]').forEach(button => {
    const key = button.dataset.blockDisplay
    const id = button.dataset.displayBlock
    if (!key || !id || !isBlockDisplayKey(key)) return
    const display = blockDisplays.get(id)
    button.setAttribute('aria-pressed', String(display?.[key] ?? show[key]))
    button.disabled = id === 'entry' ? !show.entry : !blocks.some(block => block.id === id)
  })
  const charts = $<HTMLButtonElement>('[data-include="cyclingPower"]')?.closest('fieldset')
  if (charts) charts.hidden = !activities.some(activity => activity.charts.length > 0)
  const powerCharts = activities.flatMap(activity =>
    activity.charts.filter(chart => chart.key === 'cyclingPower'),
  )
  const availableWindow = (seconds: number) =>
    powerCharts.length > 0 &&
    powerCharts.every(
      chart => chart.series?.some(series => series.seconds === seconds) ?? seconds === 30,
    )
  if (!availableWindow(session.powerWindow)) session.powerWindow = 30
  const powerControls = $('.press-kit-power-window')
  if (powerControls) powerControls.hidden = !show.cyclingPower || !powerCharts.length
  const powerScale = $('.press-kit-power-scale')
  if (powerScale)
    powerScale.hidden =
      !show.powerCurve ||
      !activities.some(activity => activity.charts.some(chart => chart.key === 'powerCurve'))
  session.dialog
    .querySelectorAll<HTMLInputElement>('input[name="press-kit-power-scale"]')
    .forEach(input => {
      input.checked = input.value === session.powerCurveScale
    })
  session.dialog
    .querySelectorAll<HTMLInputElement>('input[name="press-kit-power-window"]')
    .forEach(input => {
      input.checked = Number(input.value) === session.powerWindow
      input.disabled = !availableWindow(Number(input.value))
    })
  session.dialog.querySelectorAll<HTMLButtonElement>('[data-day-activity]').forEach(button => {
    const selected = selectedActivities.has(button.dataset.dayActivity ?? '')
    button.setAttribute('aria-pressed', String(selected))
    button.title = selected ? 'masquer cette activité' : 'inclure cette activité'
  })
  session.dialog.querySelectorAll<HTMLButtonElement>('[data-day-block]').forEach(button => {
    const block = session.subject?.day?.blocks.find(block => block.id === button.dataset.dayBlock)
    button.disabled =
      block?.kind === 'comparison' &&
      block.activities.filter(activity => selectedActivities.has(activity.url)).length < 2
  })
  const entryDisplay = $('.press-kit-entry-display')
  if (entryDisplay) entryDisplay.hidden = !show.entry
  syncDayLayout(
    blocks.length + (show.entry && runsToPlain(session.subject?.excerpt ?? []).trim() ? 1 : 0),
  )
  session.dialog.querySelectorAll<HTMLButtonElement>('[data-activity-metric]').forEach(button => {
    const url = button.dataset.metricActivity
    const key = button.dataset.activityMetric
    button.setAttribute(
      'aria-pressed',
      String(Boolean(url && key && selectedMetrics.get(url)?.has(key))),
    )
  })
  session.dialog
    .querySelectorAll<HTMLElement>('[data-metric-group]')
    .forEach(
      group =>
        (group.hidden = !activities.some(activity => activity.url === group.dataset.metricGroup)),
    )
  const textarea = $<HTMLTextAreaElement>('#press-kit-quote')
  if (textarea && document.activeElement !== textarea)
    textarea.value =
      show.tractatus && session.quote.propositions?.length
        ? propositionsText(session.quote.propositions)
        : runsToText(session.quote.runs)
  const attribution = $<HTMLInputElement>('#press-kit-attribution')
  if (attribution) {
    attribution.placeholder = titleText()
    if (document.activeElement !== attribution) attribution.value = session.quote.attribution ?? ''
  }
  session.anchorSelect?.setSelected(session.anchor)
  setHelp()
}

// `blocks` counts what the card draws: the selected blocks and the entry when it is included.
function syncDayLayout(blocks: number) {
  const setting = $('.press-kit-day-layout')
  if (!setting) return
  setting.hidden = session.overlay !== 'card' || !offersDayLayout(blocks)
  const limit = STACK_LIMIT[session.format]
  const fits = limit >= blocks
  setting
    .querySelectorAll<HTMLInputElement>('input[name="press-kit-day-layout"]')
    .forEach(input => {
      input.checked =
        input.value ===
        (stacks(session.dayLayout, session.format, blocks) ? 'vertical' : 'horizontal')
      input.disabled = input.value === 'vertical' && !fits
    })
  const hint = setting.querySelector<HTMLElement>('.press-kit-hint')
  if (!hint) return
  const format = FORMATS[session.format].label
  hint.hidden = fits
  hint.textContent = limit
    ? `${format} empile ${limit} blocs au plus`
    : `${format} garde les blocs côte à côte`
}

export function refreshLocale() {
  if (!session.subject) return
  const locale = shareLocale()
  if (session.subject.activity)
    session.subject.activity = localizeActivity(session.subject.activity, locale)
  if (session.subject.day)
    session.subject.day.blocks = session.subject.day.blocks.map(block => ({
      ...block,
      rows: block.rows.map(row => ({ ...row, label: triLabel(locale, row.label) })),
      activities: block.activities.map(activity => localizeActivity(activity, locale)),
    }))
  if (session.subject.iso) session.subject.entryDate = cardDate(session.subject.iso, locale)
  if (session.subject.kind === 'entry' && session.subject.iso)
    session.subject.date = cardDate(session.subject.iso, locale)
  if (session.subject.kind === 'activity' && session.subject.activity)
    session.subject.date = cardDate(session.subject.activity.date, locale)
  if (session.subject.kind === 'day') {
    const date =
      session.subject.day?.blocks.flatMap(block => block.activities)[0]?.date ?? session.subject.iso
    if (date) session.subject.date = cardDate(date, locale)
  }
  mountMetricControls()
  syncControls()
  layoutMetricControls()
  if (session.dialog?.open) schedule()
}

export function setTab(next: PressKitTab, focus = false) {
  session.tab = availableTabs().includes(next) ? next : 'note'
  syncControls()
  layoutMetricControls()
  if (focus) $<HTMLButtonElement>(`#press-kit-tab-${session.tab}`)?.focus()
  schedule()
}

export function isIncludeKey(name: string): name is keyof CardModel['show'] {
  return Object.hasOwn(show, name)
}
