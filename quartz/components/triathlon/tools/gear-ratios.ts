import type { GearCassettePreset } from '../../../util/triathlon-gear-ratio'
import type { GearRatioMatrix } from '../../../util/triathlon-gear-ratio'
import type { TriathlonContext } from '../runtime/context'
import type { TriathlonFormatter } from '../runtime/formatter'
import { start } from '../../../functional'
import { CERAMICSPEED_CROSS_CHAIN_RESEARCH } from '../../../util/triathlon-gear-ratio'
import { CERAMICSPEED_TEST_CADENCE_RPM } from '../../../util/triathlon-gear-ratio'
import { CERAMICSPEED_TEST_CHAINSTAY_MM } from '../../../util/triathlon-gear-ratio'
import { CERAMICSPEED_TEST_OUTPUT_WATTS } from '../../../util/triathlon-gear-ratio'
import { formatGearEfficiencyDeltaPercent } from '../../../util/triathlon-gear-ratio'
import { gearCassettePreset } from '../../../util/triathlon-gear-ratio'
import { gearRatioMatrix } from '../../../util/triathlon-gear-ratio'
import { el } from '../runtime/dom'
import { mathFrag } from '../runtime/dom'
import { setMath } from '../runtime/dom'
import {
  initialGearRatioModel,
  updateGearRatio,
  type GearRatioEffect,
  type GearRatioMessage,
  type GearRatioModel,
} from './gear-ratio-model'

export const gearEfficiencyLevel = (crossChainLossWatts: number): string =>
  `${(8 + Math.min(crossChainLossWatts / 2.5, 1) * 32).toFixed(1)}%`

export const appendGearEfficiencyValues = (host: HTMLElement, efficiency: number): void => {
  const full = el('span', 'tri-ratio-efficiency-value tri-ratio-efficiency-value--full')
  const compact = el('span', 'tri-ratio-efficiency-value tri-ratio-efficiency-value--compact')
  setMath(full, `$${formatGearEfficiencyDeltaPercent(efficiency, 2)}\\%$`)
  setMath(compact, `$${formatGearEfficiencyDeltaPercent(efficiency, 1)}\\%$`)
  host.append(full, compact)
}

export const buildGearRatioPlaceholderRow = (
  cogs: GearRatioMatrix['rows'][number]['cells'],
  efficiency = false,
): HTMLTableRowElement => {
  const row = document.createElement('tr')
  row.className = efficiency
    ? 'tri-ratio-efficiency-row tri-ratio-row--placeholder'
    : 'tri-ratio-row tri-ratio-row--placeholder'
  row.setAttribute('aria-hidden', 'true')
  row.appendChild(el('th', undefined, '\u00a0'))
  for (const _ of cogs) row.appendChild(el('td', undefined, '\u00a0'))
  return row
}

export const buildGearRatioTable = (
  formatter: TriathlonFormatter,
  matrix: GearRatioMatrix,
): HTMLTableElement => {
  const table = document.createElement('table')
  table.className = 'tri-ratio-table'
  table.setAttribute('aria-label', formatter.text('gear ratios'))

  const caption = el('caption')
  const sourceLinks = el('span', 'tri-ratio-source-links', undefined, {
    'aria-label': 'CeramicSpeed research sources',
  })
  for (const source of CERAMICSPEED_CROSS_CHAIN_RESEARCH.sources) {
    sourceLinks.appendChild(
      el('a', 'tri-ratio-source-link', `[${source.id}]`, {
        href: source.url,
        target: '_blank',
        rel: 'noopener noreferrer',
        title: source.title,
        'aria-label': `source ${source.id}: ${source.title}, opens in new tab`,
      }),
    )
  }
  caption.append(
    ...mathFrag('$\\Delta\\eta$ est. vs. ideal · CeramicSpeed '),
    sourceLinks,
    document.createTextNode(
      ` · ${CERAMICSPEED_TEST_OUTPUT_WATTS} W · ${CERAMICSPEED_TEST_CADENCE_RPM} rpm · ${CERAMICSPEED_TEST_CHAINSTAY_MM} mm chainstay`,
    ),
  )

  const thead = document.createElement('thead')
  const header = document.createElement('tr')
  header.appendChild(
    el('th', undefined, 'T', { scope: 'col', 'aria-label': 'chainring and cassette teeth' }),
  )
  const cogs = matrix.rows[0]?.cells ?? []
  for (const cell of cogs)
    header.appendChild(el('th', undefined, String(cell.cog), { scope: 'col' }))
  thead.appendChild(header)

  const tbody = document.createElement('tbody')
  matrix.rows.forEach((row, rowIndex) => {
    const tr = el('tr', `tri-ratio-row tri-ratio-row--${rowIndex + 1}`)
    tr.appendChild(
      el('th', undefined, String(row.chainring), {
        scope: 'row',
        'aria-label': `${row.chainring} tooth chainring`,
      }),
    )
    for (const cell of row.cells) {
      const ratio = cell.ratio.toFixed(2)
      const td = el('td', undefined, ratio, {
        'data-ratio-chainring': String(row.chainring),
        'data-ratio-cog': String(cell.cog),
        'data-ratio-value': ratio,
        title: `${row.chainring}T ÷ ${cell.cog}T = ${ratio}`,
      })
      td.style.setProperty('--tri-ratio-level', `${(18 + cell.level * 52).toFixed(1)}%`)
      tr.appendChild(td)
    }
    const efficiencyRow = el('tr', `tri-ratio-efficiency-row tri-ratio-row--${rowIndex + 1}`)
    const efficiencyHeader = el('th', undefined, undefined, {
      scope: 'row',
      'aria-label': `estimated CeramicSpeed drivetrain efficiency difference from ideal for ${row.chainring} tooth chainring`,
    })
    setMath(efficiencyHeader, '$\\Delta\\eta$')
    efficiencyRow.appendChild(efficiencyHeader)
    for (const [cellIndex, cell] of row.cells.entries()) {
      const efficiency = cell.drivetrainEfficiency.toFixed(3)
      const efficiencyDelta = formatGearEfficiencyDeltaPercent(cell.drivetrainEfficiency, 3)
      const visibleEfficiencyDelta = formatGearEfficiencyDeltaPercent(cell.drivetrainEfficiency, 2)
      const drivetrainLossWatts = cell.drivetrainLossWatts.toFixed(2)
      const crossChainLossWatts = cell.crossChainLossWatts.toFixed(2)
      const td = el('td', undefined, undefined, {
        'data-efficiency-chainring': String(row.chainring),
        'data-efficiency-cog': String(cell.cog),
        'data-efficiency-value': efficiency,
        'data-efficiency-delta': String(efficiencyDelta),
        'data-loss-watts': drivetrainLossWatts,
        'data-cross-chain-loss-watts': crossChainLossWatts,
        'aria-label': `${visibleEfficiencyDelta}% estimated drivetrain efficiency difference from ideal; ${cell.drivetrainEfficiency.toFixed(2)}% estimated drivetrain efficiency; ${drivetrainLossWatts} watts drivetrain loss; ${crossChainLossWatts} watts cross-chain loss`,
        tabindex: rowIndex === 0 && cellIndex === 0 ? '0' : '-1',
      })
      appendGearEfficiencyValues(td, cell.drivetrainEfficiency)
      td.style.setProperty('--tri-efficiency-level', gearEfficiencyLevel(cell.crossChainLossWatts))
      efficiencyRow.appendChild(td)
    }
    tbody.append(tr, efficiencyRow)
  })
  if (matrix.rows.length === 1) {
    tbody.append(buildGearRatioPlaceholderRow(cogs), buildGearRatioPlaceholderRow(cogs, true))
  }
  table.append(caption, thead, tbody)
  return table
}

export const setupGearRatios = (
  root: HTMLElement,
  context: TriathlonContext,
): (() => void) | null => {
  const calculator = root.querySelector<HTMLElement>('.tri-ratio')
  const chart = calculator?.querySelector<HTMLElement>('.tri-ratio-chart')
  const range = calculator?.querySelector<HTMLOutputElement>('.tri-ratio-range')
  const cassettePicker = calculator?.querySelector<HTMLElement>('.tri-ratio-cassette-picker')
  const cassetteTrigger = calculator?.querySelector<HTMLButtonElement>(
    '.tri-ratio-cassette-trigger',
  )
  const cassetteMenu = calculator?.querySelector<HTMLElement>('.tri-ratio-cassette-menu')
  const cassetteValue = calculator?.querySelector<HTMLElement>('.tri-ratio-cassette-value')
  const cassetteOptions = Array.from(
    calculator?.querySelectorAll<HTMLButtonElement>('.tri-ratio-cassette-option') ?? [],
  )
  const ringInputs = calculator?.querySelectorAll<HTMLInputElement>('.tri-ratio-ring-input')
  const secondRing = calculator?.querySelector<HTMLElement>('[data-ratio-ring="2"]')
  const layoutButtons = calculator?.querySelectorAll<HTMLButtonElement>('.tri-ratio-layout-btn')
  const firstInput = ringInputs?.[0]
  const secondInput = ringInputs?.[1]
  const initialCassetteId = cassetteOptions.find(
    option => option.getAttribute('aria-selected') === 'true',
  )?.dataset.cassetteId
  if (
    !calculator ||
    !chart ||
    !range ||
    !cassettePicker ||
    !cassetteTrigger ||
    !cassetteMenu ||
    !cassetteValue ||
    !cassetteOptions.length ||
    !initialCassetteId ||
    !firstInput ||
    !secondInput ||
    !secondRing ||
    !layoutButtons?.length
  )
    return null

  document.body.querySelector('.tri-ratio-efficiency-tip')?.remove()
  const efficiencyTip = el('div', 'tri-gloss tri-ratio-efficiency-tip', undefined, {
    id: 'tri-ratio-efficiency-tip',
    role: 'tooltip',
    'aria-hidden': 'true',
  })
  document.body.appendChild(efficiencyTip)
  let activeEfficiencyCell: HTMLTableCellElement | null = null
  let hoveredEfficiencyCell: HTMLTableCellElement | null = null
  let focusedEfficiencyCell: HTMLTableCellElement | null = null
  let efficiencyTipDismissed = false

  const efficiencyCell = (target: EventTarget | null): HTMLTableCellElement | null =>
    target instanceof Element
      ? target.closest<HTMLTableCellElement>('td[data-efficiency-delta]')
      : null

  const placeEfficiencyTip = (): void => {
    if (!activeEfficiencyCell) return
    const target = activeEfficiencyCell.getBoundingClientRect()
    const tip = efficiencyTip.getBoundingClientRect()
    const edge = 8
    const gap = 8
    const centered = target.left + (target.width - tip.width) / 2
    const left = Math.min(Math.max(edge, centered), window.innerWidth - edge - tip.width)
    const above = target.top - gap - tip.height
    const below = Math.min(target.bottom + gap, window.innerHeight - edge - tip.height)
    const top = above >= edge ? above : below
    const maximumTop = Math.max(edge, window.innerHeight - edge - tip.height)
    efficiencyTip.style.left = `${left}px`
    efficiencyTip.style.top = `${Math.min(Math.max(edge, top), maximumTop)}px`
  }

  const hideEfficiencyTip = (): void => {
    activeEfficiencyCell?.removeAttribute('aria-describedby')
    activeEfficiencyCell = null
    efficiencyTip.classList.remove('tri-gloss--on')
    efficiencyTip.setAttribute('aria-hidden', 'true')
  }

  const showEfficiencyTip = (cell: HTMLTableCellElement): void => {
    const { efficiencyChainring, efficiencyCog, efficiencyValue, efficiencyDelta } = cell.dataset
    const lossWatts = cell.dataset.lossWatts
    const crossChainLossWatts = cell.dataset.crossChainLossWatts
    if (
      !efficiencyChainring ||
      !efficiencyCog ||
      !efficiencyValue ||
      !efficiencyDelta ||
      !lossWatts ||
      !crossChainLossWatts
    )
      return
    activeEfficiencyCell?.removeAttribute('aria-describedby')
    activeEfficiencyCell = cell
    const heading = el(
      'span',
      'tri-ratio-efficiency-tip-heading',
      `${efficiencyChainring}T × ${efficiencyCog}T`,
    )
    const row = (label: string, value: string, math = false): HTMLElement => {
      const item = el('span', 'tri-ratio-efficiency-tip-row')
      const labelElement = el('span', 'tri-ratio-efficiency-tip-label', label)
      const valueElement = el('span', 'tri-ratio-efficiency-tip-value')
      if (math) setMath(valueElement, `$${value}$`)
      else valueElement.textContent = value
      item.append(labelElement, valueElement)
      return item
    }
    efficiencyTip.replaceChildren(
      heading,
      row(context.formatter.text('vs. ideal'), `${efficiencyDelta}\\%`, true),
      row(context.formatter.text('efficiency'), `${efficiencyValue}\\%`, true),
      row(context.formatter.text('drivetrain loss'), `${lossWatts} W`),
      row(context.formatter.text('cross-chain loss'), `${crossChainLossWatts} W`),
    )
    cell.setAttribute('aria-describedby', efficiencyTip.id)
    efficiencyTip.classList.add('tri-gloss--on')
    efficiencyTip.setAttribute('aria-hidden', 'false')
    placeEfficiencyTip()
  }

  const syncEfficiencyTip = (): void => {
    if (efficiencyTipDismissed) {
      hideEfficiencyTip()
      return
    }
    const cell = hoveredEfficiencyCell ?? focusedEfficiencyCell
    if (cell) showEfficiencyTip(cell)
    else hideEfficiencyTip()
  }

  const onEfficiencyPointerOver = (event: PointerEvent): void => {
    const cell = efficiencyCell(event.target)
    if (!cell || cell === hoveredEfficiencyCell) return
    hoveredEfficiencyCell = cell
    efficiencyTipDismissed = false
    syncEfficiencyTip()
  }
  const onEfficiencyPointerOut = (event: PointerEvent): void => {
    const cell = efficiencyCell(event.target)
    if (!cell || efficiencyCell(event.relatedTarget) === cell) return
    if (hoveredEfficiencyCell === cell) hoveredEfficiencyCell = null
    syncEfficiencyTip()
  }
  const onEfficiencyFocusIn = (event: FocusEvent): void => {
    const cell = efficiencyCell(event.target)
    if (!cell) return
    focusedEfficiencyCell = cell
    efficiencyTipDismissed = false
    for (const candidate of chart.querySelectorAll<HTMLTableCellElement>(
      'td[data-efficiency-delta]',
    ))
      candidate.tabIndex = candidate === cell ? 0 : -1
    syncEfficiencyTip()
  }
  const onEfficiencyFocusOut = (event: FocusEvent): void => {
    const cell = efficiencyCell(event.target)
    if (cell && focusedEfficiencyCell === cell) focusedEfficiencyCell = null
    efficiencyTipDismissed = false
    syncEfficiencyTip()
  }
  const onEfficiencyKeydown = (event: KeyboardEvent): void => {
    if (event.key === 'Escape' && efficiencyTip.getAttribute('aria-hidden') === 'false') {
      event.preventDefault()
      event.stopPropagation()
      efficiencyTipDismissed = true
      hideEfficiencyTip()
      return
    }
    const cell = efficiencyCell(event.target)
    const row = cell?.parentElement
    if (!cell || !(row instanceof HTMLTableRowElement)) return
    const rows = Array.from(
      chart.querySelectorAll<HTMLTableRowElement>(
        '.tri-ratio-efficiency-row:not(.tri-ratio-row--placeholder)',
      ),
    )
    const cells = Array.from(
      row.querySelectorAll<HTMLTableCellElement>('td[data-efficiency-delta]'),
    )
    const rowIndex = rows.indexOf(row)
    const cellIndex = cells.indexOf(cell)
    let target: HTMLTableCellElement | null = null
    if (event.key === 'ArrowLeft') target = cells[cellIndex - 1] ?? null
    if (event.key === 'ArrowRight') target = cells[cellIndex + 1] ?? null
    if (event.key === 'Home') target = cells[0] ?? null
    if (event.key === 'End') target = cells.at(-1) ?? null
    if (event.key === 'ArrowUp' || event.key === 'ArrowDown') {
      const targetRow = rows[rowIndex + (event.key === 'ArrowUp' ? -1 : 1)]
      target =
        targetRow?.querySelectorAll<HTMLTableCellElement>('td[data-efficiency-delta]')[cellIndex] ??
        null
    }
    if (!target) return
    event.preventDefault()
    target.focus()
  }

  const selectedCassette = (): GearCassettePreset | null =>
    gearCassettePreset(program.retrieve().cassetteId)

  const readChainring = (input: HTMLInputElement, value: number): number | null => {
    const valid = Number.isInteger(value) && value >= 24 && value <= 64
    if (valid) input.removeAttribute('aria-invalid')
    else input.setAttribute('aria-invalid', 'true')
    return valid ? value : null
  }

  const render = (): void => {
    const model = program.retrieve()
    const preset = gearCassettePreset(model.cassetteId)
    if (!preset) return
    hoveredEfficiencyCell = null
    focusedEfficiencyCell = null
    efficiencyTipDismissed = false
    hideEfficiencyTip()
    cassetteValue.textContent = preset.label
    for (const candidate of cassetteOptions)
      candidate.setAttribute('aria-selected', String(candidate.dataset.cassetteId === preset.id))
    secondRing.classList.toggle('tri-ratio-ring--inactive', model.layout === 1)
    secondRing.setAttribute('aria-disabled', String(model.layout === 1))
    secondInput.disabled = model.layout === 1
    for (const button of layoutButtons) {
      const count = Number(button.dataset.ratioLayout)
      const active = count === model.layout
      button.classList.toggle('tri-ratio-layout-btn--on', active)
      button.setAttribute('aria-pressed', String(active))
      button.disabled = count === 2 && preset.maximumChainrings === 1
    }
    const first = readChainring(firstInput, model.chainrings[0])
    const second = model.layout === 2 ? readChainring(secondInput, model.chainrings[1]) : null
    if (first == null || (model.layout === 2 && second == null)) {
      range.textContent = '—'
      chart.setAttribute('aria-invalid', 'true')
      return
    }
    const chainrings = model.layout === 2 && second != null ? [first, second] : [first]
    const matrix = gearRatioMatrix(chainrings, preset.cogs)
    if (!matrix) return
    range.textContent = `${matrix.minimum.toFixed(2)}–${matrix.maximum.toFixed(2)}`
    chart.removeAttribute('aria-invalid')
    chart.replaceChildren(buildGearRatioTable(context.formatter, matrix))
  }

  const onLayout = (event: Event): void => {
    if (!(event.currentTarget instanceof HTMLButtonElement)) return
    const preset = selectedCassette()
    if (!preset) return
    const layout = Number(event.currentTarget.dataset.ratioLayout)
    if (layout === 1 || layout === 2)
      program.dispatch({ type: 'set-layout', layout, maximumChainrings: preset.maximumChainrings })
  }
  const onInput = (event: Event): void => {
    if (!(event.currentTarget instanceof HTMLInputElement)) return
    program.dispatch({
      type: 'set-chainring',
      index: event.currentTarget === firstInput ? 0 : 1,
      value: event.currentTarget.valueAsNumber,
    })
  }

  const closeCassetteMenu = (restoreFocus = false): void => {
    cassetteMenu.hidden = true
    cassetteTrigger.setAttribute('aria-expanded', 'false')
    if (restoreFocus) cassetteTrigger.focus()
  }
  const openCassetteMenu = (): void => {
    if (!cassetteMenu.hidden) return
    cassetteMenu.hidden = false
    cassetteTrigger.setAttribute('aria-expanded', 'true')
    cassetteOptions
      .find(option => option.dataset.cassetteId === program.retrieve().cassetteId)
      ?.focus()
  }
  const selectCassette = (option: HTMLButtonElement): void => {
    const nextId = option.dataset.cassetteId
    if (!nextId) return
    const preset = gearCassettePreset(nextId)
    if (!preset) return
    program.dispatch({
      type: 'set-cassette',
      cassetteId: nextId,
      maximumChainrings: preset.maximumChainrings,
    })
  }
  const onCassetteTriggerClick = (): void => {
    if (cassetteMenu.hidden) openCassetteMenu()
    else closeCassetteMenu()
  }
  const onCassetteTriggerKeydown = (event: KeyboardEvent): void => {
    if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return
    event.preventDefault()
    openCassetteMenu()
  }
  const onCassetteOptionClick = (event: Event): void => {
    if (!(event.currentTarget instanceof HTMLButtonElement)) return
    selectCassette(event.currentTarget)
    closeCassetteMenu(true)
  }
  const onCassetteMenuKeydown = (event: KeyboardEvent): void => {
    if (event.key === 'Escape') {
      event.preventDefault()
      event.stopPropagation()
      closeCassetteMenu(true)
      return
    }
    const activeIndex = cassetteOptions.findIndex(option => option === document.activeElement)
    const targetIndex =
      event.key === 'Home'
        ? 0
        : event.key === 'End'
          ? cassetteOptions.length - 1
          : event.key === 'ArrowDown'
            ? Math.min(cassetteOptions.length - 1, activeIndex + 1)
            : event.key === 'ArrowUp'
              ? Math.max(0, activeIndex - 1)
              : -1
    if (targetIndex < 0) return
    event.preventDefault()
    cassetteOptions[targetIndex]?.focus()
  }
  const onCassetteFocusout = (event: FocusEvent): void => {
    if (event.relatedTarget instanceof Node && cassettePicker.contains(event.relatedTarget)) return
    closeCassetteMenu()
  }
  const onCassetteOutsidePointerdown = (event: PointerEvent): void => {
    if (event.composedPath().includes(cassettePicker)) return
    closeCassetteMenu()
  }
  const onLocale = (): void => {
    render()
  }

  const program = start<GearRatioModel, GearRatioMessage, GearRatioEffect>({
    init: () => ({
      model: initialGearRatioModel(
        initialCassetteId,
        firstInput.valueAsNumber,
        secondInput.valueAsNumber,
      ),
      effects: [],
    }),
    reduce: updateGearRatio,
    effects: () => render(),
  })
  render()

  for (const button of layoutButtons) button.addEventListener('click', onLayout)
  for (const option of cassetteOptions) option.addEventListener('click', onCassetteOptionClick)
  firstInput.addEventListener('input', onInput)
  secondInput.addEventListener('input', onInput)
  cassetteTrigger.addEventListener('click', onCassetteTriggerClick)
  cassetteTrigger.addEventListener('keydown', onCassetteTriggerKeydown)
  cassetteMenu.addEventListener('keydown', onCassetteMenuKeydown)
  cassettePicker.addEventListener('focusout', onCassetteFocusout)
  document.addEventListener('pointerdown', onCassetteOutsidePointerdown)
  chart.addEventListener('pointerover', onEfficiencyPointerOver)
  chart.addEventListener('pointerout', onEfficiencyPointerOut)
  chart.addEventListener('focusin', onEfficiencyFocusIn)
  chart.addEventListener('focusout', onEfficiencyFocusOut)
  chart.addEventListener('keydown', onEfficiencyKeydown)
  window.addEventListener('resize', placeEfficiencyTip)
  window.addEventListener('scroll', placeEfficiencyTip, true)
  window.addEventListener('tri:locale', onLocale)
  return () => {
    for (const button of layoutButtons) button.removeEventListener('click', onLayout)
    for (const option of cassetteOptions) option.removeEventListener('click', onCassetteOptionClick)
    firstInput.removeEventListener('input', onInput)
    secondInput.removeEventListener('input', onInput)
    cassetteTrigger.removeEventListener('click', onCassetteTriggerClick)
    cassetteTrigger.removeEventListener('keydown', onCassetteTriggerKeydown)
    cassetteMenu.removeEventListener('keydown', onCassetteMenuKeydown)
    cassettePicker.removeEventListener('focusout', onCassetteFocusout)
    document.removeEventListener('pointerdown', onCassetteOutsidePointerdown)
    chart.removeEventListener('pointerover', onEfficiencyPointerOver)
    chart.removeEventListener('pointerout', onEfficiencyPointerOut)
    chart.removeEventListener('focusin', onEfficiencyFocusIn)
    chart.removeEventListener('focusout', onEfficiencyFocusOut)
    chart.removeEventListener('keydown', onEfficiencyKeydown)
    window.removeEventListener('resize', placeEfficiencyTip)
    window.removeEventListener('scroll', placeEfficiencyTip, true)
    window.removeEventListener('tri:locale', onLocale)
    program.stop()
    hideEfficiencyTip()
    efficiencyTip.remove()
  }
}
