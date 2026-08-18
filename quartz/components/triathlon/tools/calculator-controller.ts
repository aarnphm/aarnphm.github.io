import type { Analytics } from '../../../plugins/stores/analytics'
import type { Sport } from '../../../plugins/stores/strava'
import type { PaceLegSpec } from '../../../util/pace-features'
import type { CalcShare } from '../../../util/triathlon-calculator'
import type { ProjectedLeg } from '../../../util/triathlon-calculator'
import type { SportThresholdVel } from '../../../util/triathlon-calculator'
import type { TriathlonCalcInput } from '../../../util/triathlon-calculator'
import type { TriathlonCalcLeg } from '../../../util/triathlon-calculator'
import type { Vo2LabZones } from '../../../util/triathlon-calculator'
import type { ZoneBand } from '../../../util/triathlon-calculator'
import type { TriathlonContext } from '../runtime/context'
import { start } from '../../../functional'
import { CALC_ANCHOR_PREFIX } from '../../../util/triathlon-calculator'
import { computeTriathlonCalcTimes } from '../../../util/triathlon-calculator'
import { deriveZoneBands } from '../../../util/triathlon-calculator'
import { encodeCalcShare } from '../../../util/triathlon-calculator'
import { formatDurationClock } from '../../../util/triathlon-calculator'
import { parseClockSeconds } from '../../../util/triathlon-calculator'
import { projectZoneTimes } from '../../../util/triathlon-calculator'
import { resolveTriathlonCalcPace } from '../../../util/triathlon-calculator'
import { solveTriathlonCalcLeg } from '../../../util/triathlon-calculator'
import { solveTriathlonCalcTarget } from '../../../util/triathlon-calculator'
import { clock } from '../../../util/triathlon-card'
import { KM_TO_MI } from '../../../util/triathlon-card'
import { isRecord } from '../../../util/type-guards'
import { bySport } from '../analytics/shared'
import { normCdf } from '../analytics/shared'
import { el } from '../runtime/dom'
import { setMath } from '../runtime/dom'
import { toggleTriUnit } from '../runtime/preferences'
import { wireEmbedCopy } from '../shell/timeline'
import {
  initialCalculatorModel,
  updateCalculator,
  type CalculatorEffect,
  type CalculatorMessage,
  type CalculatorModel,
} from './calculator-model'

const CALCULATOR_SPORTS: readonly Sport[] = ['swim', 'bike', 'run']

const isCalcShare = (value: unknown): value is CalcShare =>
  isRecord(value) &&
  Number.isInteger(value.presetIdx) &&
  (value.mode === 'a' || value.mode === 'p') &&
  (value.unit === 'i' || value.unit === 'm') &&
  typeof value.swimPaceSec === 'number' &&
  typeof value.t1Sec === 'number' &&
  typeof value.bikeMph === 'number' &&
  typeof value.t2Sec === 'number' &&
  typeof value.runPaceSec === 'number'

export const setupCalc = (root: HTMLElement, context: TriathlonContext): (() => void) | null => {
  const isImperial = (): boolean => context.presentation.distance === 'imperial'
  const btn = root.querySelector<HTMLElement>('.tri-calc-btn')
  const calc = root.querySelector<HTMLElement>('.tri-calc')
  const closeBtn = root.querySelector<HTMLElement>('.tri-calc-close')
  const pageMode = root.dataset.triView === 'calc'
  if (!calc || (!btn && !pageMode)) return null

  let live = true
  const inputVal = (k: string): string =>
    calc.querySelector<HTMLInputElement>(`.tri-calc-in[data-k="${k}"]`)?.value ?? ''
  const setInputVal = (k: string, value: string): void => {
    const input = calc.querySelector<HTMLInputElement>(`.tri-calc-in[data-k="${k}"]`)
    if (input) input.value = value
  }
  const targetInput = (): HTMLInputElement | null =>
    calc.querySelector<HTMLInputElement>('.tri-calc-target')
  const setResult = (leg: string, sec: number, forceTarget = false): void => {
    if (leg === 'total') {
      const target = targetInput()
      if (target && (forceTarget || document.activeElement !== target)) {
        target.value = formatDurationClock(sec)
      }
      return
    }
    const legInput = calc.querySelector<HTMLInputElement>(
      `.tri-calc-legtime[data-legtime="${leg}"]`,
    )
    if (legInput) {
      if (document.activeElement !== legInput) legInput.value = formatDurationClock(sec)
      return
    }
    const e = calc.querySelector<HTMLElement>(`.tri-calc-r[data-leg="${leg}"]`)
    if (e) e.textContent = formatDurationClock(sec)
  }

  const readDomCalcInput = (): TriathlonCalcInput => ({
    swimKm: Number(calc.dataset.swim) || 0,
    bikeKm: Number(calc.dataset.bike) || 0,
    runKm: Number(calc.dataset.run) || 0,
    swimPaceSec: parseClockSeconds(inputVal('swim')),
    t1Sec: parseClockSeconds(inputVal('t1')),
    bikeMph: (Number(inputVal('bike')) || 0) * (isImperial() ? 1 : KM_TO_MI),
    t2Sec: parseClockSeconds(inputVal('t2')),
    runPaceSec: parseClockSeconds(inputVal('run')) / (isImperial() ? 1 : KM_TO_MI),
  })

  const readCalcInput = (): TriathlonCalcInput => program.retrieve().input
  const renderComputation = (forceTarget = false): void => {
    const times = computeTriathlonCalcTimes(readCalcInput())
    setResult('swim', times.swimSec)
    setResult('t1', times.t1Sec)
    setResult('bike', times.bikeSec)
    setResult('t2', times.t2Sec)
    setResult('run', times.runSec)
    setResult('total', times.totalSec, forceTarget)
    if (program.retrieve().projection.active) renderProjection()
  }
  const compute = (forceTarget = false, userEdited?: boolean): void =>
    program.dispatch({ type: 'sync-input', input: readDomCalcInput(), forceTarget, userEdited })

  const commitTarget = (): void => {
    const input = targetInput()
    if (!input) return
    const paces = solveTriathlonCalcTarget(readCalcInput(), parseClockSeconds(input.value))
    if (!paces) {
      compute(true, true)
      return
    }
    setInputVal('swim', clock(paces.swimPaceSec))
    setInputVal('bike', bikeToDisp(paces.bikeMph))
    setInputVal('run', runToDisp(paces.runPaceSec))
    compute(true, true)
  }

  const commitLeg = (leg: TriathlonCalcLeg): void => {
    const input = calc.querySelector<HTMLInputElement>(`.tri-calc-legtime[data-legtime="${leg}"]`)
    if (!input) return
    const solved = solveTriathlonCalcLeg(readCalcInput(), leg, parseClockSeconds(input.value))
    if (!solved) {
      compute(true, true)
      return
    }
    if (solved.swimPaceSec != null) setInputVal('swim', clock(solved.swimPaceSec))
    if (solved.bikeMph != null) setInputVal('bike', bikeToDisp(solved.bikeMph))
    if (solved.runPaceSec != null) setInputVal('run', runToDisp(solved.runPaceSec))
    compute(true, true)
  }

  let analytics: Analytics | null = null
  const source = calc.querySelector<HTMLElement>('.tri-calc-source')
  const projPanel = calc.querySelector<HTMLElement>('.tri-calc-proj')
  const projZonesWrap = projPanel?.querySelector<HTMLElement>('.tri-calc-proj-zones') ?? null
  const projOut = projPanel?.querySelector<HTMLElement>('.tri-calc-proj-out') ?? null
  const projTab = calc.querySelector<HTMLElement>('.tri-calc-src--proj')
  let projBands: ZoneBand[] = []
  const paceHuman = (which: 'avg' | 'pred', sport: Sport): number | null =>
    analytics ? resolveTriathlonCalcPace(analytics, which, sport) : null
  const toCalcInput = (sport: Sport, v: number): string =>
    sport === 'bike'
      ? (isImperial() ? v * KM_TO_MI : v).toFixed(1)
      : clock(sport === 'run' && isImperial() ? v / KM_TO_MI : v)
  const bikeToDisp = (mph: number): string => (isImperial() ? mph : mph / KM_TO_MI).toFixed(1)
  const runToDisp = (miSec: number): string => clock(isImperial() ? miSec : miSec * KM_TO_MI)
  const applySourceInputs = (which: 'avg' | 'pred'): void => {
    let any = false
    for (const sport of CALCULATOR_SPORTS) {
      const v = paceHuman(which, sport)
      if (v == null || !Number.isFinite(v) || v <= 0) continue
      setInputVal(sport, toCalcInput(sport, v))
      any = true
    }
    if (!any) return
    compute(false, false)
  }

  const readThresholds = (): SportThresholdVel | null => {
    if (!analytics) return null
    const get = (s: Sport): number => bySport(analytics!.thresholds, s)?.vThr ?? 0
    return { swim: get('swim'), bike: get('bike'), run: get('run') }
  }
  const latestLab = (): Vo2LabZones | null => {
    const labs = analytics?.tests?.vo2max
    const r = labs && labs.length ? labs[labs.length - 1] : null
    return r ? { zonesKmh: r.zonesKmh, zonesHr: r.zonesHr, maxKmh: r.maxKmh, hrMax: r.hrMax } : null
  }
  const zoneHrLabel = (band: ZoneBand): string =>
    band.hrLo != null ? `${band.hrLo}${band.hrHi != null ? `–${band.hrHi}` : '+'}` : ''
  const legBand = (sport: Sport, leg: ProjectedLeg): { pace: string; split: string } => {
    const split = `${formatDurationClock(leg.fastSec)}–${formatDurationClock(leg.slowSec)}`
    if (sport === 'swim')
      return { pace: `${clock(360 / leg.vMaxKmh)}–${clock(360 / leg.vMinKmh)}`, split }
    if (sport === 'bike') {
      const lo = isImperial() ? leg.vMinKmh * KM_TO_MI : leg.vMinKmh
      const hi = isImperial() ? leg.vMaxKmh * KM_TO_MI : leg.vMaxKmh
      return { pace: `${lo.toFixed(1)}–${hi.toFixed(1)}`, split }
    }
    const fast = 3600 / leg.vMaxKmh
    const slow = 3600 / leg.vMinKmh
    const conv = (s: number): number => (isImperial() ? s / KM_TO_MI : s)
    return { pace: `${clock(conv(fast))}–${clock(conv(slow))}`, split }
  }
  const projUnit = (sport: Sport): string =>
    sport === 'swim'
      ? '/100m'
      : sport === 'bike'
        ? isImperial()
          ? 'mph'
          : 'km/h'
        : isImperial()
          ? '/mi'
          : '/km'
  const buildZoneSelector = (): void => {
    if (!projZonesWrap) return
    projZonesWrap.replaceChildren()
    for (const band of projBands) {
      const on = band.index === program.retrieve().projection.zone
      const btn = el('button', `tri-calc-zone${on ? ' tri-calc-zone--on' : ''}`, `Z${band.index}`, {
        type: 'button',
        role: 'tab',
        'data-zone': String(band.index),
        'aria-selected': String(on),
      })
      const hr = zoneHrLabel(band)
      btn.title = hr
        ? `${context.formatter.text(band.key)} · ${hr} bpm`
        : context.formatter.text(band.key)
      projZonesWrap.appendChild(btn)
    }
  }
  const syncZoneSelection = (): void => {
    for (const button of projZonesWrap?.querySelectorAll<HTMLElement>('.tri-calc-zone') ?? []) {
      const active = Number(button.dataset.zone) === program.retrieve().projection.zone
      button.classList.toggle('tri-calc-zone--on', active)
      button.setAttribute('aria-selected', String(active))
    }
  }
  const renderProjection = (): void => {
    if (!projOut) return
    const band =
      projBands.find(candidate => candidate.index === program.retrieve().projection.zone) ??
      projBands[0]
    const thr = readThresholds()
    const input = readCalcInput()
    const proj = band && thr ? projectZoneTimes(input, band, thr) : null
    if (!band || !proj) {
      projOut.replaceChildren(
        el('div', 'tri-ana-empty', context.formatter.text('no vo2 test logged')),
      )
      return
    }
    const ifPct = `${Math.round(proj.ifMin * 100)}–${Math.round(proj.ifMax * 100)}%`
    const frag = document.createDocumentFragment()

    const cap = el('div', 'tri-calc-proj-cap')
    cap.appendChild(el('span', 'tri-calc-proj-cap-z', context.formatter.text(band.key)))
    const hr = zoneHrLabel(band)
    if (hr) cap.appendChild(el('span', 'tri-calc-proj-cap-k', `HR ${hr}`))
    cap.appendChild(
      el('span', 'tri-calc-proj-cap-k', `${ifPct} ${context.formatter.text('threshold')}`),
    )
    frag.appendChild(cap)

    const table = el('table', 'tri-calc-proj-io')
    const tbody = el('tbody')
    const legs: [Sport, string, ProjectedLeg][] = [
      ['swim', context.formatter.text('swim'), proj.swim],
      ['bike', context.formatter.text('bike'), proj.bike],
      ['run', context.formatter.text('run'), proj.run],
    ]
    for (const [sport, label, leg] of legs) {
      const b = legBand(sport, leg)
      const tr = el('tr', 'tri-calc-proj-row')
      tr.append(
        el('th', 'tri-calc-proj-k', label),
        el('td', 'tri-calc-proj-pace', b.pace),
        el('td', 'tri-calc-proj-u', projUnit(sport)),
        el('td', 'tri-calc-proj-split', b.split),
      )
      tbody.appendChild(tr)
    }
    const finishTr = el('tr', 'tri-calc-proj-row tri-calc-proj-finish')
    finishTr.append(
      el('th', 'tri-calc-proj-k', context.formatter.text('finish')),
      el('td', 'tri-calc-proj-pace', ifPct),
      el('td', 'tri-calc-proj-u', ''),
      el(
        'td',
        'tri-calc-proj-split',
        `${formatDurationClock(proj.fastSec)}–${formatDurationClock(proj.slowSec)}`,
      ),
    )
    tbody.appendChild(finishTr)
    table.appendChild(tbody)
    frag.appendChild(table)

    const currentSec = computeTriathlonCalcTimes(input).totalSec
    const deltaSec = (proj.fastSec + proj.slowSec) / 2 - currentSec
    const delta = el('div', 'tri-calc-proj-delta')
    const deltaD = el(
      'span',
      `tri-calc-proj-delta-d${deltaSec < 0 ? ' tri-calc-proj-delta-d--fast' : ''}`,
    )
    setMath(
      deltaD,
      `$\\Delta$ $${deltaSec >= 0 ? '+' : '-'}$${formatDurationClock(Math.abs(deltaSec))}`,
    )
    delta.append(
      el('span', 'tri-calc-proj-delta-k', context.formatter.text('vs current')),
      el('span', 'tri-calc-proj-delta-v', formatDurationClock(currentSec)),
      deltaD,
    )
    frag.appendChild(delta)

    const projectionModel = program.retrieve().projection
    if (projectionModel.distribution && projectionModel.key === raceKey(input)) {
      const p =
        normCdf(
          (proj.slowSec - projectionModel.distribution.mu) / projectionModel.distribution.sigma,
        ) -
        normCdf(
          (proj.fastSec - projectionModel.distribution.mu) / projectionModel.distribution.sigma,
        )
      const likely = el('div', 'tri-calc-proj-likely', undefined, {
        title: 'model probability the actual finish lands in this projected range',
      })
      likely.append(
        el('span', 'tri-calc-proj-likely-k', 'model'),
        el('span', 'tri-calc-proj-likely-v', `${Math.round(Math.max(0, Math.min(1, p)) * 100)}%`),
      )
      frag.appendChild(likely)
    } else {
      void ensureProjModel(input)
    }

    projOut.replaceChildren(frag)
  }
  const raceKey = (i: TriathlonCalcInput): string => `${i.swimKm}-${i.bikeKm}-${i.runKm}`
  const ensureProjModel = async (i: TriathlonCalcInput): Promise<void> => {
    const key = raceKey(i)
    const projection = program.retrieve().projection
    if ((key === projection.key && projection.distribution) || key === projection.pendingKey) return
    const f = context.pace.forecaster
    if (!f?.ready) return
    program.dispatch({ type: 'projection-requested', key })
    const legs: PaceLegSpec[] = [
      { sport: 'swim', distanceKm: i.swimKm, elevationM: 0, tempC: null, windKph: null },
      { sport: 'bike', distanceKm: i.bikeKm, elevationM: 0, tempC: null, windKph: null },
      { sport: 'run', distanceKm: i.runKm, elevationM: 0, tempC: null, windKph: null },
    ]
    const fin = await f.forecastFinish(legs, i.t1Sec + i.t2Sec)
    if (!live || context.signal.aborted) return
    if (!fin || fin.slowSec <= fin.fastSec) {
      program.dispatch({ type: 'projection-failed', key })
      return
    }
    program.dispatch({
      type: 'projection-loaded',
      key,
      mu: fin.midSec,
      sigma: (fin.slowSec - fin.fastSec) / (2 * 1.2816),
    })
  }
  const pinModalTop = (): void => {
    if (pageMode) return
    const top = Math.round(calc.getBoundingClientRect().top)
    calc.style.transition = 'none'
    calc.style.top = `${top}px`
    calc.style.transform = 'translateX(-50%) scale(1)'
    void calc.offsetHeight
    calc.style.transition = ''
  }
  const unpinModalTop = (): void => {
    if (pageMode) return
    calc.style.top = ''
    calc.style.transform = ''
    calc.style.transition = ''
  }
  const syncSource = (): void => {
    const model = program.retrieve()
    for (const b of calc.querySelectorAll<HTMLElement>('.tri-calc-src')) {
      const source = b.dataset.src === 'proj' ? 'projection' : b.dataset.src
      const on = source === model.source
      b.classList.toggle('tri-calc-src--on', on)
      b.setAttribute('aria-selected', String(on))
    }
    if (!projPanel) return
    if (model.projection.active && projPanel.hidden) pinModalTop()
    projPanel.hidden = !model.projection.active
  }

  const open = () => {
    unpinModalTop()
    root.classList.add('tri-calc-open')
    calc.setAttribute('aria-hidden', 'false')
    compute()
    requestAnimationFrame(() => {
      if (root.classList.contains('tri-calc-open')) calc.focus({ preventScroll: true })
    })
  }
  const close = () => {
    const wasOpen = root.classList.contains('tri-calc-open')
    root.classList.remove('tri-calc-open')
    calc.setAttribute('aria-hidden', 'true')
    if (wasOpen && !pageMode) btn?.focus({ preventScroll: true })
  }
  const onCalcClick = (event: MouseEvent) => {
    const target = event.target
    const targetElement = target instanceof HTMLElement ? target : null
    const src = targetElement?.closest<HTMLElement>('.tri-calc-src')
    if (src?.dataset.src === 'avg' || src?.dataset.src === 'pred') {
      program.dispatch({ type: 'select-source', source: src.dataset.src })
      return
    }
    if (src?.dataset.src === 'proj') {
      program.dispatch({ type: 'select-source', source: 'projection' })
      return
    }
    const zoneBtn = targetElement?.closest<HTMLElement>('.tri-calc-zone')
    if (zoneBtn?.dataset.zone) {
      program.dispatch({ type: 'select-zone', zone: Number(zoneBtn.dataset.zone) })
      return
    }
    const p = targetElement?.closest<HTMLElement>('.tri-calc-preset')
    if (!p) return
    calc.dataset.swim = p.dataset.swim ?? ''
    calc.dataset.bike = p.dataset.bike ?? ''
    calc.dataset.run = p.dataset.run ?? ''
    for (const x of calc.querySelectorAll('.tri-calc-preset'))
      x.classList.toggle('tri-calc-preset--on', x === p)
    compute(false, true)
  }
  const onInput = (event: Event) => {
    const target = event.target
    if (!(target instanceof HTMLInputElement) || !target.classList.contains('tri-calc-in')) return
    if (
      target.classList.contains('tri-calc-target') ||
      target.classList.contains('tri-calc-legtime')
    )
      return
    compute(false, true)
  }
  const onChange = (event: Event) => {
    const target = event.target
    if (!(target instanceof HTMLInputElement)) return
    if (target.classList.contains('tri-calc-target')) {
      commitTarget()
    } else if (target.classList.contains('tri-calc-legtime')) {
      const leg = target.dataset.legtime
      if (leg === 'swim' || leg === 'bike' || leg === 'run') commitLeg(leg)
    }
  }
  const onCalcKey = (event: KeyboardEvent) => {
    const target = event.target
    if (!(target instanceof HTMLInputElement) || event.key !== 'Enter') return
    if (target.classList.contains('tri-calc-target')) {
      event.preventDefault()
      commitTarget()
      target.blur()
    } else if (target.classList.contains('tri-calc-legtime')) {
      event.preventDefault()
      const leg = target.dataset.legtime
      if (leg === 'swim' || leg === 'bike' || leg === 'run') commitLeg(leg)
      target.blur()
    }
  }
  const onKey = (event: KeyboardEvent) => {
    if (!pageMode && event.key === 'Escape') close()
  }

  const program = start<CalculatorModel, CalculatorMessage, CalculatorEffect>({
    init: () => ({ model: initialCalculatorModel(readDomCalcInput()), effects: [] }),
    reduce: updateCalculator,
    effects: effect => {
      if (effect.type === 'compute') {
        renderComputation(effect.forceTarget)
        return
      }
      if (effect.type === 'apply-source') {
        applySourceInputs(effect.source)
        return
      }
      if (effect.type === 'sync-source') {
        syncSource()
        return
      }
      syncSource()
      syncZoneSelection()
      renderProjection()
    },
  })

  if (pageMode) {
    compute()
  } else {
    btn?.addEventListener('click', open)
    closeBtn?.addEventListener('click', close)
  }
  calc.addEventListener('click', onCalcClick)
  calc.addEventListener('input', onInput)
  calc.addEventListener('change', onChange)
  calc.addEventListener('keydown', onCalcKey)
  document.addEventListener('keydown', onKey)
  const bikeUnitCell = calc.querySelector<HTMLElement>('.tri-calc-u[data-u="bike"]')
  const runUnitCell = calc.querySelector<HTMLElement>('.tri-calc-u[data-u="run"]')
  const syncUnitLabels = () => {
    if (bikeUnitCell) bikeUnitCell.textContent = isImperial() ? 'mph' : 'km/h'
    if (runUnitCell) runUnitCell.textContent = isImperial() ? '/mi' : '/km'
  }
  const onUnit = () => {
    const bikeRaw = Number(inputVal('bike')) || 0
    if (bikeRaw > 0)
      setInputVal('bike', (isImperial() ? bikeRaw * KM_TO_MI : bikeRaw / KM_TO_MI).toFixed(1))
    const runRaw = parseClockSeconds(inputVal('run'))
    if (runRaw > 0) setInputVal('run', clock(isImperial() ? runRaw / KM_TO_MI : runRaw * KM_TO_MI))
    syncUnitLabels()
    compute()
  }
  window.addEventListener('tri:unit', onUnit)
  syncUnitLabels()
  if (!isImperial()) onUnit()

  const apath = root.dataset.analyticsPath
  if (apath)
    void context.resources.analytics.load(apath).then(result => {
      if (live && result.status === 'ready') {
        analytics = result.value
        const usable = CALCULATOR_SPORTS.some(
          s => paceHuman('avg', s) != null || paceHuman('pred', s) != null,
        )
        if (source && usable) source.hidden = false
        if (usable && !program.retrieve().userEdited)
          program.dispatch({ type: 'select-source', source: 'avg' })
        const lab = latestLab()
        projBands = lab ? deriveZoneBands(lab) : []
        if (projBands.length) {
          if (!projBands.some(b => b.index === program.retrieve().projection.zone))
            program.dispatch({ type: 'select-zone', zone: projBands[0].index })
          buildZoneSelector()
          if (projTab) projTab.hidden = false
          if (source) source.hidden = false
        }
      }
    })

  const copyBtn = calc.querySelector<HTMLElement>('.tri-calc-copy')
  const currentShare = (): CalcShare => {
    const presets = Array.from(calc.querySelectorAll('.tri-calc-preset'))
    const idx = presets.findIndex(p => p.classList.contains('tri-calc-preset--on'))
    const mode: 'a' | 'p' =
      calc.querySelector('.tri-calc-src--on')?.getAttribute('data-src') === 'pred' ? 'p' : 'a'
    const ci = readCalcInput()
    return {
      presetIdx: idx >= 0 ? idx : 1,
      mode,
      unit: isImperial() ? 'i' : 'm',
      swimPaceSec: ci.swimPaceSec,
      t1Sec: ci.t1Sec,
      bikeMph: ci.bikeMph,
      t2Sec: ci.t2Sec,
      runPaceSec: ci.runPaceSec,
    }
  }
  const copyCleanup = wireEmbedCopy(
    context.formatter,
    copyBtn,
    () => `![[triathlon#${CALC_ANCHOR_PREFIX}${encodeCalcShare(currentShare())}]]`,
  )

  const onCalcFill = (event: Event): void => {
    if (!(event instanceof CustomEvent) || !isRecord(event.detail)) return
    const share = event.detail.share
    if (!isCalcShare(share)) return
    if ((share.unit === 'i') !== isImperial()) toggleTriUnit(context.preferences)
    const presets = Array.from(calc.querySelectorAll<HTMLElement>('.tri-calc-preset'))
    const preset = presets[share.presetIdx]
    if (preset) {
      calc.dataset.swim = preset.dataset.swim ?? ''
      calc.dataset.bike = preset.dataset.bike ?? ''
      calc.dataset.run = preset.dataset.run ?? ''
      for (const p of presets) p.classList.toggle('tri-calc-preset--on', p === preset)
    }
    const srcKey = share.mode === 'p' ? 'pred' : 'avg'
    for (const b of calc.querySelectorAll<HTMLElement>('.tri-calc-src')) {
      const on = b.dataset.src === srcKey
      b.classList.toggle('tri-calc-src--on', on)
      b.setAttribute('aria-selected', String(on))
    }
    setInputVal('swim', clock(share.swimPaceSec))
    setInputVal('t1', clock(share.t1Sec))
    setInputVal('bike', bikeToDisp(share.bikeMph))
    setInputVal('t2', clock(share.t2Sec))
    setInputVal('run', runToDisp(share.runPaceSec))
    compute(true, true)
    open()
  }
  window.addEventListener('tri:calc-fill', onCalcFill)

  return () => {
    live = false
    btn?.removeEventListener('click', open)
    closeBtn?.removeEventListener('click', close)
    calc.removeEventListener('click', onCalcClick)
    calc.removeEventListener('input', onInput)
    calc.removeEventListener('change', onChange)
    calc.removeEventListener('keydown', onCalcKey)
    document.removeEventListener('keydown', onKey)
    window.removeEventListener('tri:unit', onUnit)
    copyCleanup()
    window.removeEventListener('tri:calc-fill', onCalcFill)
    program.stop()
  }
}
