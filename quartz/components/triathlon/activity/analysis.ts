import type { ActivityKind } from '../../../plugins/stores/strava'
import type { StravaActivityDetail } from '../../../plugins/stores/strava'
import type { ActivitySelectionSummary } from '../../../util/triathlon-card'
import type { TriathlonPresentation } from '../../../util/triathlon-presentation'
import { activitySelectionSummary } from '../../../util/triathlon-card'
import { clock } from '../../../util/triathlon-card'
import { formatAltitude } from '../../../util/triathlon-card'
import { formatRespirationRate } from '../../../util/triathlon-card'
import { formatTemperature } from '../../../util/triathlon-card'
import { KM_TO_MI } from '../../../util/triathlon-card'
import { scrubDist } from '../../../util/triathlon-card'
import { speedKph } from '../../../util/triathlon-card'
import { triText } from '../../../util/triathlon-i18n'
import { isRecord } from '../../../util/type-guards'
import { el } from '../runtime/dom'

export type ScrubSurface = {
  wrap: HTMLElement
  samples: readonly ActivityScrubSample[]
  fmt: (index: number) => string
}

export type ActivityScrubSample = { d: number; elapsedS: number }

export const activityScrubIndexAt = (
  samples: readonly ActivityScrubSample[],
  target: number,
): number => {
  if (samples.length === 0) return -1
  let low = 0
  let high = samples.length
  while (low < high) {
    const middle = low + Math.floor((high - low) / 2)
    if (samples[middle].d < target) low = middle + 1
    else high = middle
  }
  if (low === 0) return 0
  if (low >= samples.length) return samples.length - 1
  return target - samples[low - 1].d <= samples[low].d - target ? low - 1 : low
}

export const activityScrubElapsedIndexAt = (
  samples: readonly ActivityScrubSample[],
  elapsedS: number,
): number => {
  if (samples.length === 0) return -1
  let low = 0
  let high = samples.length
  while (low < high) {
    const middle = low + Math.floor((high - low) / 2)
    if (samples[middle].elapsedS < elapsedS) low = middle + 1
    else high = middle
  }
  if (low === 0) return 0
  if (low >= samples.length) return samples.length - 1
  return elapsedS - samples[low - 1].elapsedS <= samples[low].elapsedS - elapsedS ? low - 1 : low
}

export type ActivityAnalysisRange = ActivitySelectionSummary & {
  button: HTMLButtonElement | null
  kind: 'lap' | 'segment' | 'climb' | null
  id: string | null
  label: string
}

export type PresetActivityAnalysisRange = ActivityAnalysisRange & {
  button: HTMLButtonElement
  kind: 'lap' | 'segment' | 'climb'
  id: string
}

export const analysisFinite = (value: string | undefined): number | null => {
  if (value == null || value === '') return null
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : null
}

export const analysisRouteIndex = (
  route: StravaActivityDetail['route'],
  targetDistanceKm: number,
): number => {
  let low = 0
  let high = route.length
  while (low < high) {
    const middle = low + Math.floor((high - low) / 2)
    if (route[middle].d < targetDistanceKm) low = middle + 1
    else high = middle
  }
  if (low === 0) return 0
  if (low >= route.length) return route.length - 1
  return targetDistanceKm - route[low - 1].d <= route[low].d - targetDistanceKm ? low - 1 : low
}

export const analysisRate = (
  presentation: TriathlonPresentation,
  sport: ActivityKind,
  valueKph: number,
): string => {
  if (valueKph <= 0) return '—'
  if (sport === 'bike') return speedKph(presentation, valueKph)
  if (sport === 'swim') return `${clock(360 / valueKph)} /100m`
  const imperial = presentation.distance === 'imperial'
  const distanceScale = imperial ? KM_TO_MI : 1
  return `${clock(3600 / (valueKph * distanceScale))} /${imperial ? 'mi' : 'km'}`
}

export const analysisRangeFromButton = (
  button: HTMLButtonElement,
): PresetActivityAnalysisRange | null => {
  const kind = button.dataset.rangeKind
  const id = button.dataset.rangeId
  const startElapsedS = analysisFinite(button.dataset.startElapsedS)
  const endElapsedS = analysisFinite(button.dataset.endElapsedS)
  const startDistanceKm = analysisFinite(button.dataset.startDistanceKm)
  const endDistanceKm = analysisFinite(button.dataset.endDistanceKm)
  if (
    (kind !== 'lap' && kind !== 'segment' && kind !== 'climb') ||
    id == null ||
    startElapsedS == null ||
    endElapsedS == null ||
    startDistanceKm == null ||
    endDistanceKm == null
  )
    return null
  return {
    button,
    kind,
    id,
    startElapsedS,
    endElapsedS,
    label:
      button.dataset.rangeLabel ??
      button.querySelector<HTMLElement>('.tri-analysis-range-title')?.textContent?.trim() ??
      button.textContent?.trim() ??
      id,
    startDistanceKm,
    endDistanceKm,
    distanceKm: analysisFinite(button.dataset.distanceKm) ?? endDistanceKm - startDistanceKm,
    durationS: analysisFinite(button.dataset.durationS) ?? 0,
    elevationGainM: analysisFinite(button.dataset.elevationGainM),
    averageSpeedKph: analysisFinite(button.dataset.averageSpeedKph),
    averageHeartRate: analysisFinite(button.dataset.averageHeartRate),
    averageWatts: analysisFinite(button.dataset.averageWatts),
    averageCadence: analysisFinite(button.dataset.averageCadence),
    averageRespirationRate: null,
    averageTemperatureC: null,
  }
}

export const analysisRoutePath = (
  route: StravaActivityDetail['route'],
  range: ActivityAnalysisRange,
): string => {
  const start = analysisRouteIndex(route, range.startDistanceKm)
  const end = analysisRouteIndex(route, range.endDistanceKm)
  const points = route.slice(Math.min(start, end), Math.max(start, end) + 1)
  const pad = 6
  const span = 88
  return points
    .map(
      (point, index) =>
        `${index === 0 ? 'M' : 'L'} ${(pad + point.x * span).toFixed(2)} ${(pad + (1 - point.y) * span).toFixed(2)}`,
    )
    .join(' ')
}

export type ActivityAnalysisController = {
  preview: (range: ActivityAnalysisRange) => void
  restore: () => void
  lock: (range: ActivityAnalysisRange) => void
  clear: () => void
  hasLocked: () => boolean
  dispose: () => void
}

export type ActivityRangeChange = (range: ActivityAnalysisRange | null, committed: boolean) => void

export const linkActivityAnalysis = (
  presentation: TriathlonPresentation,
  act: HTMLElement,
  analysis: HTMLElement | null,
  detail: StravaActivityDetail,
  onRange?: ActivityRangeChange,
): ActivityAnalysisController | null => {
  const route = detail.route
  const sport = detail.sport
  const rangeButtons = new Set<HTMLButtonElement>(
    act.querySelectorAll<HTMLButtonElement>('[data-analysis-range]'),
  )
  for (const button of analysis?.querySelectorAll<HTMLButtonElement>('[data-analysis-range]') ?? [])
    rangeButtons.add(button)
  const ranges = Array.from(rangeButtons)
    .map(analysisRangeFromButton)
    .filter((range): range is PresetActivityAnalysisRange => range != null)
  if (route.length < 2) return null

  const maxDistanceKm = route[route.length - 1].d || 1
  const routeSelected = act.querySelector<SVGPathElement>('.tri-route-selected')
  if (!routeSelected && !act.querySelector('.tri-analysis-selection')) return null
  const readout = analysis?.querySelector<HTMLElement>('[data-tri-analysis-readout]') ?? null
  const readoutLabel = readout?.querySelector<HTMLElement>('.tri-analysis-readout-label') ?? null
  const readoutMetrics =
    readout?.querySelector<HTMLElement>('.tri-analysis-readout-metrics') ?? null
  const stateHost = analysis ?? act
  const selectedKind = stateHost.dataset.selectedKind
  const selectedId = stateHost.dataset.selectedId
  const storedStartDistanceKm = analysisFinite(stateHost.dataset.selectionStartDistanceKm)
  const storedEndDistanceKm = analysisFinite(stateHost.dataset.selectionEndDistanceKm)
  const storedSelection =
    selectedKind === 'selection' && storedStartDistanceKm != null && storedEndDistanceKm != null
      ? activitySelectionSummary(
          route,
          analysisRouteIndex(route, storedStartDistanceKm),
          analysisRouteIndex(route, storedEndDistanceKm),
        )
      : null
  let locked: ActivityAnalysisRange | null =
    ranges.find(range => range.kind === selectedKind && range.id === selectedId) ??
    ranges.find(range => range.button.getAttribute('aria-pressed') === 'true') ??
    (storedSelection
      ? {
          ...storedSelection,
          button: null,
          kind: null,
          id: null,
          label: triText(presentation.locale, 'selection'),
        }
      : null)
  const listeners = new AbortController()
  const rangeMetrics = (range: ActivityAnalysisRange): string[] => {
    const metrics = [scrubDist(presentation, range.distanceKm, sport)]
    if (range.elevationGainM != null)
      metrics.push(`+${formatAltitude(presentation, range.elevationGainM)}`)
    metrics.push(clock(range.durationS))
    return metrics
  }
  const rangeReadoutMetrics = (range: ActivityAnalysisRange): string[] => {
    const cadenceScale = sport === 'run' ? 2 : 1
    const cadenceUnit = sport === 'run' ? 'spm' : 'rpm'
    const metrics = rangeMetrics(range)
    if (range.averageSpeedKph != null)
      metrics.push(analysisRate(presentation, sport, range.averageSpeedKph))
    if (range.averageWatts != null) metrics.push(`${Math.round(range.averageWatts)} W`)
    if (range.averageHeartRate != null) metrics.push(`${Math.round(range.averageHeartRate)} bpm`)
    if (range.averageCadence != null)
      metrics.push(`${Math.round(range.averageCadence * cadenceScale)} ${cadenceUnit}`)
    if (range.averageRespirationRate != null)
      metrics.push(formatRespirationRate(range.averageRespirationRate))
    if (range.averageTemperatureC != null)
      metrics.push(formatTemperature(presentation, range.averageTemperatureC))
    return metrics
  }
  const showReadout = (range: ActivityAnalysisRange | null): void => {
    if (!readout) return
    if (!range) {
      readout.dataset.visible = 'false'
      readout.setAttribute('aria-hidden', 'true')
      return
    }
    if (readoutLabel) readoutLabel.textContent = range.label
    if (readoutMetrics) readoutMetrics.textContent = rangeReadoutMetrics(range).join(' · ')
    readout.dataset.visible = 'true'
    readout.setAttribute('aria-hidden', 'false')
  }
  const clearRange = (committed = false): void => {
    for (const selection of act.querySelectorAll<SVGRectElement>('.tri-analysis-selection')) {
      selection.setAttribute('x', '0')
      selection.setAttribute('width', '0')
    }
    act.querySelector<SVGPathElement>('.tri-route-selected')?.setAttribute('d', '')
    onRange?.(null, committed)
  }
  const showRange = (range: ActivityAnalysisRange, committed = false): void => {
    const startDistanceKm = Math.max(0, Math.min(maxDistanceKm, range.startDistanceKm))
    const endDistanceKm = Math.max(startDistanceKm, Math.min(maxDistanceKm, range.endDistanceKm))
    const x = (startDistanceKm / maxDistanceKm) * 100
    const width = Math.max(0, ((endDistanceKm - startDistanceKm) / maxDistanceKm) * 100)
    for (const selection of act.querySelectorAll<SVGRectElement>('.tri-analysis-selection')) {
      selection.setAttribute('x', x.toFixed(2))
      selection.setAttribute('width', width.toFixed(2))
    }
    act
      .querySelector<SVGPathElement>('.tri-route-selected')
      ?.setAttribute('d', analysisRoutePath(route, range))
    onRange?.(range, committed)
  }
  const showLocked = (): void => {
    if (locked) showRange(locked, true)
    else clearRange(true)
  }
  const sameRange = (
    left: ActivityAnalysisRange | null,
    right: ActivityAnalysisRange | null,
  ): boolean =>
    left != null &&
    right != null &&
    left.kind != null &&
    right.kind != null &&
    left.id != null &&
    right.id != null &&
    left.kind === right.kind &&
    left.id === right.id
  const setPreviewed = (range: ActivityAnalysisRange | null): void => {
    for (const candidate of ranges)
      candidate.button.toggleAttribute('data-analysis-preview', sameRange(candidate, range))
  }
  const setLocked = (range: ActivityAnalysisRange | null): void => {
    locked = range
    if (range?.kind && range.id) {
      stateHost.dataset.selectedKind = range.kind
      stateHost.dataset.selectedId = range.id
      delete stateHost.dataset.selectionStartDistanceKm
      delete stateHost.dataset.selectionEndDistanceKm
    } else if (range) {
      stateHost.dataset.selectedKind = 'selection'
      stateHost.dataset.selectedId = 'selection'
      stateHost.dataset.selectionStartDistanceKm = `${range.startDistanceKm}`
      stateHost.dataset.selectionEndDistanceKm = `${range.endDistanceKm}`
    } else {
      delete stateHost.dataset.selectedKind
      delete stateHost.dataset.selectedId
      delete stateHost.dataset.selectionStartDistanceKm
      delete stateHost.dataset.selectionEndDistanceKm
    }
    for (const candidate of ranges)
      candidate.button.setAttribute('aria-pressed', String(sameRange(candidate, range)))
    showReadout(range)
    showLocked()
  }
  for (const range of ranges) {
    range.button.addEventListener(
      'pointerenter',
      () => {
        setPreviewed(range)
        showRange(range)
        showReadout(range)
      },
      { signal: listeners.signal },
    )
    range.button.addEventListener(
      'pointerleave',
      () => {
        setPreviewed(null)
        showLocked()
        showReadout(locked)
      },
      { signal: listeners.signal },
    )
    range.button.addEventListener(
      'focus',
      () => {
        setPreviewed(range)
        showRange(range)
        showReadout(range)
      },
      { signal: listeners.signal },
    )
    range.button.addEventListener(
      'blur',
      () => {
        if (range.button.matches(':hover')) return
        setPreviewed(null)
        showLocked()
        showReadout(locked)
      },
      { signal: listeners.signal },
    )
    range.button.addEventListener(
      'click',
      () => {
        if (sameRange(locked, range)) {
          setLocked(null)
        } else setLocked(range)
      },
      { signal: listeners.signal },
    )
  }
  routeSelected?.addEventListener(
    'click',
    event => {
      event.stopPropagation()
      setLocked(null)
    },
    { signal: listeners.signal },
  )
  const restoreSelection = (event: Event): void => {
    if (!(event instanceof CustomEvent) || !isRecord(event.detail)) return
    if (event.detail.selected !== true) {
      setLocked(null)
      return
    }
    const kind = typeof event.detail.kind === 'string' ? event.detail.kind : undefined
    const id = typeof event.detail.id === 'string' ? event.detail.id : undefined
    if (kind === 'selection') {
      const startDistanceKm =
        typeof event.detail.startDistanceKm === 'number' ? event.detail.startDistanceKm : null
      const endDistanceKm =
        typeof event.detail.endDistanceKm === 'number' ? event.detail.endDistanceKm : null
      if (startDistanceKm != null && endDistanceKm != null) {
        const summary = activitySelectionSummary(
          route,
          analysisRouteIndex(route, startDistanceKm),
          analysisRouteIndex(route, endDistanceKm),
        )
        if (summary) {
          setLocked({
            ...summary,
            button: null,
            kind: null,
            id: null,
            label: triText(presentation.locale, 'selection'),
          })
          return
        }
      }
    }
    setLocked(ranges.find(range => range.kind === kind && range.id === id) ?? null)
  }
  if (analysis || act.dataset.activityId)
    stateHost.addEventListener('tri:analysis-restore', restoreSelection, {
      signal: listeners.signal,
    })
  setLocked(locked)
  const controller: ActivityAnalysisController = {
    preview: range => {
      setPreviewed(range)
      showRange(range)
      showReadout(range)
    },
    restore: () => {
      setPreviewed(null)
      showLocked()
      showReadout(locked)
    },
    lock: setLocked,
    clear: () => setLocked(null),
    hasLocked: () => locked != null,
    dispose: () => {
      setPreviewed(null)
      listeners.abort()
    },
  }
  return controller
}

export const linkScrub = (
  presentation: TriathlonPresentation,
  act: HTMLElement,
  marker: SVGElement | null,
  surfaces: ScrubSurface[],
  route: StravaActivityDetail['route'],
  detail?: StravaActivityDetail,
  analysisOverride?: HTMLElement | null,
  onRange?: ActivityRangeChange,
): ActivityAnalysisController | null => {
  const analysis = analysisOverride ?? act.querySelector<HTMLElement>('[data-tri-analysis]')
  const rangeController = detail
    ? linkActivityAnalysis(presentation, act, analysis, detail, onRange)
    : null
  const pad = 6
  const span = 88
  const resolved: {
    wrap: HTMLElement
    svgEl: SVGElement
    cursor: SVGElement
    readout: HTMLElement
    samples: readonly ActivityScrubSample[]
    fmt: ScrubSurface['fmt']
  }[] = []
  for (const s of surfaces) {
    if (s.samples.length < 2) continue
    const svgEl = s.wrap.querySelector<SVGElement>('.tri-elev')
    const cursor = svgEl?.querySelector<SVGElement>('.tri-elev-cursor')
    if (!svgEl || !cursor) continue
    const readout = el('div', 'tri-fig-readout')
    const readoutHost =
      (s.wrap.dataset.triTrace
        ? s.wrap.querySelector<HTMLElement>(':scope > .tri-elev-cap')
        : null) ?? s.wrap
    readoutHost.appendChild(readout)
    resolved.push({ wrap: s.wrap, svgEl, cursor, readout, samples: s.samples, fmt: s.fmt })
  }
  if (resolved.length === 0) return rangeController
  const listeners = new AbortController()
  const frameCleanups: (() => void)[] = []
  const indexAt = (clientX: number, surface: (typeof resolved)[number]): number => {
    const { svgEl, samples } = surface
    const r = svgEl.getBoundingClientRect()
    const fraction = Math.min(1, Math.max(0, (clientX - r.left) / r.width))
    const domainStart =
      analysisFinite(svgEl.dataset.domainStartElapsedS) ??
      analysisFinite(svgEl.dataset.domainStartDistanceKm) ??
      samples[0].d
    const domainEnd =
      analysisFinite(svgEl.dataset.domainEndElapsedS) ??
      analysisFinite(svgEl.dataset.domainEndDistanceKm) ??
      samples.at(-1)?.d ??
      domainStart
    return activityScrubIndexAt(samples, domainStart + fraction * (domainEnd - domainStart))
  }
  for (const surf of resolved) {
    let pendingX: number | null = null
    let frame: number | null = null
    frameCleanups.push(() => {
      if (frame != null) window.cancelAnimationFrame(frame)
      frame = null
    })
    let drag: {
      pointerId: number
      startClientX: number
      anchorIndex: number
      range: ActivityAnalysisRange | null
    } | null = null
    const show = (clientX: number) => {
      const sampleIndex = indexAt(clientX, surf)
      const sample = surf.samples[sampleIndex]
      const routeIndex = route.length > 0 ? activityScrubElapsedIndexAt(route, sample.elapsedS) : -1
      const routePoint = route[routeIndex]
      if (drag && routePoint) {
        const nextRange =
          Math.abs(clientX - drag.startClientX) >= 3
            ? activitySelectionSummary(route, drag.anchorIndex, routeIndex)
            : null
        drag.range = nextRange
          ? {
              ...nextRange,
              button: null,
              kind: null,
              id: null,
              label: triText(presentation.locale, 'selection'),
            }
          : null
        act.classList.toggle('tri-act--selecting', drag.range != null)
        if (drag.range) rangeController?.preview(drag.range)
        else rangeController?.restore()
      }
      for (const linked of resolved) {
        const linkedIndex = activityScrubElapsedIndexAt(linked.samples, sample.elapsedS)
        const linkedSample = linked.samples[linkedIndex]
        const linkedMax = linked.samples.at(-1)?.d || 1
        const x = ((linkedSample.d / linkedMax) * 100).toFixed(2)
        linked.cursor.setAttribute('x1', x)
        linked.cursor.setAttribute('x2', x)
        if (rangeController?.hasLocked() || drag?.range || linked === surf)
          linked.readout.textContent = linked.fmt(linkedIndex)
      }
      if (marker && routePoint) {
        marker.setAttribute('cx', (pad + routePoint.x * span).toFixed(2))
        marker.setAttribute('cy', (pad + (1 - routePoint.y) * span).toFixed(2))
      }
      const linkedReadouts = Boolean(rangeController?.hasLocked() || drag?.range)
      act.classList.add('tri-act--scrub')
      for (const linked of resolved)
        linked.wrap.classList.toggle('tri-elev-wrap--read', linkedReadouts || linked === surf)
    }
    const onMove = (event: PointerEvent) => {
      if (drag && event.pointerId !== drag.pointerId) return
      pendingX = event.clientX
      if (frame != null) return
      frame = window.requestAnimationFrame(() => {
        frame = null
        if (pendingX == null || !surf.svgEl.isConnected) return
        show(pendingX)
        pendingX = null
      })
    }
    const finishFrame = (clientX: number): void => {
      if (frame != null) window.cancelAnimationFrame(frame)
      frame = null
      pendingX = clientX
      show(clientX)
      pendingX = null
    }
    const onDown = (event: PointerEvent): void => {
      if (!event.isPrimary || event.button !== 0 || drag || route.length < 2) return
      const sample = surf.samples[indexAt(event.clientX, surf)]
      pendingX = event.clientX
      drag = {
        pointerId: event.pointerId,
        startClientX: event.clientX,
        anchorIndex: activityScrubElapsedIndexAt(route, sample.elapsedS),
        range: null,
      }
      surf.svgEl.setPointerCapture(event.pointerId)
    }
    const onUp = (event: PointerEvent): void => {
      if (!drag || event.pointerId !== drag.pointerId) return
      finishFrame(event.clientX)
      const selected = drag.range
      const pointerId = drag.pointerId
      drag = null
      act.classList.remove('tri-act--selecting')
      if (selected) rangeController?.lock(selected)
      else if (rangeController?.hasLocked()) rangeController.clear()
      if (surf.svgEl.hasPointerCapture(pointerId)) surf.svgEl.releasePointerCapture(pointerId)
    }
    const onCancel = (event: PointerEvent): void => {
      if (!drag || event.pointerId !== drag.pointerId) return
      drag = null
      rangeController?.restore()
      act.classList.remove('tri-act--selecting')
      onLeave()
    }
    const onLeave = () => {
      if (drag) return
      if (frame != null) window.cancelAnimationFrame(frame)
      frame = null
      pendingX = null
      act.classList.remove('tri-act--scrub')
      for (const r of resolved) r.wrap.classList.remove('tri-elev-wrap--read')
    }
    surf.svgEl.addEventListener('pointerdown', onDown, { signal: listeners.signal })
    surf.svgEl.addEventListener('pointermove', onMove, { signal: listeners.signal })
    surf.svgEl.addEventListener('pointerup', onUp, { signal: listeners.signal })
    surf.svgEl.addEventListener('pointerleave', onLeave, { signal: listeners.signal })
    surf.svgEl.addEventListener('pointercancel', onCancel, { signal: listeners.signal })
  }
  const base: ActivityAnalysisController = rangeController ?? {
    preview: () => {},
    restore: () => {},
    lock: () => {},
    clear: () => {},
    hasLocked: () => false,
    dispose: () => {},
  }
  return {
    preview: base.preview,
    restore: base.restore,
    lock: base.lock,
    clear: base.clear,
    hasLocked: base.hasLocked,
    dispose: () => {
      listeners.abort()
      for (const cleanup of frameCleanups) cleanup()
      for (const surface of resolved) surface.readout.remove()
      base.dispose()
    },
  }
}
