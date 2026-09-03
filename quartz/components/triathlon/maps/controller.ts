import type { Analytics } from '../../../plugins/stores/analytics'
import type { StravaActivityDetail } from '../../../plugins/stores/strava'
import type { ActivityAnalysisRange } from '../activity/analysis'
import type { TriathlonContext } from '../runtime/context'
import { start } from '../../../functional'
import { buildAnalysisBar, dist } from '../../../util/triathlon-card'
import { mapRoutePointAtDistance } from '../../../util/triathlon-map-route'
import { applyMonochromeMapPalette } from '../../scripts/mapbox-client'
import { createStreetMapMatcher, type StreetMapMatcher } from '../../scripts/triathlon-map-heat'
import { detailContextFromPayload, type DetailPayload } from '../activity/data'
import { metricSpecs, renderMapDetail } from '../activity/render'
import { detailHead } from '../analytics/search'
import { createDomFactory, el, setMath } from '../runtime/dom'
import {
  mapboxStyleUrl,
  readTriMap3d,
  readTriMapStyle,
  readTriMapTheme,
  setTriMap3d,
  setTriMapStyle,
  TRI_MAP_STYLE_EVENT,
  TRI_POWER_FILTER_EVENT,
  type TriMapTheme,
} from '../runtime/preferences'
import { createMapDetailTransition } from './detail'
import { createMapboxMap, type MapboxLayerEvent, type TriathlonMapboxMap } from './mapbox'
import {
  emptyFC,
  fcBounds,
  gpsSegments,
  heatCasingWidthExpr,
  heatColorExpr,
  heatOpacityExpr,
  heatWidthExpr,
  initialMapModel,
  metricRouteFC,
  overviewRamp,
  pointFC,
  rangeFC,
  readOverviewMode,
  readRouteSport,
  routeFC,
  streetMetricColorExpr,
  streetMetricCasingWidthExpr,
  streetMetricOpacityExpr,
  streetMetricWidthExpr,
  type Overview,
  type OverviewDrawOptions,
  sameAnalysisRange,
  updateMap,
  type MapAnalysisRange,
} from './model'
import { createOverviewProvider, createRouteSportFilter } from './overview'
import { HEAT_RAMP, rampGradient } from './palette'
import { createMapSearchController } from './search-controller'

export const setupMap = (root: HTMLElement, context: TriathlonContext): (() => void) | null => {
  const domF = createDomFactory(context.presentation)
  const btn = root.querySelector<HTMLElement>('.tri-map-btn')
  const panel = root.querySelector<HTMLElement>('.tri-map')
  const scrim = root.querySelector<HTMLElement>('.tri-map-scrim')
  const closeBtn = root.querySelector<HTMLElement>('.tri-map-close')
  const title = root.querySelector<HTMLElement>('.tri-map-title')
  const search = root.querySelector<HTMLInputElement>('.tri-map-search')
  const results = root.querySelector<HTMLElement>('.tri-map-results')
  const detail = root.querySelector<HTMLElement>('.tri-map-detail')
  const pageMode = root.dataset.triView === 'maps'
  if (!panel || (!btn && !pageMode)) return null

  let live = true
  const body = root.querySelector<HTMLElement>('.tri-map-body')
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
  let data: Analytics | null = null
  let detailData: DetailPayload | null = null
  const canvas = root.querySelector<HTMLElement>('.tri-map-canvas')
  const overlay = root.querySelector<HTMLElement>('.tri-map-overlay')
  const selectionOverlay = root.querySelector<HTMLElement>('.tri-map-selection')
  const tip = root.querySelector<HTMLElement>('.tri-map-tip')
  const legendLo = overlay?.querySelector<HTMLElement>('.tri-map-legend-lo') ?? null
  const legendHi = overlay?.querySelector<HTMLElement>('.tri-map-legend-hi') ?? null
  const legendBar = overlay?.querySelector<HTMLElement>('.tri-map-legend-bar') ?? null
  const modeBtns = Array.from(root.querySelectorAll<HTMLButtonElement>('.tri-map-mode'))
  const sportBtns = Array.from(root.querySelectorAll<HTMLButtonElement>('.tri-map-sport'))
  const side = root.querySelector<HTMLElement>('.tri-map-side')
  const sideFold = side?.querySelector<HTMLButtonElement>('.tri-map-side-fold') ?? null
  const threeDimensionalBtn = side?.querySelector<HTMLButtonElement>('.tri-map-3d') ?? null
  const styleBtn = side?.querySelector<HTMLButtonElement>('.tri-map-style') ?? null
  let threeDimensional = readTriMap3d()
  const sportFilter = createRouteSportFilter(sportBtns)
  const overview = createOverviewProvider(
    () => context.presentation,
    () => detailData,
    () => program.retrieve().enabledSports,
  )

  const mapCtl = (() => {
    let map: TriathlonMapboxMap | null = null
    let started = false
    let okFlag = false
    let hoverId: string | null = null
    let styleSeq = 0
    let eventsBound = false
    let streetSequence = 0
    let matchedOverview: Overview | null = null
    let streetMapMatcher: StreetMapMatcher | null = null
    let selection: { d: StravaActivityDetail; i: number } | null = null
    const readyMap = (): TriathlonMapboxMap | null => (okFlag ? map : null)
    const clearHover = () => {
      hoverId = null
      readyMap()?.getSource('tri-hov')?.setData(emptyFC())
      if (map) map.getCanvas().style.cursor = ''
      tip?.classList.remove('tri-map-tip--on')
    }
    const applyMode = (scheduleStreet = true) => {
      const current = readyMap()
      if (!current) return
      const mode = program.retrieve().mode
      const lineWidth = mode === 'heat' ? heatWidthExpr : streetMetricWidthExpr
      const casingLineWidth = mode === 'heat' ? heatCasingWidthExpr : streetMetricCasingWidthExpr
      current.setPaintProperty(
        'tri-heat',
        'line-color',
        mode === 'heat' ? heatColorExpr : streetMetricColorExpr(mode),
      )
      current.setPaintProperty(
        'tri-heat',
        'line-opacity',
        mode === 'heat' ? heatOpacityExpr : streetMetricOpacityExpr(mode),
      )
      current.setPaintProperty('tri-heat', 'line-width', lineWidth)
      current.setPaintProperty(
        'tri-heat-casing',
        'line-opacity',
        readTriMapStyle() === 'satellite' ? 0.82 : 0,
      )
      current.setPaintProperty('tri-heat-casing', 'line-width', casingLineWidth)
      current.setPaintProperty(
        'tri-swim',
        'line-color',
        mode === 'heat' ? HEAT_RAMP[6] : streetMetricColorExpr(mode),
      )
      current.setPaintProperty(
        'tri-swim',
        'line-opacity',
        mode === 'heat' ? 0.7 : streetMetricOpacityExpr(mode),
      )
      current.setPaintProperty('tri-swim', 'line-width', lineWidth)
      current.setPaintProperty(
        'tri-swim-casing',
        'line-opacity',
        readTriMapStyle() === 'satellite' ? 0.82 : 0,
      )
      current.setPaintProperty('tri-swim-casing', 'line-width', casingLineWidth)
      const lg = overview.current().legend[mode]
      if (legendLo) setMath(legendLo, lg?.lo ?? 'low')
      if (legendHi) setMath(legendHi, lg?.hi ?? 'high')
      if (legendBar) legendBar.style.background = rampGradient(overviewRamp(mode))
      if (scheduleStreet) scheduleStreetMap()
    }
    const recolor = (d: StravaActivityDetail, i: number) => {
      const current = readyMap()
      if (!current) return
      const spec = metricSpecs(context.presentation, d, detailContextFromPayload(detailData))[i]
      if (spec) current.getSource('tri-sel')?.setData(metricRouteFC(d, spec))
    }
    const addSource = (id: string, source: Record<string, unknown>) => {
      const current = map
      if (current && !current.getSource(id)) current.addSource(id, source)
    }
    const addLayer = (layer: Record<string, unknown>, beforeId?: string) => {
      const current = map
      const id = layer.id
      if (current && typeof id === 'string' && !current.getLayer(id))
        current.addLayer(layer, beforeId)
    }
    const drawStreetMap = () => {
      const current = readyMap()
      if (!current) return
      const roadLayers = current
        .getStyle()
        ?.layers?.filter(
          (layer: { id?: unknown; type?: unknown; 'source-layer'?: unknown }) =>
            typeof layer.id === 'string' &&
            layer.type === 'line' &&
            layer['source-layer'] === 'road' &&
            !layer.id.includes('rail'),
        )
      if (!roadLayers?.length) return
      const currentOverview = overview.current()
      if (matchedOverview !== currentOverview) {
        matchedOverview = currentOverview
        streetMapMatcher = createStreetMapMatcher(
          currentOverview.streetActivities,
          currentOverview.maximumVisits,
        )
      }
      current
        .getSource('tri-heat')
        ?.setData(
          streetMapMatcher?.(
            current.queryRenderedFeatures({
              layers: roadLayers.map((layer: { id: string }) => layer.id),
            }),
          ) ?? emptyFC(),
        )
    }
    const scheduleStreetMap = () => {
      const current = readyMap()
      if (!current) return
      const sequence = ++streetSequence
      current.once('idle', () => {
        if (sequence === streetSequence) drawStreetMap()
      })
    }
    const onTraceMove = (e: MapboxLayerEvent) => {
      if (panel.classList.contains('tri-map--detail')) return
      const current = readyMap()
      if (!current) return
      const id = e.features?.[0]?.properties?.id
      if (id == null) return
      current.getCanvas().style.cursor = 'pointer'
      const key = String(id)
      if (hoverId !== key) {
        hoverId = key
        const d = detailData?.details?.[key]
        if (!d) return
        current.getSource('tri-hov')?.setData(routeFC(d))
        if (tip)
          tip.textContent = `${d.name || d.sport} · ${context.formatter.shortDate(d.date)} · ${dist(context.presentation, d.distanceKm, d.sport)}`
      }
      if (tip) {
        tip.classList.add('tri-map-tip--on')
        if (!canvas) return
        const bound = canvas.clientWidth - tip.offsetWidth - 8
        tip.style.left = `${Math.min(e.point.x + 14, Math.max(8, bound))}px`
        tip.style.top = `${e.point.y + 14}px`
      }
    }
    const onTraceClick = (e: MapboxLayerEvent) => {
      if (panel.classList.contains('tri-map--detail')) return
      const id = e.features?.[0]?.properties?.id
      if (id != null) program.dispatch({ type: 'select-route', id: String(id) })
    }
    const bindEvents = () => {
      const current = map
      if (eventsBound || !current) return
      current.on('mousemove', 'tri-hit', onTraceMove)
      current.on('mouseleave', 'tri-hit', clearHover)
      current.on('click', 'tri-hit', onTraceClick)
      current.on('moveend', scheduleStreetMap)
      eventsBound = true
    }
    const installThreeDimensionalLayers = (theme: TriMapTheme, beforeId?: string) => {
      if (!map || !threeDimensional) return
      addSource('tri-terrain', {
        type: 'raster-dem',
        url: 'mapbox://mapbox.mapbox-terrain-dem-v1',
        tileSize: 512,
        maxzoom: 14,
      })
      map.setTerrain({ source: 'tri-terrain', exaggeration: 1.25 })
      addLayer(
        {
          id: 'tri-3d-buildings',
          source: 'composite',
          'source-layer': 'building',
          filter: ['==', ['get', 'extrude'], 'true'],
          type: 'fill-extrusion',
          minzoom: 14.5,
          paint: {
            'fill-extrusion-color':
              readTriMapStyle() === 'satellite'
                ? '#b8b5af'
                : theme === 'dark'
                  ? '#34312d'
                  : '#d8cec1',
            'fill-extrusion-height': [
              'interpolate',
              ['linear'],
              ['zoom'],
              14.5,
              0,
              14.75,
              ['coalesce', ['get', 'height'], 3],
            ],
            'fill-extrusion-base': [
              'interpolate',
              ['linear'],
              ['zoom'],
              14.5,
              0,
              14.75,
              ['coalesce', ['get', 'min_height'], 0],
            ],
            'fill-extrusion-opacity': 0.72,
            'fill-extrusion-vertical-gradient': true,
          },
        },
        beforeId,
      )
    }
    const installLayers = () => {
      if (!map) return
      const theme = readTriMapTheme()
      const style = readTriMapStyle()
      const satellite = style === 'satellite'
      const casingColor = satellite ? '#fff9f3' : theme === 'dark' ? '#100f0f' : '#fff9f3'
      if (style === 'mono') applyMonochromeMapPalette(map, theme)
      const firstLabelLayer = map
        .getStyle()
        ?.layers?.find(
          (layer: { id?: unknown; type?: unknown }) =>
            typeof layer.id === 'string' && layer.type === 'symbol',
        )
      const firstLabelId = typeof firstLabelLayer?.id === 'string' ? firstLabelLayer.id : undefined
      installThreeDimensionalLayers(theme, firstLabelId)
      addSource('tri-heat', { type: 'geojson', data: emptyFC() })
      addLayer(
        {
          id: 'tri-heat-casing',
          type: 'line',
          source: 'tri-heat',
          layout: { 'line-cap': 'round', 'line-join': 'round' },
          paint: {
            'line-color': '#100f0f',
            'line-opacity': satellite ? 0.82 : 0,
            'line-width': heatCasingWidthExpr,
          },
        },
        firstLabelId,
      )
      addLayer(
        {
          id: 'tri-heat',
          type: 'line',
          source: 'tri-heat',
          layout: { 'line-cap': 'round', 'line-join': 'round' },
          paint: {
            'line-color': heatColorExpr,
            'line-opacity': heatOpacityExpr,
            'line-width': heatWidthExpr,
          },
        },
        firstLabelId,
      )
      addSource('tri-traces', { type: 'geojson', data: emptyFC() })
      addLayer(
        {
          id: 'tri-swim-casing',
          type: 'line',
          source: 'tri-traces',
          filter: ['==', ['get', 'sport'], 'swim'],
          layout: { 'line-cap': 'round', 'line-join': 'round' },
          paint: {
            'line-color': '#100f0f',
            'line-opacity': satellite ? 0.82 : 0,
            'line-width': streetMetricCasingWidthExpr,
          },
        },
        firstLabelId,
      )
      addLayer(
        {
          id: 'tri-swim',
          type: 'line',
          source: 'tri-traces',
          filter: ['==', ['get', 'sport'], 'swim'],
          layout: { 'line-cap': 'round', 'line-join': 'round' },
          paint: {
            'line-color': HEAT_RAMP[6],
            'line-opacity': 0.7,
            'line-width': streetMetricWidthExpr,
          },
        },
        firstLabelId,
      )
      addLayer({
        id: 'tri-hit',
        type: 'line',
        source: 'tri-traces',
        layout: { 'line-cap': 'round', 'line-join': 'round' },
        paint: { 'line-color': '#000', 'line-opacity': 0, 'line-width': 12 },
      })
      addSource('tri-hov', { type: 'geojson', data: emptyFC() })
      addLayer({
        id: 'tri-hov-casing',
        type: 'line',
        source: 'tri-hov',
        layout: { 'line-cap': 'round', 'line-join': 'round' },
        paint: { 'line-color': casingColor, 'line-width': 3.2 },
      })
      addLayer({
        id: 'tri-hov',
        type: 'line',
        source: 'tri-hov',
        layout: { 'line-cap': 'round', 'line-join': 'round' },
        paint: { 'line-color': '#fc4c02', 'line-width': 2 },
      })
      addSource('tri-sel', { type: 'geojson', data: emptyFC() })
      addLayer({
        id: 'tri-sel-casing',
        type: 'line',
        source: 'tri-sel',
        layout: { 'line-cap': 'round', 'line-join': 'round' },
        paint: { 'line-color': casingColor, 'line-width': 3.4 },
      })
      addLayer({
        id: 'tri-sel',
        type: 'line',
        source: 'tri-sel',
        layout: { 'line-cap': 'round', 'line-join': 'round' },
        paint: { 'line-width': 2.1, 'line-color': ['get', 'color'] },
      })
      addSource('tri-range', { type: 'geojson', data: emptyFC() })
      addLayer({
        id: 'tri-range-casing',
        type: 'line',
        source: 'tri-range',
        layout: { 'line-cap': 'round', 'line-join': 'round' },
        paint: { 'line-color': casingColor, 'line-width': 5.2 },
      })
      addLayer({
        id: 'tri-range',
        type: 'line',
        source: 'tri-range',
        layout: { 'line-cap': 'round', 'line-join': 'round' },
        paint: { 'line-color': '#fc4c02', 'line-width': 3.2 },
      })
      addSource('tri-dot', { type: 'geojson', data: emptyFC() })
      addLayer({
        id: 'tri-dot',
        type: 'circle',
        source: 'tri-dot',
        paint: {
          'circle-radius': 3.5,
          'circle-color': '#fc4c02',
          'circle-stroke-width': 1.5,
          'circle-stroke-color': casingColor,
        },
      })
      bindEvents()
      okFlag = true
      applyMode()
    }
    const refreshMapData = (fit: boolean) => {
      if (selection) select(selection.d, selection.i, fit)
      else drawOverview({ fit })
    }
    const applyMapStyle = (theme: TriMapTheme = readTriMapTheme()) => {
      const current = map
      if (!current) return
      const seq = ++styleSeq
      okFlag = false
      clearHover()
      current.setStyle(mapboxStyleUrl(readTriMapStyle(), theme))
      current.once('style.load', () => {
        if (!map || seq !== styleSeq) return
        installLayers()
        refreshMapData(false)
      })
    }
    const init = async (): Promise<void> => {
      if (started) return
      started = true
      if (!canvas) return
      canvas.classList.remove('tri-map-canvas--down')
      canvas.textContent = ''
      const created = await createMapboxMap(
        canvas,
        mapboxStyleUrl(readTriMapStyle(), readTriMapTheme()),
        threeDimensional,
      )
      if (context.signal.aborted || !live) {
        created?.remove()
        return
      }
      if (!created) {
        started = false
        canvas.classList.add('tri-map-canvas--down')
        canvas.textContent = context.formatter.text('map unavailable')
        return
      }
      map = created
      await new Promise<void>(resolve => created.once('load', () => resolve()))
      installLayers()
    }
    const setOverviewData = (redrawStreetMap = true) => {
      const current = readyMap()
      if (!current) return
      clearHover()
      const ov = overview.current()
      if (redrawStreetMap) current.getSource('tri-heat')?.setData(emptyFC())
      current.getSource('tri-traces')?.setData(ov.traces)
      applyMode(redrawStreetMap)
    }
    const drawOverview = ({ fit = true, redrawStreetMap = true }: OverviewDrawOptions = {}) => {
      const current = readyMap()
      if (!current) return
      setOverviewData(redrawStreetMap)
      const b = fcBounds(overview.current().traces)
      if (fit && b) current.fitBounds(b, { padding: 48, maxZoom: 13, duration: reduce ? 0 : 600 })
    }
    const fitSelection = (
      d: StravaActivityDetail,
      range: MapAnalysisRange | null,
      duration: number,
    ): void => {
      const current = readyMap()
      if (!current) return
      const b = fcBounds(range ? rangeFC(d, range) : routeFC(d)) ?? fcBounds(routeFC(d))
      const bottom =
        selectionOverlay?.getAttribute('aria-hidden') === 'false'
          ? selectionOverlay.offsetHeight + 56
          : 40
      if (b)
        current.fitBounds(b, {
          padding: { top: 40, right: 40, bottom, left: 40 },
          maxZoom: range ? 17 : 15,
          duration: reduce ? 0 : duration,
        })
    }
    const select = (d: StravaActivityDetail, i: number, fit = true) => {
      selection = { d, i }
      const current = readyMap()
      if (!current) return
      const range = program.retrieve().analysisRange
      clearHover()
      current.getSource('tri-traces')?.setData(emptyFC())
      current.getSource('tri-range')?.setData(range ? rangeFC(d, range) : emptyFC())
      recolor(d, i)
      current.setPaintProperty('tri-heat', 'line-opacity', 0.06)
      current.setPaintProperty('tri-heat-casing', 'line-opacity', 0)
      if (fit) fitSelection(d, range, 600)
    }
    const selectRange = (d: StravaActivityDetail, range: MapAnalysisRange | null, fit: boolean) => {
      readyMap()
        ?.getSource('tri-range')
        ?.setData(range ? rangeFC(d, range) : emptyFC())
      if (fit) fitSelection(d, range, 450)
    }
    const moveDot = (d: StravaActivityDetail, distanceKm: number) => {
      const point = mapRoutePointAtDistance(gpsSegments(d), distanceKm)
      if (point) readyMap()?.getSource('tri-dot')?.setData(pointFC(point.lng, point.lat))
    }
    const clearSelection = () => {
      selection = null
      const current = readyMap()
      if (!current) return
      clearHover()
      current.getSource('tri-sel')?.setData(emptyFC())
      current.getSource('tri-range')?.setData(emptyFC())
      current.getSource('tri-dot')?.setData(emptyFC())
    }
    const resize = () => map?.resize()
    const applyThreeDimensional = () => {
      const current = readyMap()
      if (!current) return
      if (threeDimensional) {
        const firstLabelLayer = current
          .getStyle()
          ?.layers?.find(
            (layer: { id?: unknown; type?: unknown }) =>
              typeof layer.id === 'string' && layer.type === 'symbol',
          )
        installThreeDimensionalLayers(
          readTriMapTheme(),
          typeof firstLabelLayer?.id === 'string' ? firstLabelLayer.id : undefined,
        )
      } else {
        current.setTerrain(null)
        if (current.getLayer('tri-3d-buildings')) current.removeLayer('tri-3d-buildings')
        if (current.getSource('tri-terrain')) current.removeSource('tri-terrain')
      }
      current.easeTo({ pitch: threeDimensional ? 55 : 0, duration: reduce ? 0 : 240 })
    }
    const dispose = () => {
      clearHover()
      if (map?.remove) map.remove()
      map = null
      started = false
      okFlag = false
      eventsBound = false
      matchedOverview = null
      streetMapMatcher = null
      selection = null
    }
    return {
      init,
      ok: () => readyMap() != null,
      drawOverview,
      applyMode,
      select,
      selectRange,
      recolor,
      moveDot,
      clearSelection,
      applyMapStyle,
      applyThreeDimensional,
      resize,
      dispose,
    }
  })()

  const load = (): void => {
    const status = program.retrieve().status
    if (status === 'idle' || status === 'failed') program.dispatch({ type: 'load' })
  }
  let detailCleanup: (() => void) | null = null
  let detailLayoutFrame = 0
  let detailSelectionFrame = 0
  const cancelDetailFrames = () => {
    window.cancelAnimationFrame(detailLayoutFrame)
    window.cancelAnimationFrame(detailSelectionFrame)
    detailLayoutFrame = 0
    detailSelectionFrame = 0
  }
  const finishCloseDetail = () => {
    cancelDetailFrames()
    detailCleanup?.()
    detailCleanup = null
    panel.classList.remove('tri-map--detail')
    detail?.replaceChildren()
    selectionOverlay?.replaceChildren()
    program.dispatch({ type: 'clear-route' })
    requestAnimationFrame(() => mapCtl.resize())
  }
  const detailTransition = createMapDetailTransition({
    detail,
    selection: selectionOverlay,
    reducedMotion: reduce,
    onClosed: finishCloseDetail,
  })
  const closeDetail = detailTransition.close
  const toMain = () => {
    closeDetail()
    mapSearch.clear()
  }
  const close = () => {
    const wasOpen = root.classList.contains('tri-map-open')
    root.classList.remove('tri-map-open')
    panel.setAttribute('aria-hidden', 'true')
    if (wasOpen && !pageMode) btn?.focus({ preventScroll: true })
  }
  const mapAnalysisRange = (range: ActivityAnalysisRange | null): MapAnalysisRange | null => {
    if (!range) return null
    const { button, ...selection } = range
    void button
    return selection
  }
  const renderRoute = (id: string, initialMetric = 0, selectMap = true) => {
    if (!detail) return
    const d = detailData?.details?.[id]
    if (!d) return
    cancelDetailFrames()
    detailTransition.cancel()
    detailCleanup?.()
    detailCleanup = null
    const card = el('div', 'tri-pop-card')
    const { head, back } = detailHead(
      context.formatter.shortDate(d.date),
      d.name || d.sport,
      context.formatter.text('go back'),
    )
    card.appendChild(head)
    const mapMode = mapCtl.ok()
    const analysisNode = mapMode ? buildAnalysisBar(domF, d) : null
    const analysis =
      analysisNode instanceof HTMLElement && analysisNode.querySelector('[data-analysis-range]')
        ? analysisNode
        : null
    selectionOverlay?.replaceChildren(...(analysis ? [analysis] : []))
    selectionOverlay?.setAttribute('aria-hidden', 'true')
    const routeView = renderMapDetail(context.presentation, d, {
      detailContext: detailContextFromPayload(detailData),
      mapMode,
      initialMetric,
      onMetric: metric => {
        if (metric !== program.retrieve().metric)
          program.dispatch({ type: 'select-metric', metric })
      },
      onHover: mapMode ? point => mapCtl.moveDot(d, point.d) : undefined,
      analysis,
      onRange: mapMode
        ? (range, committed) => {
            const next = mapAnalysisRange(range)
            if (!committed) {
              mapCtl.selectRange(d, next, false)
              return
            }
            if (!sameAnalysisRange(program.retrieve().analysisRange, next))
              program.dispatch({ type: 'select-range', range: next })
          }
        : undefined,
    })
    card.appendChild(routeView.element)
    detail.replaceChildren(card)
    detailCleanup = routeView.mount()
    panel.classList.add('tri-map--detail')
    panel.classList.remove('tri-map--searching')
    results?.setAttribute('aria-hidden', 'true')
    back.addEventListener('click', () => closeDetail(true), { once: true })
    if (mapMode) {
      detailLayoutFrame = window.requestAnimationFrame(() => {
        detailLayoutFrame = 0
        if (program.retrieve().selectedRouteId !== id) return
        if (analysis) selectionOverlay?.setAttribute('aria-hidden', 'false')
        mapCtl.resize()
        if (!selectMap) return
        detailSelectionFrame = window.requestAnimationFrame(() => {
          detailSelectionFrame = 0
          if (program.retrieve().selectedRouteId === id) mapCtl.select(d, initialMetric)
        })
      })
    } else {
      body?.scrollTo({ top: 0 })
    }
  }
  const startMap = () =>
    void mapCtl.init().then(() => {
      mapCtl.resize()
      mapCtl.drawOverview()
    })
  const open = () => {
    toMain()
    root.classList.add('tri-map-open')
    panel.setAttribute('aria-hidden', 'false')
    load()
    startMap()
    panel.focus({ preventScroll: true })
  }
  const onKey = (event: KeyboardEvent) => {
    if (event.key !== 'Escape') return
    if (panel.classList.contains('tri-map--searching') && search?.value) {
      search.value = ''
      mapSearch.run()
      return
    }
    if (panel.classList.contains('tri-map--detail')) {
      closeDetail(true)
      return
    }
    if (search && search.value) {
      search.value = ''
      mapSearch.run()
      return
    }
    close()
  }
  const onPanelClick = (event: MouseEvent) => {
    if (!panel.classList.contains('tri-map--searching')) return
    if (event.target instanceof Element && event.target.closest('.tri-map-search-wrap')) return
    panel.classList.remove('tri-map--searching')
    results?.setAttribute('aria-hidden', 'true')
  }
  const onModeClick = (event: MouseEvent) => {
    const button =
      event.target instanceof Element ? event.target.closest<HTMLElement>('.tri-map-mode') : null
    const m = readOverviewMode(button?.dataset.mode)
    if (m && m !== program.retrieve().mode) program.dispatch({ type: 'set-mode', mode: m })
  }
  const onSportClick = (event: MouseEvent) => {
    const button =
      event.target instanceof Element ? event.target.closest<HTMLElement>('.tri-map-sport') : null
    const s = readRouteSport(button?.dataset.sport)
    if (s) program.dispatch({ type: 'toggle-sport', sport: s })
  }
  const syncStyleBtn = () =>
    styleBtn?.setAttribute('aria-pressed', String(readTriMapStyle() === 'satellite'))
  const syncThreeDimensionalBtn = () =>
    threeDimensionalBtn?.setAttribute('aria-pressed', String(threeDimensional))
  const onThreeDimensionalClick = () => {
    threeDimensional = !threeDimensional
    setTriMap3d(threeDimensional)
    syncThreeDimensionalBtn()
    mapCtl.applyThreeDimensional()
  }
  const onStyleClick = () =>
    setTriMapStyle(readTriMapStyle() === 'satellite' ? 'mono' : 'satellite')
  const onFold = () => {
    if (!side) return
    const folded = side.classList.toggle('tri-map-side--folded')
    sideFold?.setAttribute('aria-expanded', String(!folded))
    sideFold?.setAttribute('aria-label', folded ? 'Expand map controls' : 'Collapse map controls')
  }
  const onMapStyle = () => {
    syncStyleBtn()
    program.dispatch({ type: 'set-style', style: readTriMapStyle() })
  }
  const onThemeChange = (event: CustomEventMap['themechange']) => {
    if (readTriMapStyle() !== 'satellite') mapCtl.applyMapStyle(event.detail.theme)
  }
  const onUnit = () => {
    const model = program.retrieve()
    if (model.selectedRouteId) renderRoute(model.selectedRouteId, model.metric, false)
  }
  const onPowerFilter = () => {
    overview.clear()
    const model = program.retrieve()
    if (model.selectedRouteId) renderRoute(model.selectedRouteId, model.metric, false)
    else mapCtl.drawOverview({ fit: false })
  }
  const mapSearch = createMapSearchController({
    context,
    panel,
    search,
    results,
    analytics: () => data,
    details: () => detailData,
    detailsReady: () => program.retrieve().status === 'ready',
    setSports: sports => program.dispatch({ type: 'set-sports', sports }),
    selectRoute: id => program.dispatch({ type: 'select-route', id }),
  })
  const program = start({
    init: () => ({ model: initialMapModel(readTriMapStyle()), effects: [] }),
    reduce: updateMap,
    effects: (effect, { dispatch, retrieve }) => {
      if (effect.type === 'load-artifacts') {
        const analyticsPath = root.dataset.analyticsPath
        const detailPath = root.dataset.detailPath
        void Promise.all([
          analyticsPath ? context.resources.analytics.load(analyticsPath) : Promise.resolve(null),
          detailPath ? context.resources.detail.load(detailPath) : Promise.resolve(null),
        ]).then(([analyticsResult, detailResult]) => {
          if (!live || context.signal.aborted) return
          if (analyticsResult?.status === 'ready') data = analyticsResult.value
          if (detailResult?.status === 'ready') detailData = detailResult.value
          if (detailPath && detailResult?.status !== 'ready') {
            dispatch({ type: 'failed', request: effect.request })
            return
          }
          overview.clear()
          dispatch({ type: 'loaded', request: effect.request })
          if (search?.value) mapSearch.run()
        })
        return
      }
      if (effect.type === 'draw-overview') {
        sportFilter.sync(retrieve().enabledSports)
        if (!retrieve().selectedRouteId) mapCtl.clearSelection()
        mapCtl.drawOverview(effect.options)
        return
      }
      if (effect.type === 'draw-route') {
        renderRoute(effect.id, effect.metric)
        return
      }
      if (effect.type === 'apply-metric') {
        const activity = detailData?.details?.[effect.id]
        if (activity) mapCtl.recolor(activity, effect.metric)
        return
      }
      if (effect.type === 'draw-range') {
        const id = retrieve().selectedRouteId
        const activity = id ? detailData?.details?.[id] : null
        if (activity) mapCtl.selectRange(activity, effect.range, true)
        return
      }
      if (effect.type === 'apply-mode') {
        const accent = overviewRamp(effect.mode)[6]
        for (const button of modeBtns) {
          const on = button.dataset.mode === effect.mode
          button.setAttribute('aria-pressed', String(on))
          button.style.background = on ? accent : ''
          button.style.borderColor = on ? accent : ''
        }
        mapCtl.applyMode()
        return
      }
      mapCtl.applyMapStyle()
    },
  })
  syncStyleBtn()
  syncThreeDimensionalBtn()

  if (pageMode) {
    load()
    startMap()
  } else {
    btn?.addEventListener('click', open)
    closeBtn?.addEventListener('click', close)
    title?.addEventListener('click', toMain)
    scrim?.addEventListener('click', close)
  }
  panel.addEventListener('click', onPanelClick)
  overlay?.addEventListener('click', onModeClick)
  side?.addEventListener('click', onSportClick)
  sideFold?.addEventListener('click', onFold)
  threeDimensionalBtn?.addEventListener('click', onThreeDimensionalClick)
  styleBtn?.addEventListener('click', onStyleClick)
  document.addEventListener('keydown', onKey)
  document.addEventListener('themechange', onThemeChange)
  window.addEventListener(TRI_MAP_STYLE_EVENT, onMapStyle)
  window.addEventListener('tri:unit', onUnit)
  window.addEventListener(TRI_POWER_FILTER_EVENT, onPowerFilter)

  return () => {
    live = false
    btn?.removeEventListener('click', open)
    closeBtn?.removeEventListener('click', close)
    title?.removeEventListener('click', toMain)
    scrim?.removeEventListener('click', close)
    panel.removeEventListener('click', onPanelClick)
    overlay?.removeEventListener('click', onModeClick)
    side?.removeEventListener('click', onSportClick)
    sideFold?.removeEventListener('click', onFold)
    threeDimensionalBtn?.removeEventListener('click', onThreeDimensionalClick)
    styleBtn?.removeEventListener('click', onStyleClick)
    document.removeEventListener('keydown', onKey)
    document.removeEventListener('themechange', onThemeChange)
    window.removeEventListener(TRI_MAP_STYLE_EVENT, onMapStyle)
    window.removeEventListener('tri:unit', onUnit)
    window.removeEventListener(TRI_POWER_FILTER_EVENT, onPowerFilter)
    mapSearch.dispose()
    program.stop()
    mapCtl.dispose()
    detailTransition.cancel()
    cancelDetailFrames()
    detailCleanup?.()
    detailCleanup = null
  }
}
