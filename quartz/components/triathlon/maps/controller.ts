import type { Analytics } from '../../../plugins/stores/analytics'
import type { StravaActivityDetail } from '../../../plugins/stores/strava'
import type { StravaDetailIndex } from '../../../util/strava-detail'
import type { ActivityAnalysisRange } from '../activity/analysis'
import type { TriathlonContext } from '../runtime/context'
import { start } from '../../../functional'
import { applyMapElevation, applyMapTerrain } from '../../../util/mapbox-overlays'
import { buildAnalysisBar, dist } from '../../../util/triathlon-card'
import { mapRoutePointAtDistance } from '../../../util/triathlon-map-route'
import { applyMonochromeMapPalette } from '../../scripts/mapbox-client'
import { createStreetMapMatcher, type StreetMapMatcher } from '../../scripts/triathlon-map-heat'
import {
  detailContextFromPayload,
  parseActivityDetailFromShardMap,
  readActivityDetailFromShard,
  readDetailIndex,
  warmActivityShard,
  type DetailPayload,
} from '../activity/data'
import { metricSpecs, renderMapDetail } from '../activity/render'
import { detailHead } from '../analytics/search'
import { createDomFactory, el, setMath } from '../runtime/dom'
import {
  mapboxStyleUrl,
  readTriMap3d,
  readTriMapElevation,
  readTriMapStyle,
  readTriMapTheme,
  setTriMap3d,
  setTriMapElevation,
  setTriMapStyle,
  TRI_MAP_STYLE_EVENT,
  TRI_MAP_3D_EVENT,
  TRI_MAP_ELEVATION_EVENT,
  TRI_POWER_FILTER_EVENT,
  type TriMapTheme,
} from '../runtime/preferences'
import { createMapDetailTransition } from './detail'
import { createMapboxMap, type MapboxLayerEvent, type TriathlonMapboxMap } from './mapbox'
import {
  DEFAULT_OVERVIEW_BOUNDS,
  emptyFC,
  fcBounds,
  gpsSegments,
  heatCasingWidthExpr,
  heatColorExpr,
  heatOpacityExpr,
  heatWidthExpr,
  initialMapModel,
  metricRouteFC,
  overviewBounds,
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

const sameBounds = (
  left: [[number, number], [number, number]] | null,
  right: [[number, number], [number, number]] | null,
): boolean =>
  left != null &&
  right != null &&
  Math.abs(left[0][0] - right[0][0]) < 1e-4 &&
  Math.abs(left[0][1] - right[0][1]) < 1e-4 &&
  Math.abs(left[1][0] - right[1][0]) < 1e-4 &&
  Math.abs(left[1][1] - right[1][1]) < 1e-4

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
  let mapData: DetailPayload | null = null
  let detailData: DetailPayload | null = null
  const loadedDetails = new Map<string, StravaActivityDetail>()
  const pendingActivityDetails = new Map<string, Promise<StravaActivityDetail | null>>()
  const shardCache = new Map<string, Promise<Record<string, unknown> | null>>()
  const resolvedShardCache = new Map<string, Record<string, unknown>>()
  let detailIndexState: { baseUrl: string; index: StravaDetailIndex } | null = null
  let detailIndexPromise: Promise<{ baseUrl: string; index: StravaDetailIndex } | null> | null =
    null
  const activeDetails = (): DetailPayload | null => detailData ?? mapData
  const fullActivityDetail = (id: string): StravaActivityDetail | null => {
    const fromFull = detailData?.details?.[id]
    if (fromFull) return fromFull
    const cached = loadedDetails.get(id)
    if (cached) return cached
    for (const shardDetails of resolvedShardCache.values()) {
      if (!(id in shardDetails)) continue
      const parsed = parseActivityDetailFromShardMap(shardDetails, id)
      if (parsed) {
        loadedDetails.set(id, parsed)
        if (mapData) mapData.details[id] = parsed
        return parsed
      }
    }
    return null
  }
  const ensureDetailIndex = (): Promise<{ baseUrl: string; index: StravaDetailIndex } | null> => {
    if (detailIndexState) return Promise.resolve(detailIndexState)
    if (detailIndexPromise) return detailIndexPromise
    const detailPath = root.dataset.detailPath
    if (!detailPath) return Promise.resolve(null)
    detailIndexPromise = readDetailIndex(
      new URL(detailPath, window.location.href).toString(),
      context.signal,
    ).then(result => {
      detailIndexPromise = null
      if (!live || context.signal.aborted || !result) return null
      detailIndexState = result
      if (mapData) {
        const { index } = result
        mapData.zones = index.zones ?? mapData.zones
        mapData.powerCurveRef = index.powerCurveRef
        mapData.powerCurveYearRef = index.powerCurveYearRef
        mapData.runPowerCurveRef = index.runPowerCurveRef
        mapData.runPowerCurveYearRef = index.runPowerCurveYearRef
        mapData.powerCurveYear = index.powerCurveYear ?? mapData.powerCurveYear
        mapData.criticalPower = index.criticalPower ?? mapData.criticalPower
        mapData.criticalPowerYear = index.criticalPowerYear ?? mapData.criticalPowerYear
        mapData.ftp = index.ftp ?? mapData.ftp
        mapData.goalFtp = index.goalFtp ?? mapData.goalFtp
        mapData.vt1Hr = index.vt1Hr ?? mapData.vt1Hr
      }
      return result
    })
    return detailIndexPromise
  }
  const ensureActivityDetail = (id: string): Promise<StravaActivityDetail | null> => {
    const existing = fullActivityDetail(id)
    if (existing) return Promise.resolve(existing)
    const inflight = pendingActivityDetails.get(id)
    if (inflight) return inflight
    const detailPath = root.dataset.detailPath
    if (!detailPath) return Promise.resolve(null)
    const baseUrl =
      detailIndexState?.baseUrl ?? new URL(detailPath, window.location.href).toString()
    const recordLoaded = (loaded: StravaActivityDetail | null): StravaActivityDetail | null => {
      if (!live || context.signal.aborted || !loaded) return null
      loadedDetails.set(id, loaded)
      if (mapData) mapData.details[id] = loaded
      return loaded
    }
    const load = (async (): Promise<StravaActivityDetail | null> => {
      const knownShard = mapData?.shardById?.[id]
      if (knownShard) {
        const direct = await readActivityDetailFromShard(
          new URL(knownShard, baseUrl).toString(),
          id,
          context.signal,
          shardCache,
          resolvedShardCache,
        )
        if (direct) return recordLoaded(direct)
      }
      const month = mapData?.details?.[id]?.date?.slice(0, 7)
      if (month && /^\d{4}-\d{2}$/.test(month)) {
        const primaryShard = `strava-detail/${month}.json`
        if (primaryShard !== knownShard) {
          const fromMonth = await readActivityDetailFromShard(
            new URL(primaryShard, baseUrl).toString(),
            id,
            context.signal,
            shardCache,
            resolvedShardCache,
          )
          if (fromMonth) return recordLoaded(fromMonth)
        }
      }
      const indexResult = await ensureDetailIndex()
      if (!indexResult) return null
      const candidates =
        month && /^\d{4}-\d{2}$/.test(month)
          ? [
              ...indexResult.index.shards.filter(path => path.startsWith(`strava-detail/${month}`)),
              ...indexResult.index.shards.filter(
                path => !path.startsWith(`strava-detail/${month}`),
              ),
            ]
          : indexResult.index.shards
      for (const shardPath of candidates) {
        const found = await readActivityDetailFromShard(
          new URL(shardPath, indexResult.baseUrl).toString(),
          id,
          context.signal,
          shardCache,
          resolvedShardCache,
        )
        if (found) return recordLoaded(found)
      }
      return null
    })().finally(() => {
      pendingActivityDetails.delete(id)
    })
    pendingActivityDetails.set(id, load)
    return load
  }
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
  const elevationBtn = side?.querySelector<HTMLButtonElement>('.tri-map-elevation') ?? null
  const styleBtn = side?.querySelector<HTMLButtonElement>('.tri-map-style') ?? null
  let threeDimensional = readTriMap3d()
  let elevation = readTriMapElevation()
  const sportFilter = createRouteSportFilter(sportBtns)
  const overview = createOverviewProvider(
    () => context.presentation,
    activeDetails,
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
    let lastFittedBounds: [[number, number], [number, number]] | null = null
    let selection: { d: StravaActivityDetail; i: number } | null = null
    const readyMap = (): TriathlonMapboxMap | null => (okFlag ? map : null)
    const ensureMatcher = (): StreetMapMatcher | null => {
      const currentOverview = overview.current()
      if (matchedOverview !== currentOverview) {
        matchedOverview = currentOverview
        streetMapMatcher = createStreetMapMatcher(
          currentOverview.streetActivities,
          currentOverview.maximumVisits,
        )
      }
      return streetMapMatcher
    }
    const computeStreetFeatures = (current: TriathlonMapboxMap) => {
      const roadLayers = current
        .getStyle()
        ?.layers?.filter(
          (layer: { id?: unknown; type?: unknown; 'source-layer'?: unknown }) =>
            typeof layer.id === 'string' &&
            layer.type === 'line' &&
            layer['source-layer'] === 'road' &&
            !layer.id.includes('rail'),
        )
      if (!roadLayers?.length) return null
      const matcher = ensureMatcher()
      if (!matcher) return emptyFC()
      const rendered = current.queryRenderedFeatures({
        layers: roadLayers.map((layer: { id: string }) => layer.id),
      })
      if (!rendered.length) return null
      return matcher(rendered)
    }
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
      if (d.route.length === 0) {
        current.getSource('tri-sel')?.setData(routeFC(d, HEAT_RAMP[6]))
        return
      }
      const spec = metricSpecs(context.presentation, d, detailContextFromPayload(activeDetails()))[
        i
      ]
      if (spec) current.getSource('tri-sel')?.setData(metricRouteFC(d, spec))
      else current.getSource('tri-sel')?.setData(routeFC(d, HEAT_RAMP[6]))
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
      if (!current || !activeDetails() || selection) return
      const features = computeStreetFeatures(current)
      if (features) current.getSource('tri-heat')?.setData(features)
    }
    const scheduleStreetMap = () => {
      const current = readyMap()
      if (!current || selection) return
      if (activeDetails()) drawStreetMap()
      const sequence = ++streetSequence
      current.once('idle', () => {
        if (sequence === streetSequence && !selection) drawStreetMap()
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
        const d = activeDetails()?.details?.[key]
        if (!d) return
        current.getSource('tri-hov')?.setData(routeFC(d))
        if (tip)
          tip.textContent = `${d.name || d.sport} · ${context.formatter.shortDate(d.date)} · ${dist(context.presentation, d.distanceKm, d.sport)}`
        void ensureActivityDetail(key)
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
    const applyElevation = () => {
      if (!map) return
      applyMapElevation(
        map,
        elevation,
        readTriMapStyle(),
        readTriMapTheme(),
        context.presentation.distance,
      )
    }
    const installLayers = () => {
      if (!map) return
      const theme = readTriMapTheme()
      const style = readTriMapStyle()
      const satellite = style === 'satellite'
      const casingColor = satellite ? '#fff9f3' : theme === 'dark' ? '#100f0f' : '#fff9f3'
      if (style === 'mono') applyMonochromeMapPalette(map, theme)
      const firstLabelId = map.getStyle()?.layers?.find(layer => layer.type === 'symbol')?.id
      applyMapTerrain(map, threeDimensional, style, theme)
      applyElevation()
      const hasOverviewData = activeDetails() != null && !selection
      const initialTraces = hasOverviewData ? overview.current().traces : emptyFC()
      const initialHeat =
        hasOverviewData && map ? (computeStreetFeatures(map) ?? emptyFC()) : emptyFC()
      addSource('tri-heat', { type: 'geojson', data: initialHeat })
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
      addSource('tri-traces', { type: 'geojson', data: initialTraces })
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
      applyMode(!hasOverviewData || initialHeat.features.length === 0)
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
      const initialBounds =
        (activeDetails() ? overviewBounds(overview.current().traces) : null) ??
        DEFAULT_OVERVIEW_BOUNDS
      lastFittedBounds = initialBounds
      const created = await createMapboxMap(
        canvas,
        mapboxStyleUrl(readTriMapStyle(), readTriMapTheme()),
        threeDimensional,
        initialBounds,
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
      current.getSource('tri-traces')?.setData(ov.traces)
      applyMode(redrawStreetMap)
    }
    const drawOverview = ({ fit = true, redrawStreetMap = true }: OverviewDrawOptions = {}) => {
      const current = readyMap()
      if (!current) return
      setOverviewData(redrawStreetMap)
      const b = overviewBounds(overview.current().traces)
      if (fit && b && !sameBounds(b, lastFittedBounds)) {
        lastFittedBounds = b
        current.fitBounds(b, { padding: 48, maxZoom: 13, duration: reduce ? 0 : 600 })
      }
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
      if (b) {
        lastFittedBounds = null
        current.fitBounds(b, {
          padding: { top: 40, right: 40, bottom, left: 40 },
          maxZoom: range ? 17 : 15,
          duration: reduce ? 0 : duration,
        })
      }
    }
    const select = (
      d: StravaActivityDetail,
      i: number,
      fit = true,
      duration = 0,
      resizeFirst = false,
    ) => {
      selection = { d, i }
      const current = readyMap()
      if (!current) return
      if (resizeFirst) current.resize()
      const range = program.retrieve().analysisRange
      clearHover()
      current.getSource('tri-traces')?.setData(emptyFC())
      current.getSource('tri-range')?.setData(range ? rangeFC(d, range) : emptyFC())
      recolor(d, i)
      current.setPaintProperty('tri-heat', 'line-opacity', 0.06)
      current.setPaintProperty('tri-heat-casing', 'line-opacity', 0)
      if (fit) fitSelection(d, range, duration)
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
      applyMapTerrain(current, threeDimensional, readTriMapStyle(), readTriMapTheme())
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
      lastFittedBounds = null
      selection = null
    }
    return {
      init,
      ok: () => readyMap() != null,
      warmOverview: () => {
        if (activeDetails()) ensureMatcher()
      },
      drawOverview,
      applyMode,
      select,
      selectRange,
      recolor,
      moveDot,
      clearSelection,
      applyMapStyle,
      applyThreeDimensional,
      applyElevation: () => {
        if (readyMap()) applyElevation()
      },
      resize,
      dispose,
    }
  })()

  const load = (): void => {
    const status = program.retrieve().status
    if (status === 'idle' || status === 'failed') program.dispatch({ type: 'load' })
  }
  let detailCleanup: (() => void) | null = null
  const finishCloseDetail = () => {
    detailCleanup?.()
    detailCleanup = null
    panel.classList.remove('tri-map--detail')
    detail?.setAttribute('aria-hidden', 'true')
    detail?.replaceChildren()
    selectionOverlay?.replaceChildren()
    selectionOverlay?.setAttribute('aria-hidden', 'true')
    mapCtl.resize()
    program.dispatch({ type: 'clear-route' })
  }
  const detailTransition = createMapDetailTransition({
    detail,
    selection: selectionOverlay,
    reducedMotion: reduce,
    onClosed: finishCloseDetail,
  })
  const closeDetail = detailTransition.close
  const toMain = () => {
    if (panel.classList.contains('tri-map--detail') || program.retrieve().selectedRouteId != null) {
      closeDetail()
    }
    if (panel.classList.contains('tri-map--searching') || Boolean(search?.value)) {
      mapSearch.clear()
    }
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
  const renderRoute = (id: string, initialMetric = 0, selectMap = true, fitMap = selectMap) => {
    if (!detail) return
    const full = fullActivityDetail(id)
    const d = full ?? mapData?.details?.[id]
    if (!d) {
      void ensureActivityDetail(id).then(loaded => {
        if (loaded && program.retrieve().selectedRouteId === id) {
          renderRoute(id, program.retrieve().metric, selectMap, fitMap)
        }
      })
      return
    }
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
    const analysisNode = mapMode && full ? buildAnalysisBar(domF, d) : null
    const analysis =
      analysisNode instanceof HTMLElement && analysisNode.querySelector('[data-analysis-range]')
        ? analysisNode
        : null
    selectionOverlay?.replaceChildren(...(analysis ? [analysis] : []))
    selectionOverlay?.setAttribute('aria-hidden', analysis ? 'false' : 'true')
    const routeView = renderMapDetail(context.presentation, d, {
      detailContext: detailContextFromPayload(activeDetails()),
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
    const enteringDetail = !panel.classList.contains('tri-map--detail')
    panel.classList.add('tri-map--detail')
    panel.classList.remove('tri-map--searching')
    results?.setAttribute('aria-hidden', 'true')
    detail.replaceChildren(card)
    detail.setAttribute('aria-hidden', 'false')
    back.addEventListener('click', () => closeDetail(true), { once: true })
    if (mapMode) {
      if (selectMap) {
        mapCtl.select(d, initialMetric, fitMap, 0, enteringDetail)
      } else if (enteringDetail) {
        mapCtl.resize()
      }
    } else {
      body?.scrollTo({ top: 0 })
    }
    detailCleanup = routeView.mount()
    if (!full) {
      void Promise.all([ensureActivityDetail(id), ensureDetailIndex()]).then(([loaded]) => {
        if (loaded && program.retrieve().selectedRouteId === id) {
          renderRoute(id, program.retrieve().metric, true, Boolean(loaded.analysisRanges?.length))
        }
      })
    } else if (!detailIndexState && d.sport === 'bike' && d.deviceWatts) {
      void ensureDetailIndex().then(indexResult => {
        if (indexResult && program.retrieve().selectedRouteId === id) {
          renderRoute(id, program.retrieve().metric, false, false)
        }
      })
    }
  }
  let detailWarmTimer = 0
  const scheduleDetailWarmup = () => {
    if (detailWarmTimer || detailIndexState) return
    detailWarmTimer = window.setTimeout(() => {
      detailWarmTimer = 0
      if (!live || context.signal.aborted) return
      void ensureDetailIndex().then(res => {
        if (!live || context.signal.aborted || !res) return
        const latestShard = res.index.shards[0]
        if (latestShard) {
          void warmActivityShard(
            new URL(latestShard, res.baseUrl).toString(),
            context.signal,
            shardCache,
            resolvedShardCache,
          )
        }
      })
    }, 150)
  }
  const startMap = () =>
    void mapCtl.init().then(() => {
      mapCtl.resize()
      mapCtl.drawOverview()
      scheduleDetailWarmup()
    })
  const open = () => {
    const wasReady =
      mapCtl.ok() &&
      program.retrieve().status === 'ready' &&
      !panel.classList.contains('tri-map--detail')
    toMain()
    root.classList.add('tri-map-open')
    panel.setAttribute('aria-hidden', 'false')
    if (wasReady) {
      window.requestAnimationFrame(() => mapCtl.resize())
    } else {
      load()
      startMap()
    }
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
    setTriMap3d(!threeDimensional)
  }
  const onThreeDimensionalChange = () => {
    threeDimensional = readTriMap3d()
    syncThreeDimensionalBtn()
    mapCtl.applyThreeDimensional()
  }
  const onStyleClick = () =>
    setTriMapStyle(readTriMapStyle() === 'satellite' ? 'mono' : 'satellite')
  const syncElevationBtn = () => elevationBtn?.setAttribute('aria-pressed', String(elevation))
  const onElevationClick = () => {
    setTriMapElevation(!elevation)
  }
  const onElevationChange = () => {
    elevation = readTriMapElevation()
    syncElevationBtn()
    mapCtl.applyElevation()
  }
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
    mapCtl.applyElevation()
    const model = program.retrieve()
    if (model.selectedRouteId) renderRoute(model.selectedRouteId, model.metric, false)
  }
  const onPowerFilter = () => {
    overview.clear()
    mapCtl.warmOverview()
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
    details: activeDetails,
    detailsReady: () => program.retrieve().status === 'ready',
    setSports: sports => program.dispatch({ type: 'set-sports', sports }),
    selectRoute: id => program.dispatch({ type: 'select-route', id }),
    prefetchRoute: id => {
      void ensureActivityDetail(id)
    },
  })
  const program = start({
    init: () => ({ model: initialMapModel(readTriMapStyle()), effects: [] }),
    reduce: updateMap,
    effects: (effect, { dispatch, retrieve }) => {
      if (effect.type === 'load-artifacts') {
        const detailPath = root.dataset.detailPath
        const mapPath =
          root.dataset.mapPath ?? detailPath?.replace(/strava-detail\.json$/, 'strava-map.json')
        const completeLoad = (payload: DetailPayload | null, isFullDetail: boolean) => {
          if (!live || context.signal.aborted) return
          if (isFullDetail) detailData = payload
          else mapData = payload
          if ((mapPath || detailPath) && !payload) {
            dispatch({ type: 'failed', request: effect.request })
            return
          }
          overview.clear()
          mapCtl.warmOverview()
          dispatch({ type: 'loaded', request: effect.request })
          if (search?.value) mapSearch.run()
        }
        if (mapPath && context.resources.map) {
          void context.resources.map.load(mapPath).then(mapResult => {
            if (!live || context.signal.aborted) return
            if (mapResult.status === 'ready') {
              completeLoad(mapResult.value, false)
              return
            }
            if (!detailPath) {
              completeLoad(null, false)
              return
            }
            void context.resources.detail.load(detailPath).then(detailResult => {
              completeLoad(detailResult.status === 'ready' ? detailResult.value : null, true)
            })
          })
          return
        }
        void (detailPath ? context.resources.detail.load(detailPath) : Promise.resolve(null)).then(
          detailResult => {
            completeLoad(detailResult?.status === 'ready' ? detailResult.value : null, true)
          },
        )
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
        const activity = activeDetails()?.details?.[effect.id]
        if (activity) mapCtl.recolor(activity, effect.metric)
        return
      }
      if (effect.type === 'draw-range') {
        const id = retrieve().selectedRouteId
        const activity = id ? activeDetails()?.details?.[id] : null
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
  syncElevationBtn()

  let warmupTimer = 0
  if (pageMode) {
    load()
    startMap()
  } else {
    btn?.addEventListener('click', open)
    closeBtn?.addEventListener('click', close)
    title?.addEventListener('click', toMain)
    scrim?.addEventListener('click', close)
    warmupTimer = window.setTimeout(() => {
      warmupTimer = 0
      if (!live || context.signal.aborted) return
      load()
      startMap()
    }, 80)
  }
  panel.addEventListener('click', onPanelClick)
  overlay?.addEventListener('click', onModeClick)
  side?.addEventListener('click', onSportClick)
  sideFold?.addEventListener('click', onFold)
  threeDimensionalBtn?.addEventListener('click', onThreeDimensionalClick)
  elevationBtn?.addEventListener('click', onElevationClick)
  styleBtn?.addEventListener('click', onStyleClick)
  document.addEventListener('keydown', onKey)
  document.addEventListener('themechange', onThemeChange)
  window.addEventListener(TRI_MAP_STYLE_EVENT, onMapStyle)
  window.addEventListener(TRI_MAP_3D_EVENT, onThreeDimensionalChange)
  window.addEventListener(TRI_MAP_ELEVATION_EVENT, onElevationChange)
  window.addEventListener('tri:unit', onUnit)
  window.addEventListener(TRI_POWER_FILTER_EVENT, onPowerFilter)

  return () => {
    live = false
    if (warmupTimer) window.clearTimeout(warmupTimer)
    if (detailWarmTimer) window.clearTimeout(detailWarmTimer)
    btn?.removeEventListener('click', open)
    closeBtn?.removeEventListener('click', close)
    title?.removeEventListener('click', toMain)
    scrim?.removeEventListener('click', close)
    panel.removeEventListener('click', onPanelClick)
    overlay?.removeEventListener('click', onModeClick)
    side?.removeEventListener('click', onSportClick)
    sideFold?.removeEventListener('click', onFold)
    threeDimensionalBtn?.removeEventListener('click', onThreeDimensionalClick)
    elevationBtn?.removeEventListener('click', onElevationClick)
    styleBtn?.removeEventListener('click', onStyleClick)
    document.removeEventListener('keydown', onKey)
    document.removeEventListener('themechange', onThemeChange)
    window.removeEventListener(TRI_MAP_STYLE_EVENT, onMapStyle)
    window.removeEventListener(TRI_MAP_3D_EVENT, onThreeDimensionalChange)
    window.removeEventListener(TRI_MAP_ELEVATION_EVENT, onElevationChange)
    window.removeEventListener('tri:unit', onUnit)
    window.removeEventListener(TRI_POWER_FILTER_EVENT, onPowerFilter)
    mapSearch.dispose()
    program.stop()
    mapCtl.dispose()
    detailTransition.cancel()
    detailCleanup?.()
    detailCleanup = null
  }
}
