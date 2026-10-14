import { render } from 'preact'
import { useEffect, useRef } from 'preact/hooks'
import type { CardLayer, CardScene, LayerBounds } from './scripts/press-kit-layers'
import { buildSelect, type Select, type SelectOption } from './controls/select'
import { readCardBackground } from './scripts/press-kit-background'
import { paintRulers, type RulerPoint } from './scripts/press-kit-rulers'

interface Placement {
  x: number
  y: number
  scaleX: number
  scaleY: number
  hidden: boolean
}
type ResizeAxis = 'x' | 'y' | 'xy'
const resizeHandles: { axis: ResizeAxis; label: string }[] = [
  { axis: 'x', label: 'largeur' },
  { axis: 'y', label: 'hauteur' },
  { axis: 'xy', label: 'largeur et hauteur' },
]
interface Background {
  image: HTMLImageElement
  name: string
  fit: 'cover' | 'contain'
  zoom: number
  x: number
  y: number
  opacity: number
}
interface Composition {
  placements: Record<string, Placement>
  order: string[]
}
interface Layout {
  formats: Record<string, Composition>
  metricGap: number
  title?: string
  background?: Background
}
interface Draft {
  value: Layout
  past: Layout[]
  future: Layout[]
}
interface LayerPlacement {
  layer: CardLayer
  placement: Placement
}
interface PointerGesture {
  pointer: number
  x: number
  y: number
  selection: string[]
}
type Gesture = PointerGesture &
  (
    | { kind: 'marquee'; additive: boolean; bounds: LayerBounds }
    | {
        kind: 'transform'
        initial: LayerPlacement[]
        bounds: LayerBounds
        before: Layout
        resize: ResizeAxis | null
        point: RulerPoint
        moveAxis?: 'x' | 'y'
        moved: boolean
        toggleOnClick?: string
      }
  )
interface LayerRaster {
  width: number
  height: number
  image: CanvasImageSource
  pending?: Promise<void>
  failed?: { width: number; height: number }
}
const empty = (): Layout => ({ formats: {}, metricGap: 0 })
const copy = (layout: Layout): Layout => ({
  metricGap: layout.metricGap,
  title: layout.title,
  formats: Object.fromEntries(
    Object.entries(layout.formats).map(([key, composition]) => [
      key,
      {
        placements: Object.fromEntries(
          Object.entries(composition.placements).map(([id, value]) => [id, { ...value }]),
        ),
        order: [...composition.order],
      },
    ]),
  ),
  background: layout.background ? { ...layout.background } : undefined,
})
const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value))

// Hosts the shared select inside the editor's Preact tree; it rebuilds when its options change.
function EditorSelect({
  id,
  label,
  options,
  selected,
  onSelect,
}: {
  id: string
  label: string
  options: SelectOption<string>[]
  selected: string
  onSelect: (value: string) => void
}) {
  const host = useRef<HTMLDivElement>(null)
  const select = useRef<Select<string>>()
  const latest = useRef(onSelect)
  latest.current = onSelect
  const shape = JSON.stringify(options)
  useEffect(() => {
    const control = buildSelect({
      id,
      label,
      groups: [{ options }],
      selected,
      onSelect: value => latest.current(value),
    })
    host.current?.replaceChildren(control.element)
    const unmount = control.mount()
    select.current = control
    return () => {
      unmount()
      control.element.remove()
      select.current = undefined
    }
  }, [id, label, shape])
  useEffect(() => select.current?.setSelected(selected), [selected])
  return <div ref={host} />
}

export class PressKitEditor {
  private details = document.createElement('details')
  private controls = document.createElement('div')
  private overlay = document.createElement('div')
  private rulers = document.createElement('div')
  private showRulers = false
  private pointer?: RulerPoint
  private file = document.createElement('input')
  private drafts = new Map<string, Draft>()
  private draft: Draft = { value: empty(), past: [], future: [] }
  private scene?: CardScene
  private rasters = new Map<CardLayer, LayerRaster>()
  private selected = new Set<string>()
  private gesture?: Gesture
  private frame = 0
  private revision = 0
  private pendingImage?: Promise<void>
  private key = ''
  private formatKey = ''
  private available = false
  private enabled = false
  private loading = false
  private error = ''
  private observer: ResizeObserver

  constructor(
    private canvas: HTMLCanvasElement,
    controls: HTMLElement,
    private signal: AbortSignal,
    private changed: () => void,
    private requestScene: () => void,
    private defaultTitle: () => string,
  ) {
    this.details.className = 'press-kit-editor'
    this.details.dataset.for = 'image'
    const summary = document.createElement('summary')
    const label = document.createElement('span')
    label.className = 'press-kit-editor-label'
    label.textContent = 'réglages avancés'
    const help = document.createElement('span')
    help.id = 'press-kit-editor-help'
    help.className = 'press-kit-editor-help'
    help.setAttribute('role', 'tooltip')
    help.textContent =
      'Glisser dans un espace vide pour sélectionner plusieurs éléments. Maj + clic ajoute ou retire un élément ; Maj + glisser dans le vide ajoute une zone. Glisser un élément sélectionné déplace la sélection ; maintenir Maj verrouille le déplacement horizontal ou vertical. Les poignées règlent sa largeur et sa hauteur ; Maj + glisser un coin conserve les proportions. Flèches : 1 px, Maj : 10 px. Ctrl/Cmd + A : tout sélectionner. Échap annule le geste ou vide la sélection. Modifications conservées sur cette page. Image traitée dans ce navigateur, sans envoi.'
    summary.setAttribute('aria-label', label.textContent)
    summary.setAttribute('aria-describedby', help.id)
    summary.append(label, help)
    controls.closest('dialog')?.addEventListener(
      'keydown',
      event => {
        if (event.key !== 'Escape' || getComputedStyle(help).visibility !== 'visible') return
        event.preventDefault()
        event.stopPropagation()
        summary.dataset.helpDismissed = ''
      },
      { signal, capture: true },
    )
    const resetHelp = () => delete summary.dataset.helpDismissed
    summary.addEventListener('pointerleave', resetHelp, { signal })
    summary.addEventListener('blur', resetHelp, { signal })
    this.file.type = 'file'
    this.file.accept = 'image/png,image/jpeg,image/webp,image/avif,image/gif'
    this.file.hidden = true
    this.file.addEventListener(
      'change',
      () => {
        this.importImage(this.file.files?.[0])
        this.file.value = ''
      },
      { signal },
    )
    this.details.append(summary, this.controls, this.file)
    controls.append(this.details)
    this.overlay.className = 'press-kit-editor-overlay'
    this.overlay.setAttribute('aria-label', 'éléments de la carte')
    this.overlay.setAttribute('role', 'group')
    this.overlay.tabIndex = 0
    this.overlay.dataset.siteCursorNative = ''
    this.overlay.hidden = true
    canvas.parentElement?.append(this.overlay)
    this.rulers.className = 'press-kit-editor-rulers'
    this.rulers.hidden = true
    this.rulers.setAttribute('aria-hidden', 'true')
    canvas.parentElement?.append(this.rulers)
    this.details.addEventListener(
      'toggle',
      () => {
        if (this.details.open && !this.enabled) {
          this.enabled = true
          this.requestScene()
        }
        if (!this.details.open) this.cancelGesture()
        this.update()
      },
      { signal },
    )
    this.overlay.addEventListener('keydown', event => this.keydown(event), { signal })
    this.overlay.addEventListener(
      'keyup',
      event => {
        if (event.key === 'Shift' && this.gesture?.kind === 'transform')
          this.transformGesture(this.gesture, false)
      },
      { signal },
    )
    this.details.addEventListener('keydown', event => this.keydown(event), { signal })
    this.overlay.addEventListener('pointerdown', event => this.pointerdown(event), { signal })
    this.overlay.addEventListener('pointermove', event => this.pointermove(event), { signal })
    this.overlay.addEventListener(
      'pointerleave',
      () => {
        this.pointer = undefined
        this.positionOverlay()
      },
      { signal },
    )
    this.overlay.addEventListener('pointerup', event => this.finishGesture(event), { signal })
    const cancelPointer = (event: PointerEvent) => {
      if (this.gesture?.pointer === event.pointerId) this.cancelGesture()
    }
    this.overlay.addEventListener('pointercancel', cancelPointer, { signal })
    this.overlay.addEventListener('lostpointercapture', cancelPointer, { signal })
    this.observer = new ResizeObserver(() => this.positionOverlay())
    this.observer.observe(canvas)
    if (canvas.parentElement) this.observer.observe(canvas.parentElement)
    signal.addEventListener(
      'abort',
      () => {
        this.revision++
        cancelAnimationFrame(this.frame)
        this.observer.disconnect()
        render(null, this.controls)
        render(null, this.overlay)
        render(null, this.rulers)
        this.details.remove()
        this.overlay.remove()
        this.rulers.remove()
        canvas.parentElement?.removeAttribute('data-editor-rulers')
        this.scene = undefined
        this.rasters.clear()
        this.drafts.clear()
        this.draft = { value: empty(), past: [], future: [] }
      },
      { once: true },
    )
    this.update()
  }

  get active() {
    return this.enabled
  }

  get title() {
    return this.draft.value.title
  }

  get metricGap() {
    return this.draft.value.metricGap
  }

  context(key: string, formatKey: string, available: boolean) {
    this.available = available
    if (key !== this.key) {
      this.cancelGesture()
      this.revision++
      this.loading = false
      this.error = ''
      this.key = key
      this.draft = this.drafts.get(key) ?? { value: empty(), past: [], future: [] }
      this.drafts.set(key, this.draft)
      this.selected.clear()
      this.scene = undefined
    }
    if (formatKey !== this.formatKey) {
      this.cancelGesture()
      this.formatKey = formatKey
      this.selected.clear()
      this.scene = undefined
    }
    this.overlay.hidden = true
    this.rulers.hidden = true
    if (!available) this.canvas.parentElement?.removeAttribute('data-editor-rulers')
  }

  setScene(scene: CardScene) {
    this.cancelGesture()
    this.scene = scene
    this.rasters.clear()
    this.compose()
    this.update()
  }

  private get composition(): Composition {
    return (this.draft.value.formats[this.formatKey] ??= { placements: {}, order: [] })
  }

  private placement(layer: CardLayer): Placement {
    return (
      this.composition.placements[layer.id] ?? {
        x: layer.x,
        y: layer.y,
        scaleX: 1,
        scaleY: 1,
        hidden: false,
      }
    )
  }

  private selection() {
    return this.layers()
      .filter(layer => this.selected.has(layer.id))
      .map(layer => ({ layer, placement: { ...this.placement(layer) } }))
  }

  private bounds(items: LayerPlacement[]): LayerBounds | undefined {
    if (!items.length) return
    const x = Math.min(...items.map(({ placement }) => placement.x))
    const y = Math.min(...items.map(({ placement }) => placement.y))
    return {
      x,
      y,
      width:
        Math.max(
          ...items.map(({ layer, placement }) => placement.x + layer.width * placement.scaleX),
        ) - x,
      height:
        Math.max(
          ...items.map(({ layer, placement }) => placement.y + layer.height * placement.scaleY),
        ) - y,
    }
  }

  private resizeLimits(items: LayerPlacement[]) {
    return {
      minX: Math.max(
        ...items.map(
          ({ layer, placement }) =>
            Math.max(0.1, (layer.renderer?.minWidth ?? 0) / layer.width) / placement.scaleX,
        ),
      ),
      minY: Math.max(
        ...items.map(
          ({ layer, placement }) =>
            Math.max(0.1, (layer.renderer?.minHeight ?? 0) / layer.height) / placement.scaleY,
        ),
      ),
      maxX: Math.min(...items.map(({ placement }) => 5 / placement.scaleX)),
      maxY: Math.min(...items.map(({ placement }) => 5 / placement.scaleY)),
    }
  }

  private resizedBounds(
    items: LayerPlacement[],
    bounds: LayerBounds,
    dx: number,
    dy: number,
    axis: ResizeAxis,
    lock = false,
  ) {
    const { minX, minY, maxX, maxY } = this.resizeLimits(items)
    const { width, height } = bounds
    if (axis === 'xy' && lock) {
      const factor = clamp(
        1 + (dx * width + dy * height) / (width ** 2 + height ** 2),
        Math.max(minX, minY),
        Math.min(maxX, maxY),
      )
      return { ...bounds, width: width * factor, height: height * factor }
    }
    return {
      ...bounds,
      width: axis === 'y' ? width : width * clamp(1 + dx / width, minX, maxX),
      height: axis === 'x' ? height : height * clamp(1 + dy / height, minY, maxY),
    }
  }

  private placeSelection(
    composition: Composition,
    items: LayerPlacement[],
    before: LayerBounds,
    after: LayerBounds,
  ) {
    const sx = after.width / before.width,
      sy = after.height / before.height
    for (const { layer, placement } of items)
      composition.placements[layer.id] = {
        ...placement,
        x: after.x + (placement.x - before.x) * sx,
        y: after.y + (placement.y - before.y) * sy,
        scaleX: placement.scaleX * sx,
        scaleY: placement.scaleY * sy,
      }
  }

  private editBounds(after: LayerBounds) {
    const items = this.selection(),
      before = this.bounds(items)
    if (before) this.edit(layout => this.placeSelection(layout, items, before, after))
  }

  private selectLayer(id: string, additive: boolean) {
    if (additive) {
      if (this.selected.has(id)) this.selected.delete(id)
      else this.selected.add(id)
    } else if (!this.selected.has(id)) this.selected = new Set([id])
  }

  private layers() {
    const layers = this.scene?.layers ?? []
    const order = this.composition.order
    return [...layers].sort((a, b) => {
      const ai = order.indexOf(a.id),
        bi = order.indexOf(b.id)
      return (
        (ai < 0 ? layers.indexOf(a) : layers.length + ai) -
        (bi < 0 ? layers.indexOf(b) : layers.length + bi)
      )
    })
  }

  private remember(before: Layout) {
    this.draft.past.push(before)
    if (this.draft.past.length > 30) this.draft.past.shift()
    this.draft.future = []
  }

  private edit(mutate: (composition: Composition, document: Layout) => void) {
    const before = copy(this.draft.value)
    mutate(this.composition, this.draft.value)
    this.remember(before)
    if (before.metricGap !== this.metricGap || before.title !== this.title) this.requestScene()
    else this.queue()
    this.update()
  }

  private history(redo: boolean) {
    this.cancelGesture()
    const from = redo ? this.draft.future : this.draft.past
    const to = redo ? this.draft.past : this.draft.future
    const value = from.pop()
    if (!value) return
    const before = copy(this.draft.value)
    to.push(before)
    this.draft.value = value
    if (before.metricGap !== this.metricGap || before.title !== this.title) this.requestScene()
    else this.queue()
    this.update()
  }

  private queue() {
    this.changed()
    if (!this.frame)
      this.frame = requestAnimationFrame(() => {
        this.frame = 0
        this.compose()
      })
  }

  private layerImage(
    layer: CardLayer,
    width: number,
    height: number,
  ): Pick<LayerRaster, 'image' | 'width' | 'height'> {
    const renderer = layer.renderer
    if (!renderer || (width === layer.width && height === layer.height))
      return { image: layer.image, width, height }
    let raster = this.rasters.get(layer)
    if (!raster) {
      raster = { width: layer.width, height: layer.height, image: layer.image }
      this.rasters.set(layer, raster)
    }
    if (
      !raster.pending &&
      (raster.width !== width || raster.height !== height) &&
      (raster.failed?.width !== width || raster.failed?.height !== height)
    ) {
      const target = raster
      const scene = this.scene
      target.failed = undefined
      target.pending = (async () => {
        const image = await renderer.render(width, height)
        if (this.signal.aborted || this.scene !== scene) return
        target.image = image
        target.width = width
        target.height = height
      })()
        .catch(() => {
          target.failed = { width, height }
        })
        .finally(() => {
          target.pending = undefined
          if (this.signal.aborted || this.scene !== scene) return
          this.changed()
          // Read the latest placement after decoding; one pending render per chart coalesces drags.
          this.compose()
          this.update()
        })
    }
    return raster
  }

  private get redrawFailed() {
    return this.layers().some(layer => {
      const p = this.placement(layer)
      const failed = this.rasters.get(layer)?.failed
      return (
        !p.hidden &&
        failed?.width === layer.width * p.scaleX &&
        failed?.height === layer.height * p.scaleY
      )
    })
  }

  private compose() {
    const scene = this.scene
    if (!scene || !this.available || this.signal.aborted) return
    const ctx = this.canvas.getContext('2d')
    if (!ctx) return
    const width = Math.round(scene.width * scene.scale),
      height = Math.round(scene.height * scene.scale)
    if (this.canvas.width !== width) this.canvas.width = width
    if (this.canvas.height !== height) this.canvas.height = height
    ctx.setTransform(scene.scale, 0, 0, scene.scale, 0, 0)
    ctx.clearRect(0, 0, scene.width, scene.height)
    if (scene.background) {
      ctx.fillStyle = scene.background
      ctx.fillRect(0, 0, scene.width, scene.height)
    }
    const bg = this.draft.value.background
    if (bg) {
      const ratio =
        (bg.fit === 'cover' ? Math.max : Math.min)(
          scene.width / bg.image.naturalWidth,
          scene.height / bg.image.naturalHeight,
        ) * bg.zoom
      const width = bg.image.naturalWidth * ratio,
        height = bg.image.naturalHeight * ratio
      ctx.globalAlpha = bg.opacity
      ctx.drawImage(
        bg.image,
        (scene.width - width) * bg.x,
        (scene.height - height) * bg.y,
        width,
        height,
      )
      ctx.globalAlpha = 1
    }
    for (const layer of this.layers()) {
      const p = this.placement(layer)
      if (!p.hidden) {
        const width = layer.width * p.scaleX,
          height = layer.height * p.scaleY
        const raster = this.layerImage(layer, width, height)
        ctx.save()
        ctx.beginPath()
        ctx.rect(p.x, p.y, width, height)
        ctx.clip()
        ctx.drawImage(raster.image, p.x, p.y, raster.width, raster.height)
        ctx.restore()
      }
    }
    this.details.setAttribute(
      'aria-busy',
      String([...this.rasters.values()].some(raster => raster.pending)),
    )
    this.positionOverlay()
  }

  async flush() {
    await this.pendingImage
    while (!this.signal.aborted) {
      if (this.frame) {
        cancelAnimationFrame(this.frame)
        this.frame = 0
      }
      this.compose()
      const pending = [...this.rasters.values()].flatMap(raster =>
        raster.pending ? [raster.pending] : [],
      )
      if (!pending.length) return !this.redrawFailed
      await Promise.all(pending)
    }
    return false
  }

  private positionOverlay() {
    const scene = this.scene,
      parent = this.canvas.parentElement
    if (!scene || !parent) return
    const rect = this.canvas.getBoundingClientRect(),
      host = parent.getBoundingClientRect()
    this.overlay.style.cssText = `left:${rect.left - host.left}px;top:${rect.top - host.top}px;width:${rect.width}px;height:${rect.height}px`
    const position = (node: HTMLElement, bounds: LayerBounds) => {
      node.style.cssText = `left:${(bounds.x / scene.width) * 100}%;top:${(bounds.y / scene.height) * 100}%;width:${(bounds.width / scene.width) * 100}%;height:${(bounds.height / scene.height) * 100}%`
      const handleSize = window.matchMedia('(pointer: coarse)').matches ? 44 : 24
      node.toggleAttribute(
        'data-compact-x',
        (bounds.height * rect.height) / scene.height < handleSize * 2,
      )
      node.toggleAttribute(
        'data-compact-y',
        (bounds.width * rect.width) / scene.width < handleSize * 2,
      )
    }
    for (const node of this.overlay.querySelectorAll<HTMLElement>('[data-editor-layer]')) {
      const layer = scene.layers.find(item => item.id === node.dataset.editorLayer)
      if (!layer) continue
      const p = this.placement(layer)
      position(node, {
        x: p.x,
        y: p.y,
        width: layer.width * p.scaleX,
        height: layer.height * p.scaleY,
      })
      node.dataset.x = String(p.x)
      node.dataset.y = String(p.y)
      node.dataset.scaleX = String(p.scaleX)
      node.dataset.scaleY = String(p.scaleY)
    }
    const group = this.overlay.querySelector<HTMLElement>('[data-editor-selection]'),
      bounds = this.bounds(this.selection())
    if (group && bounds) position(group, bounds)
    const marquee = this.overlay.querySelector<HTMLElement>('.press-kit-editor-marquee')
    if (marquee && this.gesture?.kind === 'marquee') position(marquee, this.gesture.bounds)
    if (!this.rulers.hidden) {
      this.rulers.style.cssText = `left:${rect.left - host.left - 24}px;top:${rect.top - host.top - 24}px`
      paintRulers(
        this.rulers,
        rect.width,
        rect.height,
        scene.width,
        scene.height,
        bounds,
        this.pointer,
      )
      const coordinates = this.controls.querySelector('output')
      if (coordinates)
        coordinates.textContent = bounds
          ? `x ${Math.round(bounds.x)} · y ${Math.round(bounds.y)} px`
          : this.pointer
            ? `x ${Math.round(this.pointer.x)} · y ${Math.round(this.pointer.y)} px`
            : `${scene.width} × ${scene.height} px`
    }
  }

  private pointerPosition(event: PointerEvent) {
    const scene = this.scene
    if (!scene) return
    const rect = this.canvas.getBoundingClientRect()
    return {
      x: ((event.clientX - rect.left) * scene.width) / rect.width,
      y: ((event.clientY - rect.top) * scene.height) / rect.height,
    }
  }

  private pointerdown(event: PointerEvent) {
    if (event.button !== 0 || this.gesture) return
    const point = this.pointerPosition(event)
    if (!point || !(event.target instanceof Element)) return
    const target = event.target.closest<HTMLElement>('[data-editor-layer]')
    const layer = this.scene?.layers.find(item => item.id === target?.dataset.editorLayer)
    const group = event.target.closest('[data-editor-selection]')
    const selection = [...this.selected]
    const additive = event.shiftKey || event.metaKey || event.ctrlKey
    event.preventDefault()
    if (!layer && !group) {
      if (!additive) this.selected.clear()
      this.gesture = {
        kind: 'marquee',
        pointer: event.pointerId,
        ...point,
        selection,
        additive,
        bounds: { ...point, width: 0, height: 0 },
      }
      this.overlay.dataset.gesture = 'select'
      this.overlay.setPointerCapture(event.pointerId)
      this.overlay.focus({ preventScroll: true })
      this.update()
      return
    }
    const axis = event.target.closest<HTMLElement>('[data-editor-resize]')?.dataset.editorResize
    const resize = axis === 'x' || axis === 'y' || axis === 'xy' ? axis : null
    const toggleOnClick =
      layer &&
      !resize &&
      event.shiftKey &&
      !event.metaKey &&
      !event.ctrlKey &&
      this.selected.has(layer.id)
        ? layer.id
        : undefined
    if (layer && !resize && !toggleOnClick) this.selectLayer(layer.id, additive)
    if (layer && !this.selected.has(layer.id)) {
      this.update()
      this.overlay
        .querySelector<HTMLElement>(
          `[data-editor-layer="${CSS.escape(layer.id)}"] .press-kit-editor-select`,
        )
        ?.focus({ preventScroll: true })
      return
    }
    const initial = this.selection(),
      bounds = this.bounds(initial)
    if (!bounds) return
    this.gesture = {
      kind: 'transform',
      pointer: event.pointerId,
      ...point,
      selection,
      initial,
      bounds,
      before: copy(this.draft.value),
      resize,
      point,
      moved: false,
      toggleOnClick,
    }
    this.overlay.dataset.gesture = resize ? `resize-${resize}` : 'move'
    this.overlay.setPointerCapture(event.pointerId)
    this.update()
    const owner = group
      ? '[data-editor-selection]'
      : `[data-editor-layer="${CSS.escape(layer?.id ?? '')}"]`
    this.overlay
      .querySelector<HTMLElement>(
        `${owner} ${resize ? `[data-editor-resize="${resize}"]` : group ? '.press-kit-editor-group-move' : '.press-kit-editor-select'}`,
      )
      ?.focus({ preventScroll: true })
  }

  private pointermove(event: PointerEvent) {
    const gesture = this.gesture,
      scene = this.scene,
      point = this.pointerPosition(event)
    this.pointer = point
    if (!gesture) {
      if (this.showRulers) this.positionOverlay()
      return
    }
    if (event.pointerId !== gesture.pointer || !scene || !point) return
    const dx = point.x - gesture.x,
      dy = point.y - gesture.y
    if (gesture.kind === 'marquee') {
      const x = clamp(point.x, 0, scene.width),
        y = clamp(point.y, 0, scene.height)
      const box = {
        x: Math.min(gesture.x, x),
        y: Math.min(gesture.y, y),
        width: Math.abs(x - gesture.x),
        height: Math.abs(y - gesture.y),
      }
      gesture.bounds = box
      const selected = new Set(gesture.additive ? gesture.selection : [])
      const rect = this.canvas.getBoundingClientRect()
      if (Math.hypot((dx * rect.width) / scene.width, (dy * rect.height) / scene.height) >= 3)
        for (const layer of this.layers()) {
          const p = this.placement(layer)
          if (
            !p.hidden &&
            p.x < box.x + box.width &&
            p.x + layer.width * p.scaleX > box.x &&
            p.y < box.y + box.height &&
            p.y + layer.height * p.scaleY > box.y
          )
            selected.add(layer.id)
        }
      if (
        selected.size !== this.selected.size ||
        [...selected].some(id => !this.selected.has(id))
      ) {
        this.selected = selected
        this.update()
      } else this.positionOverlay()
      return
    }
    gesture.point = point
    const rect = this.canvas.getBoundingClientRect()
    if (Math.hypot((dx * rect.width) / scene.width, (dy * rect.height) / scene.height) >= 3)
      gesture.moved = true
    this.transformGesture(gesture, event.shiftKey)
  }

  private transformGesture(gesture: Extract<Gesture, { kind: 'transform' }>, shift: boolean) {
    let dx = gesture.point.x - gesture.x,
      dy = gesture.point.y - gesture.y
    if (!gesture.resize) {
      if (!shift) gesture.moveAxis = undefined
      else if (gesture.moved) gesture.moveAxis ??= Math.abs(dx) >= Math.abs(dy) ? 'x' : 'y'
      if (gesture.moveAxis === 'x') dy = 0
      if (gesture.moveAxis === 'y') dx = 0
      if (!gesture.moved) return
    }
    const after = gesture.resize
      ? this.resizedBounds(gesture.initial, gesture.bounds, dx, dy, gesture.resize, shift)
      : { ...gesture.bounds, x: gesture.bounds.x + dx, y: gesture.bounds.y + dy }
    this.placeSelection(this.composition, gesture.initial, gesture.bounds, after)
    this.queue()
    this.positionOverlay()
  }

  private finishGesture(event: PointerEvent) {
    const gesture = this.gesture
    if (!gesture || event.pointerId !== gesture.pointer) return
    this.gesture = undefined
    delete this.overlay.dataset.gesture
    if (gesture.kind === 'transform') {
      if (gesture.toggleOnClick && !gesture.moved) this.selected.delete(gesture.toggleOnClick)
      const changed = gesture.initial.some(({ layer, placement }) => {
        const current = this.placement(layer)
        return (
          Math.abs(current.x - placement.x) > 1e-8 ||
          Math.abs(current.y - placement.y) > 1e-8 ||
          Math.abs(current.scaleX - placement.scaleX) > 1e-8 ||
          Math.abs(current.scaleY - placement.scaleY) > 1e-8
        )
      })
      if (changed) this.remember(gesture.before)
      else this.draft.value = gesture.before
    }
    if (this.overlay.hasPointerCapture(event.pointerId))
      this.overlay.releasePointerCapture(event.pointerId)
    this.update()
  }

  private cancelGesture() {
    const gesture = this.gesture
    if (!gesture) return
    this.gesture = undefined
    delete this.overlay.dataset.gesture
    this.selected = new Set(gesture.selection)
    if (gesture.kind === 'transform') this.draft.value = gesture.before
    if (this.overlay.hasPointerCapture(gesture.pointer))
      this.overlay.releasePointerCapture(gesture.pointer)
    this.queue()
    this.update()
  }

  private keydown(event: KeyboardEvent) {
    if (
      event.isComposing ||
      (event.target instanceof Element && event.target.closest('input, textarea, .g-select'))
    ) {
      event.stopPropagation()
      return
    }
    if (event.key === 'Escape') {
      if (!this.gesture && !this.selected.size) return
      event.preventDefault()
      event.stopPropagation()
      if (this.gesture) this.cancelGesture()
      else {
        this.selected.clear()
        this.update()
      }
      return
    }
    event.stopPropagation()
    if (event.key === 'Shift' && this.gesture?.kind === 'transform') {
      this.transformGesture(this.gesture, true)
      return
    }
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'a') {
      event.preventDefault()
      this.selected = new Set(
        this.layers()
          .filter(layer => !this.placement(layer).hidden)
          .map(layer => layer.id),
      )
      this.update()
      return
    }
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'z') {
      event.preventDefault()
      this.history(event.shiftKey)
      return
    }
    const items = this.selection()
    if (!items.length) return
    const step = event.shiftKey ? 10 : 1
    if (
      ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Delete', 'Backspace'].includes(event.key)
    ) {
      event.preventDefault()
      this.edit(layout => {
        for (const { layer, placement } of items) {
          const p = { ...placement }
          if (event.key === 'ArrowLeft') p.x -= step
          if (event.key === 'ArrowRight') p.x += step
          if (event.key === 'ArrowUp') p.y -= step
          if (event.key === 'ArrowDown') p.y += step
          if (event.key === 'Delete' || event.key === 'Backspace') p.hidden = true
          layout.placements[layer.id] = p
        }
      })
    }
  }

  private resizeKey(event: KeyboardEvent, axis: ResizeAxis) {
    if (!['ArrowUp', 'ArrowRight', 'ArrowDown', 'ArrowLeft'].includes(event.key)) return
    event.preventDefault()
    event.stopPropagation()
    const items = this.selection(),
      bounds = this.bounds(items)
    if (!bounds) return
    const step = event.shiftKey ? 10 : 1
    const dx = event.key === 'ArrowLeft' ? -step : event.key === 'ArrowRight' ? step : 0
    const dy = event.key === 'ArrowUp' ? -step : event.key === 'ArrowDown' ? step : 0
    this.editBounds(this.resizedBounds(items, bounds, dx, dy, axis))
  }

  private importImage(file?: File) {
    if (!file) return
    const revision = ++this.revision
    this.loading = true
    this.error = ''
    this.update()
    this.pendingImage = (async () => {
      const image = await readCardBackground(file)
      if (this.signal.aborted || revision !== this.revision) return
      this.edit((_, document) => {
        document.background = {
          image,
          name: file.name,
          fit: 'cover',
          zoom: 1,
          x: 0.5,
          y: 0.5,
          opacity: 1,
        }
      })
    })()
      .catch(error => {
        if (revision === this.revision)
          this.error = error instanceof Error ? error.message : 'image illisible'
      })
      .finally(() => {
        if (revision !== this.revision || this.signal.aborted) return
        this.loading = false
        this.update()
      })
  }

  private update() {
    if (this.signal.aborted) return
    this.overlay.hidden = !this.details.open || !this.available || !this.scene
    this.rulers.hidden = this.overlay.hidden || !this.showRulers
    this.canvas.parentElement?.toggleAttribute('data-editor-rulers', !this.rulers.hidden)
    const layers = this.layers()
    const selected = this.selection()
    const bounds = this.bounds(selected)
    const limits = selected.length ? this.resizeLimits(selected) : undefined
    const label = selected.length === 1 ? selected[0].layer.label : `${selected.length} éléments`
    const allHidden = selected.every(({ placement }) => placement.hidden)
    const bg = this.draft.value.background
    const number = (
      label: string,
      value: number,
      min: number,
      max: number,
      apply: (value: number) => void,
    ) => (
      <label>
        {label}
        <input
          type="number"
          value={Math.round(value * 10) / 10}
          min={min}
          max={max}
          step="1"
          onChange={event => {
            const value = event.currentTarget.valueAsNumber
            if (Number.isFinite(value)) apply(clamp(value, min, max))
          }}
        />
      </label>
    )
    const handles = (name: string) =>
      resizeHandles.map(({ axis, label }) => (
        <button
          key={axis}
          type="button"
          class="press-kit-editor-resize"
          data-editor-resize={axis}
          aria-label={`redimensionner ${label} : ${name}`}
          onKeyDown={event => this.resizeKey(event, axis)}
        />
      ))
    render(
      <>
        <div class="press-kit-segments press-kit-editor-toolbar" role="group" aria-label="édition">
          <button
            type="button"
            class="press-kit-segment"
            aria-label="annuler la modification"
            disabled={!this.draft.past.length}
            onClick={() => this.history(false)}
          >
            annuler
          </button>
          <button
            type="button"
            class="press-kit-segment"
            aria-label="rétablir la modification"
            disabled={!this.draft.future.length}
            onClick={() => this.history(true)}
          >
            rétablir
          </button>
          <button
            type="button"
            class="press-kit-segment"
            onClick={() =>
              this.edit((layout, document) => {
                layout.placements = {}
                layout.order = []
                document.background = undefined
                document.metricGap = 0
                document.title = undefined
              })
            }
          >
            réinitialiser
          </button>
          <button
            type="button"
            class="press-kit-segment"
            aria-label={bg ? 'changer l’image de fond' : 'ajouter une image de fond'}
            title={bg ? 'changer l’image de fond' : 'ajouter une image de fond'}
            disabled={this.loading}
            onClick={() => this.file.click()}
          >
            fond…
          </button>
        </div>
        <div class="press-kit-editor-view">
          <button
            type="button"
            class="g-button"
            aria-pressed={this.showRulers}
            onClick={() => {
              this.showRulers = !this.showRulers
              this.update()
            }}
          >
            règles
          </button>
          {this.showRulers && <output aria-label="position sur la carte" />}
        </div>
        <label class="press-kit-editor-field">
          titre de l’image
          <input
            type="text"
            value={this.title ?? ''}
            placeholder={this.defaultTitle()}
            maxLength={240}
            onChange={event => {
              const title = event.currentTarget.value.trim() || undefined
              if (title !== this.title)
                this.edit((_, document) => {
                  document.title = title
                })
            }}
          />
        </label>
        {this.loading && <p role="status">chargement de l’image…</p>}
        {this.error && <p role="alert">{this.error}</p>}
        {this.redrawFailed && <p role="alert">impossible de redimensionner le graphique</p>}
        {this.formatKey.endsWith(':chips') &&
          number('écart des cellules (px)', this.metricGap, 0, 24, value =>
            this.edit((_, document) => {
              document.metricGap = value
            }),
          )}
        {bg && (
          <fieldset>
            <legend>{bg.name}</legend>
            <div class="press-kit-editor-field">
              <span aria-hidden="true">ajustement</span>
              <EditorSelect
                id="press-kit-editor-fit"
                label="ajustement"
                options={[
                  { value: 'cover', label: 'remplir' },
                  { value: 'contain', label: 'contenir' },
                ]}
                selected={bg.fit}
                onSelect={fit => {
                  if (fit === 'cover' || fit === 'contain')
                    this.edit((_, document) => {
                      if (document.background) document.background.fit = fit
                    })
                }}
              />
            </div>
            <div class="press-kit-editor-fields">
              {number('zoom (%)', bg.zoom * 100, 100, 400, value =>
                this.edit((_, document) => {
                  if (document.background) document.background.zoom = value / 100
                }),
              )}
              {number('opacité (%)', bg.opacity * 100, 0, 100, value =>
                this.edit((_, document) => {
                  if (document.background) document.background.opacity = value / 100
                }),
              )}
              {number('cadrage x (%)', bg.x * 100, 0, 100, value =>
                this.edit((_, document) => {
                  if (document.background) document.background.x = value / 100
                }),
              )}
              {number('cadrage y (%)', bg.y * 100, 0, 100, value =>
                this.edit((_, document) => {
                  if (document.background) document.background.y = value / 100
                }),
              )}
            </div>
            <button
              type="button"
              class="g-button"
              onClick={() =>
                this.edit((_, document) => {
                  document.background = undefined
                })
              }
            >
              retirer l’image
            </button>
          </fieldset>
        )}
        <div class="press-kit-editor-field">
          <span aria-hidden="true">élément</span>
          <EditorSelect
            id="press-kit-editor-layer"
            label="élément"
            options={[
              {
                value: '',
                label:
                  selected.length > 1
                    ? `${selected.length} éléments sélectionnés`
                    : 'sélectionner…',
              },
              ...layers.map(layer => ({
                value: layer.id,
                label: `${layer.label}${this.placement(layer).hidden ? ' (masqué)' : ''}`,
              })),
            ]}
            selected={selected.length === 1 ? selected[0].layer.id : ''}
            onSelect={value => {
              this.selected = new Set(value ? [value] : [])
              this.update()
            }}
          />
        </div>
        {bounds && limits && (
          <fieldset>
            <legend>{label}</legend>
            <div class="press-kit-editor-fields">
              {number(
                'x (px)',
                bounds.x,
                -(this.scene?.width ?? 1200),
                this.scene?.width ?? 1200,
                value => this.editBounds({ ...bounds, x: value }),
              )}
              {number(
                'y (px)',
                bounds.y,
                -(this.scene?.height ?? 1920),
                this.scene?.height ?? 1920,
                value => this.editBounds({ ...bounds, y: value }),
              )}
              {number(
                'largeur (px)',
                bounds.width,
                bounds.width * limits.minX,
                bounds.width * limits.maxX,
                value => this.editBounds({ ...bounds, width: value }),
              )}
              {number(
                'hauteur (px)',
                bounds.height,
                bounds.height * limits.minY,
                bounds.height * limits.maxY,
                value => this.editBounds({ ...bounds, height: value }),
              )}
            </div>
            <div class="press-kit-editor-actions">
              <button
                type="button"
                class="g-button"
                aria-pressed={!allHidden}
                onClick={() =>
                  this.edit(layout => {
                    for (const { layer, placement } of selected)
                      layout.placements[layer.id] = { ...placement, hidden: !allHidden }
                  })
                }
              >
                {allHidden ? 'afficher' : 'masquer'}
              </button>
              <button
                type="button"
                class="g-button"
                onClick={() =>
                  this.edit(layout => {
                    layout.order = [
                      ...layers
                        .filter(layer => !this.selected.has(layer.id))
                        .map(layer => layer.id),
                      ...selected.map(({ layer }) => layer.id),
                    ]
                  })
                }
              >
                devant
              </button>
              <button
                type="button"
                class="g-button"
                onClick={() =>
                  this.edit(layout => {
                    layout.order = [
                      ...selected.map(({ layer }) => layer.id),
                      ...layers
                        .filter(layer => !this.selected.has(layer.id))
                        .map(layer => layer.id),
                    ]
                  })
                }
              >
                derrière
              </button>
            </div>
          </fieldset>
        )}
      </>,
      this.controls,
    )
    render(
      <>
        {layers
          .filter(layer => !this.placement(layer).hidden)
          .map(layer => (
            <div
              key={layer.id}
              class={`press-kit-editor-layer${this.selected.has(layer.id) ? ' is-selected' : ''}`}
              data-editor-layer={layer.id}
            >
              <button
                type="button"
                class="press-kit-editor-select"
                aria-label={`déplacer ${layer.label}`}
                aria-pressed={this.selected.has(layer.id)}
                onClick={event => {
                  // Pointer selection happens before dragging; keyboard and AT activate via click.
                  if (event.detail) return
                  this.selectLayer(layer.id, event.shiftKey || event.metaKey || event.ctrlKey)
                  this.update()
                }}
              />
              {selected.length === 1 && this.selected.has(layer.id) && handles(layer.label)}
            </div>
          ))}
        {selected.length > 1 && !allHidden && (
          <div
            class="press-kit-editor-selection"
            data-editor-selection=""
            role="group"
            aria-label={`sélection de ${label}`}
          >
            <button
              type="button"
              class="press-kit-editor-group-move"
              aria-label={`déplacer la sélection de ${label}`}
            >
              {label}
            </button>
            {handles(`sélection de ${label}`)}
          </div>
        )}
        <div
          class="press-kit-editor-marquee"
          hidden={this.gesture?.kind !== 'marquee'}
          aria-hidden="true"
        />
      </>,
      this.overlay,
    )
    this.positionOverlay()
  }
}
