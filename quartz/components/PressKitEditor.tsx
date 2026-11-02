import { render } from 'preact'
import type { CardLayer, CardScene } from './scripts/press-kit-layers'
import { updateEditorView } from './PressKitEditor/controls-view'
import {
  cancelGesture,
  finishGesture,
  keydown,
  pointerdown,
  pointermove,
  transformGesture,
} from './PressKitEditor/gestures'
import {
  composition,
  empty,
  placement,
  type Draft,
  type EditorState,
  type LayerRaster,
} from './PressKitEditor/layout'
import { compose, flushStage, importImage, positionOverlay } from './PressKitEditor/stage'

export class PressKitEditor {
  private state: EditorState

  constructor(
    canvas: HTMLCanvasElement,
    controls: HTMLElement,
    signal: AbortSignal,
    changed: () => void,
    requestScene: () => void,
    defaultTitle: () => string,
  ) {
    const state: EditorState = {
      canvas,
      signal,
      changed,
      requestScene,
      defaultTitle,
      syncView: () => updateEditorView(state),
      details: document.createElement('details'),
      controls: document.createElement('div'),
      overlay: document.createElement('div'),
      rulers: document.createElement('div'),
      showRulers: false,
      rulerToggle:
        controls
          .closest('dialog')
          ?.querySelector<HTMLButtonElement>('button[data-editor-rulers]') ?? null,
      file: document.createElement('input'),
      drafts: new Map<string, Draft>(),
      draft: { value: empty(), past: [], future: [] },
      rasters: new Map<CardLayer, LayerRaster>(),
      tints: new Map(),
      selected: new Set<string>(),
      frame: 0,
      revision: 0,
      key: '',
      formatKey: '',
      railAnchors: new Map<string, { right: number; bottom: number }>(),
      available: false,
      enabled: false,
      loading: false,
      error: '',
      observer: new ResizeObserver(() => positionOverlay(state)),
    }
    this.state = state
    state.rulerToggle?.addEventListener(
      'click',
      () => {
        if (!state.available) return
        state.showRulers = state.rulers.hidden !== false
        if (state.showRulers) state.details.open = true
        state.syncView()
      },
      { signal },
    )
    state.details.className = 'press-kit-editor'
    state.details.dataset.for = 'image'
    const summary = document.createElement('summary')
    const label = document.createElement('span')
    label.className = 'press-kit-editor-label'
    label.textContent = 'réglages avancés'
    const help = document.createElement('span')
    help.id = 'press-kit-editor-help'
    help.className = 'press-kit-editor-help'
    help.setAttribute('role', 'tooltip')
    help.textContent =
      'Glisser dans un espace vide pour sélectionner plusieurs éléments. Maj + clic ajoute ou retire un élément ; Maj + glisser dans le vide ajoute une zone. Glisser un élément sélectionné déplace la sélection ; maintenir Maj verrouille le déplacement horizontal ou vertical. Les poignées règlent sa largeur et sa hauteur ; une sélection de mesures change d’espacement en gardant la taille du texte. Maj + glisser un coin conserve les proportions de la sélection. Flèches : 1 px, Maj : 10 px. Ctrl/Cmd + A : tout sélectionner. Échap annule le geste ou vide la sélection. Modifications conservées sur cette page. Image traitée dans ce navigateur, sans envoi.'
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
    state.file.type = 'file'
    state.file.accept = 'image/png,image/jpeg,image/webp,image/avif,image/gif'
    state.file.hidden = true
    state.file.addEventListener(
      'change',
      () => {
        importImage(state, state.file.files?.[0])
        state.file.value = ''
      },
      { signal },
    )
    state.details.append(summary, state.controls, state.file)
    controls.append(state.details)
    state.overlay.className = 'press-kit-editor-overlay'
    state.overlay.setAttribute('aria-label', 'éléments de la carte')
    state.overlay.setAttribute('role', 'group')
    state.overlay.tabIndex = 0
    state.overlay.dataset.siteCursorNative = ''
    state.overlay.hidden = true
    canvas.parentElement?.append(state.overlay)
    state.rulers.className = 'press-kit-editor-rulers'
    state.rulers.hidden = true
    state.rulers.setAttribute('aria-hidden', 'true')
    canvas.parentElement?.append(state.rulers)
    state.details.addEventListener(
      'toggle',
      () => {
        if (state.details.open && !state.enabled) {
          state.enabled = true
          state.requestScene()
        }
        if (!state.details.open) cancelGesture(state)
        state.syncView()
      },
      { signal },
    )
    state.overlay.addEventListener('keydown', event => keydown(state, event), { signal })
    state.overlay.addEventListener(
      'keyup',
      event => {
        if (event.key === 'Shift' && state.gesture?.kind === 'transform')
          transformGesture(state, state.gesture, false)
      },
      { signal },
    )
    state.details.addEventListener('keydown', event => keydown(state, event), { signal })
    state.overlay.addEventListener('pointerdown', event => pointerdown(state, event), { signal })
    state.overlay.addEventListener('pointermove', event => pointermove(state, event), { signal })
    state.overlay.addEventListener(
      'pointerleave',
      () => {
        state.pointer = undefined
        positionOverlay(state)
      },
      { signal },
    )
    state.overlay.addEventListener('pointerup', event => finishGesture(state, event), { signal })
    const cancelPointer = (event: PointerEvent) => {
      if (state.gesture?.pointer === event.pointerId) cancelGesture(state)
    }
    state.overlay.addEventListener('pointercancel', cancelPointer, { signal })
    state.overlay.addEventListener('lostpointercapture', cancelPointer, { signal })
    state.observer.observe(canvas)
    if (canvas.parentElement) state.observer.observe(canvas.parentElement)
    signal.addEventListener(
      'abort',
      () => {
        state.revision++
        cancelAnimationFrame(state.frame)
        state.observer.disconnect()
        render(null, state.controls)
        render(null, state.overlay)
        render(null, state.rulers)
        state.details.remove()
        state.overlay.remove()
        state.rulers.remove()
        canvas.parentElement?.removeAttribute('data-editor-rulers')
        state.scene = undefined
        state.rasters.clear()
        state.tints.clear()
        state.drafts.clear()
        state.draft = { value: empty(), past: [], future: [] }
      },
      { once: true },
    )
    state.syncView()
  }

  get active() {
    return this.state.enabled
  }

  get title() {
    return this.state.draft.value.title
  }

  get metricGap() {
    return this.state.draft.value.metricGap
  }

  get railTextScale() {
    return this.state.draft.value.railTextScale
  }

  context(key: string, formatKey: string, available: boolean) {
    const { state } = this
    state.available = available
    if (state.rulerToggle) state.rulerToggle.disabled = !available
    if (key !== state.key) {
      cancelGesture(state)
      state.revision++
      state.loading = false
      state.error = ''
      state.key = key
      state.draft = state.drafts.get(key) ?? { value: empty(), past: [], future: [] }
      state.drafts.set(key, state.draft)
      state.selected.clear()
      state.railAnchors.clear()
      state.scene = undefined
    }
    if (formatKey !== state.formatKey) {
      cancelGesture(state)
      state.formatKey = formatKey
      state.selected.clear()
      state.railAnchors.clear()
      state.scene = undefined
    }
    state.overlay.hidden = true
    state.rulers.hidden = true
    if (!available) state.canvas.parentElement?.removeAttribute('data-editor-rulers')
  }

  setScene(scene: CardScene) {
    const { state } = this
    cancelGesture(state)
    state.scene = scene
    // Reflow enlarged text around a manually placed rail's bottom-right corner.
    const comp = composition(state)
    for (const [block, anchor] of state.railAnchors) {
      const blockLayers = scene.layers.filter(
        layer => layer.id.startsWith(`${block}/stat/`) && !placement(state, layer).hidden,
      )
      if (!blockLayers.length) continue
      const right = Math.max(...blockLayers.map(layer => layer.x + layer.width))
      const bottom = Math.max(...blockLayers.map(layer => layer.y + layer.height))
      for (const layer of blockLayers)
        comp.placements[layer.id] = {
          ...placement(state, layer),
          x: layer.x + anchor.right - right,
          y: layer.y + anchor.bottom - bottom,
          scaleX: 1,
          scaleY: 1,
        }
    }
    state.railAnchors.clear()
    state.rasters.clear()
    state.tints.clear()
    compose(state)
    state.syncView()
  }

  flush() {
    return flushStage(this.state)
  }
}
