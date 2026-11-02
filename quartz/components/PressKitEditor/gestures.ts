import {
  bounds,
  clamp,
  composition,
  copy,
  layers,
  placement,
  placeSelection,
  remember,
  resizedBounds,
  selection,
  selectLayer,
  type EditorState,
  type Gesture,
  type ResizeAxis,
} from './layout'
import { edit, editBounds, positionOverlay, queue } from './stage'

export const cancelGesture = (state: EditorState) => {
  const gesture = state.gesture
  if (!gesture) return
  state.gesture = undefined
  delete state.overlay.dataset.gesture
  state.selected = new Set(gesture.selection)
  if (gesture.kind === 'transform') state.draft.value = gesture.before
  if (state.overlay.hasPointerCapture(gesture.pointer))
    state.overlay.releasePointerCapture(gesture.pointer)
  queue(state)
  state.syncView()
}

export const history = (state: EditorState, redo: boolean) => {
  cancelGesture(state)
  state.railAnchors.clear()
  const from = redo ? state.draft.future : state.draft.past
  const to = redo ? state.draft.past : state.draft.future
  const value = from.pop()
  if (!value) return
  const before = copy(state.draft.value)
  to.push(before)
  state.draft.value = value
  if (
    before.metricGap !== value.metricGap ||
    before.railTextScale !== value.railTextScale ||
    before.title !== value.title
  )
    state.requestScene()
  else queue(state)
  state.syncView()
}

const pointerPosition = (state: EditorState, event: PointerEvent) => {
  const scene = state.scene
  if (!scene) return
  const rect = state.canvas.getBoundingClientRect()
  return {
    x: ((event.clientX - rect.left) * scene.width) / rect.width,
    y: ((event.clientY - rect.top) * scene.height) / rect.height,
  }
}

export const transformGesture = (
  state: EditorState,
  gesture: Extract<Gesture, { kind: 'transform' }>,
  shift: boolean,
) => {
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
    ? resizedBounds(state, gesture.initial, gesture.bounds, dx, dy, gesture.resize, shift)
    : { ...gesture.bounds, x: gesture.bounds.x + dx, y: gesture.bounds.y + dy }
  placeSelection(composition(state), gesture.initial, gesture.bounds, after)
  queue(state)
  positionOverlay(state)
}

export const pointerdown = (state: EditorState, event: PointerEvent) => {
  if (event.button !== 0 || state.gesture) return
  const point = pointerPosition(state, event)
  if (!point || !(event.target instanceof Element)) return
  const target = event.target.closest<HTMLElement>('[data-editor-layer]')
  const layer = state.scene?.layers.find(item => item.id === target?.dataset.editorLayer)
  const group = event.target.closest('[data-editor-selection], .press-kit-editor-group-move')
  const sel = [...state.selected]
  const additive = event.shiftKey || event.metaKey || event.ctrlKey
  event.preventDefault()
  if (!layer && !group) {
    if (!additive) state.selected.clear()
    state.gesture = {
      kind: 'marquee',
      pointer: event.pointerId,
      ...point,
      selection: sel,
      additive,
      bounds: { ...point, width: 0, height: 0 },
    }
    state.overlay.dataset.gesture = 'select'
    state.overlay.setPointerCapture(event.pointerId)
    state.overlay.focus({ preventScroll: true })
    state.syncView()
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
    state.selected.has(layer.id)
      ? layer.id
      : undefined
  if (layer && !resize && !toggleOnClick) selectLayer(state, layer.id, additive)
  if (layer && !state.selected.has(layer.id)) {
    state.syncView()
    state.overlay
      .querySelector<HTMLElement>(
        `[data-editor-layer="${CSS.escape(layer.id)}"] .press-kit-editor-select`,
      )
      ?.focus({ preventScroll: true })
    return
  }
  const initial = selection(state),
    selBounds = bounds(initial)
  if (!selBounds) return
  state.gesture = {
    kind: 'transform',
    pointer: event.pointerId,
    ...point,
    selection: sel,
    initial,
    bounds: selBounds,
    before: copy(state.draft.value),
    resize,
    point,
    moved: false,
    toggleOnClick,
  }
  state.overlay.dataset.gesture = resize ? `resize-${resize}` : 'move'
  state.overlay.setPointerCapture(event.pointerId)
  state.syncView()
  const owner = group
    ? '[data-editor-selection]'
    : `[data-editor-layer="${CSS.escape(layer?.id ?? '')}"]`
  state.overlay
    .querySelector<HTMLElement>(
      group && !resize
        ? '.press-kit-editor-group-move'
        : `${owner} ${resize ? `[data-editor-resize="${resize}"]` : '.press-kit-editor-select'}`,
    )
    ?.focus({ preventScroll: true })
}

export const pointermove = (state: EditorState, event: PointerEvent) => {
  const gesture = state.gesture,
    scene = state.scene,
    point = pointerPosition(state, event)
  state.pointer = point
  if (!gesture) {
    if (state.showRulers) positionOverlay(state)
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
    const rect = state.canvas.getBoundingClientRect()
    if (Math.hypot((dx * rect.width) / scene.width, (dy * rect.height) / scene.height) >= 3)
      for (const layer of layers(state)) {
        const p = placement(state, layer)
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
      selected.size !== state.selected.size ||
      [...selected].some(id => !state.selected.has(id))
    ) {
      state.selected = selected
      state.syncView()
    } else positionOverlay(state)
    return
  }
  gesture.point = point
  const rect = state.canvas.getBoundingClientRect()
  if (Math.hypot((dx * rect.width) / scene.width, (dy * rect.height) / scene.height) >= 3)
    gesture.moved = true
  transformGesture(state, gesture, event.shiftKey)
}

export const finishGesture = (state: EditorState, event: PointerEvent) => {
  const gesture = state.gesture
  if (!gesture || event.pointerId !== gesture.pointer) return
  state.gesture = undefined
  delete state.overlay.dataset.gesture
  if (gesture.kind === 'transform') {
    if (gesture.toggleOnClick && !gesture.moved) state.selected.delete(gesture.toggleOnClick)
    const changed = gesture.initial.some(({ layer, placement: p }) => {
      const current = placement(state, layer)
      return (
        Math.abs(current.x - p.x) > 1e-8 ||
        Math.abs(current.y - p.y) > 1e-8 ||
        Math.abs(current.scaleX - p.scaleX) > 1e-8 ||
        Math.abs(current.scaleY - p.scaleY) > 1e-8
      )
    })
    if (changed) remember(state, gesture.before)
    else state.draft.value = gesture.before
  }
  if (state.overlay.hasPointerCapture(event.pointerId))
    state.overlay.releasePointerCapture(event.pointerId)
  state.syncView()
}

export const keydown = (state: EditorState, event: KeyboardEvent) => {
  if (
    event.isComposing ||
    (event.target instanceof Element && event.target.closest('input, textarea, .g-select'))
  ) {
    event.stopPropagation()
    return
  }
  if (event.key === 'Escape') {
    if (!state.gesture && !state.selected.size) return
    event.preventDefault()
    event.stopPropagation()
    if (state.gesture) cancelGesture(state)
    else {
      state.selected.clear()
      state.syncView()
    }
    return
  }
  event.stopPropagation()
  if (event.key === 'Shift' && state.gesture?.kind === 'transform') {
    transformGesture(state, state.gesture, true)
    return
  }
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'a') {
    event.preventDefault()
    state.selected = new Set(
      layers(state)
        .filter(layer => !placement(state, layer).hidden)
        .map(layer => layer.id),
    )
    state.syncView()
    return
  }
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'z') {
    event.preventDefault()
    history(state, event.shiftKey)
    return
  }
  const items = selection(state)
  if (!items.length) return
  const step = event.shiftKey ? 10 : 1
  if (
    ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Delete', 'Backspace'].includes(event.key)
  ) {
    event.preventDefault()
    edit(state, layout => {
      for (const { layer, placement: p } of items) {
        const next = { ...p }
        if (event.key === 'ArrowLeft') next.x -= step
        if (event.key === 'ArrowRight') next.x += step
        if (event.key === 'ArrowUp') next.y -= step
        if (event.key === 'ArrowDown') next.y += step
        if (event.key === 'Delete' || event.key === 'Backspace') next.hidden = true
        layout.placements[layer.id] = next
      }
    })
  }
}

export const resizeKey = (state: EditorState, event: KeyboardEvent, axis: ResizeAxis) => {
  if (!['ArrowUp', 'ArrowRight', 'ArrowDown', 'ArrowLeft'].includes(event.key)) return
  event.preventDefault()
  event.stopPropagation()
  const items = selection(state),
    selBounds = bounds(items)
  if (!selBounds) return
  const step = event.shiftKey ? 10 : 1
  const dx = event.key === 'ArrowLeft' ? -step : event.key === 'ArrowRight' ? step : 0
  const dy = event.key === 'ArrowUp' ? -step : event.key === 'ArrowDown' ? step : 0
  editBounds(state, resizedBounds(state, items, selBounds, dx, dy, axis))
}
