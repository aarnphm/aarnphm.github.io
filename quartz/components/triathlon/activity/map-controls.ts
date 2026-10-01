import { LAYERS_ICON } from '../../../util/triathlon-card'
import { el, svg } from '../runtime/dom'
import {
  readTriMap3d,
  readTriMapElevation,
  readTriMapStyle,
  setTriMap3d,
  setTriMapElevation,
  setTriMapStyle,
  TRI_MAP_3D_EVENT,
  TRI_MAP_ELEVATION_EVENT,
  TRI_MAP_STYLE_EVENT,
} from '../runtime/preferences'

export const buildActivityMapControls = (
  text: (key: string) => string,
): { element: HTMLElement; dispose: () => void } => {
  const controls = el('div', 'tri-map-side', undefined, {
    role: 'group',
    'aria-label': text('map controls'),
  })
  const button = (className: string, label: string, paths: readonly string[]): HTMLElement => {
    const item = el('button', className, undefined, {
      type: 'button',
      'aria-label': text(label),
      title: text(label),
    })
    const icon = svg('svg', { viewBox: '0 0 24 24', 'aria-hidden': 'true' })
    for (const d of paths) icon.append(svg('path', { d }))
    item.append(icon)
    return item
  }
  const fold = button('tri-map-side-fold', 'Collapse map controls', ['M6 9l6 6 6-6'])
  fold.setAttribute('aria-expanded', 'true')
  const body = el('div', 'tri-map-side-body')
  const elevation = button('tri-map-elevation', 'Elevation contours and hill shading', [
    'm3 20 7-16 7 16H3Z',
    'm7.5 9.7 2.5 2.8 2.5-2.8M14 13l3-6 5 13h-5',
  ])
  const terrain = button('tri-map-3d', '3D terrain and buildings', [
    'M12 3 20 7.5 12 12 4 7.5 12 3Z',
    'M4 7.5v9L12 21v-9',
    'M20 7.5v9L12 21',
  ])
  const style = button('tri-map-style', 'satellite', LAYERS_ICON)
  body.append(elevation, terrain, style)
  controls.append(fold, body)
  const sync = (): void => {
    elevation.setAttribute('aria-pressed', String(readTriMapElevation()))
    terrain.setAttribute('aria-pressed', String(readTriMap3d()))
    style.setAttribute('aria-pressed', String(readTriMapStyle() === 'satellite'))
  }
  const onClick = (event: MouseEvent): void => {
    if (!(event.target instanceof Node)) return
    if (fold.contains(event.target)) {
      const expanded = fold.getAttribute('aria-expanded') !== 'true'
      const label = text(expanded ? 'Collapse map controls' : 'Expand map controls')
      fold.setAttribute('aria-expanded', String(expanded))
      fold.setAttribute('aria-label', label)
      fold.setAttribute('title', label)
      body.hidden = !expanded
      controls.classList.toggle('tri-map-side--folded', !expanded)
    } else if (elevation.contains(event.target)) setTriMapElevation(!readTriMapElevation())
    else if (terrain.contains(event.target)) setTriMap3d(!readTriMap3d())
    else if (style.contains(event.target))
      setTriMapStyle(readTriMapStyle() === 'satellite' ? 'mono' : 'satellite')
  }
  sync()
  controls.addEventListener('click', onClick)
  for (const event of [TRI_MAP_3D_EVENT, TRI_MAP_ELEVATION_EVENT, TRI_MAP_STYLE_EVENT])
    window.addEventListener(event, sync)
  return {
    element: controls,
    dispose: () => {
      controls.removeEventListener('click', onClick)
      for (const event of [TRI_MAP_3D_EVENT, TRI_MAP_ELEVATION_EVENT, TRI_MAP_STYLE_EVENT])
        window.removeEventListener(event, sync)
    },
  }
}
