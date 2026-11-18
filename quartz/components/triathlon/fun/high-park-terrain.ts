import type { Palette, Pin, TerrainData, TerrainView } from './terrain-gl'

type TerrainModule = typeof import('./terrain-gl')

// The renderer is its own LazyScripts bundle (quartz.config.ts), fetched with the heightfield
// only when the map scrolls near view; WebGL never touches pages that do not show it.
let modulePromise: Promise<TerrainModule> | undefined
const loadModule = () =>
  (modulePromise ??= (
    import(new URL('high-park-terrain.js', import.meta.url).href) as Promise<TerrainModule>
  ).catch((error: unknown) => {
    modulePromise = undefined
    throw error
  }))
let terrainPromise: Promise<TerrainData> | undefined
const loadTerrain = (href: string) =>
  (terrainPromise ??= fetch(href)
    .then(response => {
      if (!response.ok) throw new Error(`${href}: ${response.status}`)
      return response.json() as Promise<TerrainData>
    })
    .catch((error: unknown) => {
      terrainPromise = undefined
      throw error
    }))

const FT_PER_M = 3.28084

/** `M x y l dx dy ...` (the lap traces' format) to absolute points. */
export const tracePoints = (d: string): [number, number][] => {
  const points: [number, number][] = []
  for (const run of d.split('M').filter(Boolean)) {
    const numbers = run
      .replace('l', ' ')
      .trim()
      .split(/[\s,]+|(?=-)/)
      .map(Number)
    let [x, y] = [numbers[0], numbers[1]]
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue
    points.push([x, y])
    for (let i = 2; i + 1 < numbers.length; i += 2) {
      x += numbers[i]
      y += numbers[i + 1]
      points.push([x, y])
    }
  }
  return points
}

const decode = (terrain: TerrainData): Uint8Array =>
  Uint8Array.from(atob(terrain.h), c => c.charCodeAt(0))

/** Bilinear height in metres at a frame point (x east, y south, metres). */
export const heightAt = (terrain: TerrainData, bytes: Uint8Array, x: number, y: number): number => {
  const gx = Math.min(Math.max((x - terrain.x0) / terrain.step, 0), terrain.cols - 1.001)
  const gy = Math.min(Math.max((y - terrain.y0) / terrain.step, 0), terrain.rows - 1.001)
  const [i, j] = [Math.floor(gx), Math.floor(gy)]
  const [fx, fy] = [gx - i, gy - j]
  const at = (c: number, r: number) => bytes[r * terrain.cols + c]
  const top = at(i, j) * (1 - fx) + at(i + 1, j) * fx
  const bottom = at(i, j + 1) * (1 - fx) + at(i + 1, j + 1) * fx
  return terrain.min + (top * (1 - fy) + bottom * fy) * terrain.scale
}

// Custom properties can hold `color-mix()`, which a canvas fillStyle may not parse; a probe's
// computed colour is always `rgb()` or `color(srgb …)`, and both reduce to rgb here.
const rgb = (css: string): string => {
  const srgb = /color\(srgb ([\d.]+) ([\d.]+) ([\d.]+)/.exec(css)
  if (!srgb) return css
  const [r, g, b] = srgb.slice(1).map(v => Math.round(Number(v) * 255))
  return `rgb(${r},${g},${b})`
}
const PALETTE: [keyof Palette, string][] = [
  ['paper', '--light'],
  ['ink', '--dark'],
  ['muted', '--darkgray'],
  ['faint', '--gray'],
  ['accent', '--hp-best'],
  ['low', '--hp-low'],
  ['mid', '--hp-mid'],
  ['high', '--hp-high'],
]
const palette = (probe: HTMLElement): Palette => {
  const out = {} as Palette
  for (const [key, name] of PALETTE) {
    probe.style.color = `var(${name})`
    out[key] = rgb(getComputedStyle(probe).color)
  }
  return out
}

/**
 * Turns the High Park map into a 3D relief of the park with the fastest lap standing on it.
 * The SSR'd SVG stays as the 2D view and the fallback; every write here is an inline style or
 * Text.data, which the site's `:has()` rules do not watch (see site-has-restyle-cost).
 */
export const setupHighParkTerrain = (
  root: HTMLElement,
  imperial: () => boolean,
  signal: AbortSignal,
): (() => void) | null => {
  const figure = root.querySelector<HTMLElement>('.tri-hp-map[data-terrain]')
  const plate = figure?.querySelector<HTMLElement>('.tri-hp-plate')
  const svg = plate?.querySelector<SVGSVGElement>('svg')
  const route = svg?.querySelector('.tri-hp-trace-best')?.getAttribute('d')
  const flat = figure?.querySelector<HTMLElement>('.tri-hp-2d')
  const tabs = figure?.querySelector<HTMLElement>('.tri-hp-dims')
  if (!figure || !plate || !svg || !route || !flat || !tabs) return null
  const tabButtons = [...tabs.querySelectorAll<HTMLButtonElement>('[data-dim]')]
  const href = figure.dataset.terrain!
  const exaggeration = Number(figure.dataset.exaggeration) || 5
  const ghosts = [...svg.querySelectorAll('.tri-hp-traces path')]
    .map(path => path.getAttribute('d') ?? '')
    .filter(Boolean)

  let view: TerrainView | null = null
  let canvas: HTMLCanvasElement | null = null
  let probe: HTMLElement | null = null
  let labels: { id: string; node: HTMLElement; text: Text }[] = []
  let dim: '2d' | '3d' = '3d'
  let starting = false
  let dead = false

  const placeLabels = () => {
    if (!view) return
    for (const pin of view.pins()) {
      const label = labels.find(entry => entry.id === pin.id)
      if (!label) continue
      label.node.style.visibility = pin.visible ? '' : 'hidden'
      label.node.style.transform = `translate(${pin.left.toFixed(1)}px,${pin.top.toFixed(1)}px)`
    }
  }
  const pinHeights: Record<string, number> = {}
  const labelText = () => {
    const fmt = (m: number) =>
      imperial() ? `${Math.round(m * FT_PER_M)} ft` : `${Math.round(m)} m`
    for (const { id, text } of labels) {
      const next = `${id} ${fmt(pinHeights[id] ?? 0)}`
      if (text.data !== next) text.data = next
    }
  }

  const show = (next: '2d' | '3d') => {
    dim = next
    const three = next === '3d' && view !== null
    svg.style.visibility = three ? 'hidden' : ''
    for (const node of plate.querySelectorAll<HTMLElement>('.tri-hp-start-label'))
      node.style.display = three ? 'none' : ''
    if (canvas) canvas.style.display = three ? '' : 'none'
    for (const { node } of labels) node.style.display = three ? '' : 'none'
    // The scale and north arrow only hold for the flat map; hidden, they keep the legend's height.
    flat.style.visibility = three ? 'hidden' : ''
    for (const button of tabButtons) {
      const pressed = String(button.dataset.dim === next)
      if (button.getAttribute('aria-pressed') !== pressed)
        button.setAttribute('aria-pressed', pressed)
    }
    if (three) {
      view!.resize()
      view!.render()
    }
  }

  const start = async () => {
    if (starting || view) return
    starting = true
    try {
      const [mod, terrain] = await Promise.all([loadModule(), loadTerrain(href)])
      if (dead || signal.aborted) return
      const points = tracePoints(route)
      const bytes = decode(terrain)
      const heights = points.map(([x, y]) => heightAt(terrain, bytes, x, y))
      // The start is also the loop's low point, so two pins say it all.
      const top = heights.indexOf(Math.max(...heights))
      const pins: Pin[] = [
        { id: 'start', x: points[0][0], y: points[0][1] },
        { id: 'top', x: points[top][0], y: points[top][1] },
      ]
      pinHeights.start = heights[0]
      pinHeights.top = heights[top]
      canvas = document.createElement('canvas')
      canvas.className = 'tri-hp-terrain'
      canvas.setAttribute('role', 'img')
      canvas.setAttribute(
        'aria-label',
        `Relief of High Park with the fastest lap on it, elevation exaggerated ${exaggeration} times.`,
      )
      canvas.style.display = 'none'
      probe = document.createElement('span')
      probe.className = 'tri-hp-probe'
      probe.setAttribute('aria-hidden', 'true')
      labels = pins.map(({ id }) => {
        const node = document.createElement('span')
        node.className = 'tri-hp-pin'
        node.style.display = 'none'
        const text = document.createTextNode('')
        node.append(text)
        return { id, node, text }
      })
      // One insertion for everything the relief adds to the page.
      plate.append(canvas, probe, ...labels.map(({ node }) => node))
      labelText()
      view = mod.mountTerrain(canvas, {
        terrain,
        route,
        ghosts,
        pins,
        exaggeration,
        palette: palette(probe),
        onFrame: placeLabels,
      })
      if (!view) {
        // No WebGL2: the flat map stays and the tabs never show.
        canvas.remove()
        probe.remove()
        for (const { node } of labels) node.remove()
        labels = []
        return
      }
      // The module keeps its canvas sized itself (a ResizeObserver on the canvas).
      tabs.style.display = ''
      show(dim)
    } catch (error) {
      if (!signal.aborted)
        console.warn('High Park relief could not load; the flat map stays.', error)
    } finally {
      starting = false
    }
  }

  const near = new IntersectionObserver(
    entries => {
      if (!entries.some(entry => entry.isIntersecting)) return
      near.disconnect()
      void start()
    },
    { rootMargin: '300px' },
  )
  near.observe(figure)

  const onClick = (event: MouseEvent) => {
    const button = (event.target as Element).closest<HTMLButtonElement>('button')
    if (!button || !figure.contains(button)) return
    const next = button.dataset.dim
    if ((next === '2d' || next === '3d') && next !== dim && view) show(next)
  }
  const onTheme = () =>
    requestAnimationFrame(() => {
      if (!view || !probe) return
      view.setPalette(palette(probe))
      view.render()
    })
  const onUnit = () => labelText()
  figure.addEventListener('click', onClick)
  document.addEventListener('themechange', onTheme)
  window.addEventListener('tri:unit', onUnit)

  return () => {
    dead = true
    near.disconnect()
    figure.removeEventListener('click', onClick)
    document.removeEventListener('themechange', onTheme)
    window.removeEventListener('tri:unit', onUnit)
    view?.dispose()
    view = null
    canvas?.remove()
    probe?.remove()
    for (const { node } of labels) node.remove()
  }
}
