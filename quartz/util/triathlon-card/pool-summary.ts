import type { StravaActivityDetail } from '../../plugins/stores/strava'
import type { TriNodeFactory } from './construction-contracts'
import { STROKE_LABEL, SWIM_STROKES } from '../../plugins/stores/apple'
import { dist } from './metric-text'

export const buildSwimStrokes = <N>(f: TriNodeFactory<N>, d: StravaActivityDetail): N | null => {
  const strokes = d.strokes
  if (!strokes) return null
  const entries = SWIM_STROKES.map(s => [s, strokes[s] ?? 0] as const).filter(([, m]) => m > 0)
  const total = entries.reduce((sum, [, m]) => sum + m, 0)
  if (entries.length === 0 || total <= 0) return null
  const box = f.el('div', 'tri-pool-strokes')
  const bar = f.el('div', 'tri-stroke-bar')
  const legend = f.el('ul', 'tri-stroke-legend')
  for (const [s, m] of entries) {
    f.add(
      bar,
      f.el('span', `tri-stroke-seg tri-stroke-${s}`, undefined, {
        style: `width:${((m / total) * 100).toFixed(2)}%`,
      }),
    )
    const li = f.el('li', 'tri-stroke-leg')
    f.add(
      li,
      f.el('span', `tri-stroke-dot tri-stroke-${s}`),
      f.el('span', 'tri-stroke-name', STROKE_LABEL[s]),
      f.el('span', 'tri-stroke-distance', dist(f.presentation, m / 1000, 'swim')),
    )
    f.add(legend, li)
  }
  f.add(box, bar, legend)
  return box
}

export const buildPool = <N>(f: TriNodeFactory<N>, d: StravaActivityDetail): N => {
  const lengths = Math.max(1, Math.round((d.distanceKm * 1000) / 25))
  const wrap = f.el('div', 'tri-pool-wrap')
  const cap = f.el('div', 'tri-pool-cap')
  f.add(
    cap,
    f.el('span', 'tri-pool-lengths', `${lengths} × 25m`),
    f.el('span', 'tri-pool-distance', dist(f.presentation, d.distanceKm, 'swim')),
  )
  f.add(wrap, buildPoolOverview(f), cap)
  const strokes = buildSwimStrokes(f, d)
  if (strokes) f.add(wrap, strokes)
  return wrap
}

const buildPoolOverview = <N>(f: TriNodeFactory<N>): N => {
  const fig = f.svg('svg', {
    class: 'tri-route tri-pool',
    viewBox: '0 0 100 56',
    preserveAspectRatio: 'xMidYMid meet',
  })
  f.add(
    fig,
    f.svg('rect', { x: 6, y: 12, width: 88, height: 32, rx: 16, ry: 16, class: 'tri-pool-lane' }),
  )
  f.add(fig, f.svg('line', { x1: 22, y1: 28, x2: 78, y2: 28, class: 'tri-pool-mid' }))
  return fig
}
