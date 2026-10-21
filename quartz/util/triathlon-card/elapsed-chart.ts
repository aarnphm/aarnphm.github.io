import type { TriNodeFactory } from './construction-contracts'
import { preferredActivityWind } from '../activity-wind'
import { triText } from '../triathlon-i18n'

export const elapsedSeriesPath = <P extends { elapsedS: number }>(
  points: readonly P[],
  pick: (point: P) => number | null,
  px: (elapsedS: number) => number,
  py: (value: number) => number,
  bottom: number,
  area = false,
): string => {
  let result = ''
  let firstX: number | null = null
  let lastX = 0
  const finish = (): void => {
    if (area && firstX != null) result += `L ${lastX.toFixed(2)} ${bottom} Z `
    firstX = null
  }
  for (const point of points) {
    const value = pick(point)
    if (value == null || !Number.isFinite(value)) {
      finish()
      continue
    }
    const x = px(point.elapsedS)
    const y = py(value)
    if (firstX == null) {
      firstX = x
      result += area ? `M ${x.toFixed(2)} ${bottom} L ` : 'M '
    } else result += 'L '
    result += `${x.toFixed(2)} ${y.toFixed(2)} `
    lastX = x
  }
  finish()
  return result.trim()
}

/** Headwind strip along the bottom row of a 30-unit elapsed chart; returns the segment count. */
export const addWindStrip = <N>(
  f: TriNodeFactory<N>,
  graph: N,
  windSource: ReturnType<typeof preferredActivityWind>,
  px: (elapsedS: number) => number,
  totalElapsedS: number,
): number => {
  const wind = windSource.samples
  let windSegments = 0
  const windSegment = (start: number, end: number, value: number): void => {
    if (!(end > start)) return
    const direction = value > 0 ? 'headwind' : value < 0 ? 'tailwind' : 'calm'
    f.add(
      graph,
      f.svg('rect', {
        class: `tri-cycling-power-wind tri-cycling-power-wind--${direction}`,
        x: px(start).toFixed(3),
        y: 29,
        width: (px(end) - px(start)).toFixed(3),
        height: 1,
        'data-headwind-kph': value.toFixed(1),
        'data-analysis-provider': windSource.provider,
        'data-analysis-source':
          windSource.provider === 'mywindsock' ? 'provider-native' : 'garden-estimate',
      }),
    )
    windSegments++
  }
  for (let index = 1; index < wind.length; index++) {
    const previous = wind[index - 1]
    const current = wind[index]
    const left = previous.headwindKph
    const right = current.headwindKph
    if (left == null || right == null || !Number.isFinite(left) || !Number.isFinite(right)) continue
    const start = Math.max(0, Math.min(totalElapsedS, previous.elapsedS))
    const end = Math.max(0, Math.min(totalElapsedS, current.elapsedS))
    if (left * right < 0) {
      const crossing = start + (end - start) * (Math.abs(left) / (Math.abs(left) + Math.abs(right)))
      windSegment(start, crossing, left / 2)
      windSegment(crossing, end, right / 2)
    } else windSegment(start, end, (left + right) / 2)
  }
  return windSegments
}

export const addWindLegend = <N>(
  f: TriNodeFactory<N>,
  legend: N,
  windSource: ReturnType<typeof preferredActivityWind>,
  windSegments: number,
): void => {
  const text = (key: string): string => triText(f.presentation.locale, key)
  if (windSegments > 0)
    f.add(
      legend,
      f.el('span', 'tri-cycling-power-key tri-cycling-power-key--headwind', text('+ headwind')),
      f.el('span', 'tri-cycling-power-key tri-cycling-power-key--tailwind', text('− tailwind')),
      f.el(
        'span',
        'tri-elev-range',
        text(windSource.provider === 'mywindsock' ? 'myWindsock' : 'Garden wind estimate'),
        {
          'data-analysis-provider': windSource.provider,
          'data-analysis-source':
            windSource.provider === 'mywindsock' ? 'provider-native' : 'garden-estimate',
        },
      ),
    )
  else f.add(legend, f.el('span', undefined, text('wind unavailable')))
}
