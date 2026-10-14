import { ticks } from 'd3'
import { render } from 'preact'
import type { LayerBounds } from './press-kit-layers'

export interface RulerPoint {
  x: number
  y: number
}

export function paintRulers(
  host: HTMLElement,
  width: number,
  height: number,
  canvasWidth: number,
  canvasHeight: number,
  selection?: LayerBounds,
  pointer?: RulerPoint,
) {
  const gutter = 24
  const px = (value: number) => gutter + (value * width) / canvasWidth
  const py = (value: number) => gutter + (value * height) / canvasHeight
  const horizontal = ticks(0, canvasWidth, Math.max(1, Math.floor(width / 60)))
  const vertical = ticks(0, canvasHeight, Math.max(1, Math.floor(height / 60)))
  render(
    <svg width={width + gutter} height={height + gutter} aria-hidden="true">
      <rect class="press-kit-ruler-surface" width={width + gutter} height={gutter} />
      <rect class="press-kit-ruler-surface" width={gutter} height={height + gutter} />
      <text x="3" y="14">
        px
      </text>
      {horizontal.map(value => (
        <g key={`x-${value}`}>
          <path d={`M${px(value)} 18v6`} />
          <text x={px(value) + 3} y="12" text-anchor={px(value) > width ? 'end' : 'start'}>
            {value}
          </text>
        </g>
      ))}
      {vertical.map(value => (
        <g key={`y-${value}`}>
          <path d={`M18 ${py(value)}h6`} />
          <text transform={`translate(12 ${py(value) + 3}) rotate(-90)`} text-anchor="end">
            {value}
          </text>
        </g>
      ))}
      <path d={`M${gutter} 0v${height + gutter}M0 ${gutter}h${width + gutter}`} />
      {selection && (
        <g class="press-kit-ruler-selection">
          <rect
            x={px(selection.x)}
            y="16"
            width={(selection.width * width) / canvasWidth}
            height="8"
          />
          <rect
            x="16"
            y={py(selection.y)}
            width="8"
            height={(selection.height * height) / canvasHeight}
          />
          <path
            d={`M${px(selection.x)} ${gutter}V${height + gutter}M${gutter} ${py(selection.y)}H${width + gutter}`}
          />
        </g>
      )}
      {pointer && (
        <g class="press-kit-ruler-pointer">
          <path d={`M${px(pointer.x)} 0v24M0 ${py(pointer.y)}h24`} />
        </g>
      )}
    </svg>,
    host,
  )
}
