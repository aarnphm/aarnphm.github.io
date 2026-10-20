import { MathText } from '../../util/math-text'
import style from '../styles/memoryAlloc.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Layout = 'seq' | 'array' | 'array-lb' | 'record' | 'align' | 'risc'
type Float = 'none' | 'end'
type Props = { caption?: string; layout?: Layout; float?: Float }

type Cell = { label: string; lit?: boolean; soft?: boolean }

const CELLS: Record<Exclude<Layout, 'align' | 'risc'>, Cell[]> = {
  seq: [
    { label: '0', soft: true },
    { label: 'a', lit: true },
    { label: 'b' },
    { label: 'MaxMem', soft: true },
  ],
  array: [
    { label: '0', soft: true },
    { label: 'a[0]', lit: true },
    { label: '⋯', soft: true },
    { label: 'a[x]' },
    { label: '⋯', soft: true },
    { label: 'a[n−1]' },
    { label: 'MaxMem', soft: true },
  ],
  'array-lb': [
    { label: '0', soft: true },
    { label: 'a[lb]', lit: true },
    { label: '⋯', soft: true },
    { label: 'a[x]' },
    { label: '⋯', soft: true },
    { label: 'a[ub]' },
    { label: 'MaxMem', soft: true },
  ],
  record: [
    { label: '0', soft: true },
    { label: 'r.f₀', lit: true },
    { label: '⋯', soft: true },
    { label: 'r.fᵢ' },
    { label: '⋯', soft: true },
    { label: 'r.fᵤ₋₁' },
    { label: 'MaxMem', soft: true },
  ],
}

const ALIGN_ROWS = [
  { title: 'packed', cells: ['a', 'b', 'c', 'd'], marks: ['0', '1', '2', '3', '4'] },
  {
    title: 'aligned',
    cells: ['a', 'b', '·', 'c', '·', '·', 'd'],
    marks: ['0', '1', '2', '3', '4', '5', '6', '8'],
  },
] as const

const RISC_BANDS: { name: string; note: string; lit?: boolean; soft?: boolean }[] = [
  { name: 'Stack', note: 'SP ↓', lit: true },
  { name: 'Dynamic Data', note: '' },
  { name: 'Static Data', note: '0x10000000' },
  { name: 'Text', note: 'PC / 0x00400000' },
  { name: 'Reserved', note: '0', soft: true },
]

const NOTES: Record<Exclude<Layout, 'align' | 'risc'>, string> = {
  seq: 'adr(a), adr(b) grow from 0 toward MaxMem',
  array: 'adr(a[x]) = adr(a) + x · size',
  'array-lb': 'adr(a[x]) = adr(a) + (x − lb) · size',
  record: 'adr(r.fᵢ) = adr(r) + offset(fᵢ)',
}

const MemoryAllocImpl: QuartzMdxComponent<Props> = ({
  caption,
  layout = 'seq',
  float = 'none',
}) => {
  const floatClass = float === 'end' ? ' memory-alloc--float-end' : ''

  return (
    <figure class={`memory-alloc${floatClass}`} data-ma-layout={layout}>
      {layout === 'risc' ? (
        <div
          class="ma-risc"
          role="img"
          aria-label="RISC-V memory layout from reserved through stack."
        >
          {[...RISC_BANDS].reverse().map(band => (
            <div
              class={
                band.lit ? 'ma-band ma-band--lit' : band.soft ? 'ma-band ma-band--soft' : 'ma-band'
              }
              key={band.name}
            >
              <span class="ma-band-name">{band.name}</span>
              {band.note ? <span class="ma-band-note">{band.note}</span> : null}
            </div>
          ))}
        </div>
      ) : null}

      {layout === 'align' ? (
        <div class="ma-align" role="img" aria-label="Packed versus aligned field layout.">
          {ALIGN_ROWS.map(row => (
            <div class="ma-align-row" key={row.title}>
              <p class="ma-align-title">{row.title}</p>
              <div class="ma-strip">
                {row.cells.map((c, i) => (
                  <div
                    class={c === '·' ? 'ma-cell ma-cell--soft' : 'ma-cell'}
                    key={`${row.title}-${i}`}
                  >
                    {c}
                  </div>
                ))}
              </div>
              <div class="ma-marks">
                {row.marks.map(m => (
                  <span key={`${row.title}-m-${m}`}>{m}</span>
                ))}
              </div>
            </div>
          ))}
        </div>
      ) : null}

      {layout !== 'align' && layout !== 'risc' ? (
        <div class="ma-block" role="img" aria-label={`Memory allocation for ${layout}`}>
          <p class="ma-label">memory</p>
          <div class="ma-strip">
            {CELLS[layout].map((cell, i) => (
              <div
                class={
                  cell.lit
                    ? 'ma-cell ma-cell--lit'
                    : cell.soft
                      ? 'ma-cell ma-cell--soft'
                      : 'ma-cell'
                }
                key={`${layout}-${i}`}
              >
                {cell.label}
              </div>
            ))}
          </div>
          <p class="ma-note">{NOTES[layout]}</p>
        </div>
      ) : null}

      {caption ? (
        <figcaption class="ma-caption">
          <MathText text={caption} mathClass="ma-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const MemoryAllocComponent = MemoryAllocImpl as QuartzMdxComponent<Props>
MemoryAllocComponent.css = style

export const MemoryAlloc = registerMdxComponent('MemoryAlloc', MemoryAllocComponent)

export default (() => MemoryAlloc) satisfies (opts: undefined) => QuartzMdxComponent<Props>
