import type { JSX } from 'preact'
import {
  BRANCHES_VIEWS,
  DIRECTIONS,
  FEATURE_NEURONS,
  LIT_NEURON,
  concatenated,
  loading,
  type BranchesView,
} from '../../util/crosscoder-branches'
import { MathText } from '../../util/math-text'
import style from '../styles/crosscoderBranches.scss'
import { H, Label, OneBlock, Panel, Parallel, Sequential, W } from './crosscoder-branches-parts'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string; view?: BranchesView }

let instance = 0

// ---------- superposition: three feature directions in a two-neuron space ----------

const O = W / 2
const R = 70
const tip = (angle: number, r = R) => {
  const rad = (angle * Math.PI) / 180
  return { x: O + r * Math.cos(rad), y: O - r * Math.sin(rad) }
}

const Superposition = ({ marker }: { marker: string }) => (
  <svg class="cb-svg" viewBox={`0 0 ${W} ${W}`} role="img" aria-label="Feature directions">
    <defs>
      <marker
        id={marker}
        viewBox="0 0 10 10"
        refX="10"
        refY="5"
        markerWidth="7"
        markerHeight="7"
        markerUnits="userSpaceOnUse"
        orient="auto"
      >
        <path class="cb-arrowhead" d="M0,0 L10,5 L0,10 z" />
      </marker>
    </defs>
    <circle class="cb-unit" cx={O} cy={O} r={R} />
    <line class="cb-axis" x1={16} y1={O} x2={178} y2={O} />
    <line class="cb-axis" x1={O} y1={184} x2={O} y2={22} />
    <Label x={180} y={O - 8} w={20} tex="n_1" />
    <Label x={O + 4} y={6} w={24} tex="n_2" />
    {DIRECTIONS.map(d => {
      const t = tip(d.angle)
      const at = tip(d.angle, R + 15)
      return (
        <g>
          <line class="cb-drop" x1={t.x} y1={t.y} x2={t.x} y2={O} />
          <line class="cb-vector" x1={O} y1={O} x2={t.x} y2={t.y} marker-end={`url(#${marker})`} />
          <circle class="cb-neuron cb-neuron--lit" cx={t.x} cy={O} r={4} />
          <Label x={at.x - 12} y={at.y - 8} w={24} tex={`d_${d.id}`} align="center" />
        </g>
      )
    })}
  </svg>
)

const LoadingTable = () => (
  <table class="cb-table">
    <thead>
      <tr>
        <th scope="col">feature</th>
        {(['n1', 'n2'] as const).map(n => (
          <th scope="col" class={n === LIT_NEURON ? 'cb-lit' : undefined}>
            <MathText text={`$n_${n[1]}$`} mathClass="cb-math" />
          </th>
        ))}
      </tr>
    </thead>
    <tbody>
      {DIRECTIONS.map(d => (
        <tr>
          <th scope="row">
            <MathText text={`$d_${d.id}$`} mathClass="cb-math" />
          </th>
          {(['n1', 'n2'] as const).map(n => (
            <td class={n === LIT_NEURON ? 'cb-lit' : undefined}>{loading(d[n])}</td>
          ))}
        </tr>
      ))}
    </tbody>
  </table>
)

// ---------- joint: one crosscoder latent over both layers' neurons ----------

const CELL = 20
const HALF_GAP = 8
const CELLS_X = (W - (8 * CELL + HALF_GAP)) / 2
const CELLS_Y = 124
const Z = { x: O, y: 34, w: 40, h: 24 }

const Joint = () => {
  const cells = concatenated().map((c, i) => ({
    ...c,
    x: CELLS_X + i * CELL + (c.layer === 'next' ? HALF_GAP : 0),
  }))
  const half = (layer: 'l' | 'next') => {
    const xs = cells.filter(c => c.layer === layer).map(c => c.x)
    return { x: Math.min(...xs), w: 4 * CELL }
  }
  return (
    <>
      {cells
        .filter(c => c.lit)
        .map(c => (
          <line class="cb-edge" x1={Z.x} y1={Z.y + Z.h} x2={c.x + CELL / 2} y2={CELLS_Y} />
        ))}
      <rect class="cb-latent" x={Z.x - Z.w / 2} y={Z.y} width={Z.w} height={Z.h} />
      <Label x={Z.x - Z.w / 2} y={Z.y + 4} w={Z.w} tex="z" align="center" />
      <Label x={half('l').x - 14} y={80} w={36} tex="d^{l}" />
      <Label x={half('next').x + 46} y={80} w={44} tex="d^{l+1}" />
      {cells.map(c => (
        <g>
          <rect class="cb-cell" x={c.x} y={CELLS_Y} width={CELL} height={CELL} />
          <circle
            class={c.lit ? 'cb-neuron cb-neuron--lit' : 'cb-neuron'}
            cx={c.x + CELL / 2}
            cy={CELLS_Y + CELL / 2}
            r={5}
          />
        </g>
      ))}
      {(['l', 'next'] as const).map(layer => (
        <Label
          x={half(layer).x}
          y={CELLS_Y + CELL + 6}
          w={half(layer).w}
          tex={layer === 'l' ? 'a^{l}' : 'a^{l+1}'}
          align="center"
        />
      ))}
      <Label x={0} y={H - 30} w={W} h={20} tex="[\,a^{l};\ a^{l+1}\,]" align="center" />
    </>
  )
}

// ---------- views ----------

type Step = { label: string; draw: () => JSX.Element }

const STEPS: Record<Exclude<BranchesView, 'superposition'>, Step[]> = {
  parallel: [
    { label: 'sequential: block $l+1$ reads $r_{l+1}=r_l+u_l$', draw: () => <Sequential /> },
    {
      label: 'almost parallel: both blocks read $r_l$; $u_l$ reaches block $l+1$ by one soft edge',
      draw: () => <Parallel />,
    },
  ],
  spread: [
    { label: 'one feature, one neuron', draw: () => <OneBlock lit={[0]} neurons={1} /> },
    {
      label: 'superposition spreads the feature over four neurons',
      draw: () => <OneBlock lit={[0, 1, 2, 3]} neurons={4} />,
    },
    {
      label: 'in almost-parallel form the four neurons can sit in two layers',
      draw: () => <Parallel {...FEATURE_NEURONS} />,
    },
    {
      label: 'the same neurons in the sequential form',
      draw: () => <Sequential {...FEATURE_NEURONS} />,
    },
  ],
  joint: [
    {
      label: 'the four neurons in blocks $l$ and $l+1$',
      draw: () => <Parallel {...FEATURE_NEURONS} />,
    },
    { label: 'crosscoder input: both layers concatenated, one latent $z$', draw: () => <Joint /> },
  ],
}

const CrosscoderBranchesImpl: QuartzMdxComponent<Props> = ({ caption, view = 'parallel' }) => {
  const id = `cb-${++instance}`
  const v: BranchesView = BRANCHES_VIEWS.includes(view) ? view : 'parallel'
  return (
    <figure class="crosscoder-branches" data-cb-view={v}>
      {v === 'superposition' ? (
        <div class="cb-pair">
          <Superposition marker={`${id}-arrow`} />
          <LoadingTable />
        </div>
      ) : (
        <ol class="cb-steps">
          {STEPS[v].map(step => (
            <li class="cb-step">
              <Panel label={step.label.replace(/\$/g, '')}>{step.draw()}</Panel>
              <p class="cb-step-label">
                <MathText text={step.label} mathClass="cb-math" />
              </p>
            </li>
          ))}
        </ol>
      )}
      {caption ? (
        <figcaption class="cb-caption">
          <MathText text={caption} mathClass="cb-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const CrosscoderBranchesComponent = CrosscoderBranchesImpl
CrosscoderBranchesComponent.css = style

export const CrosscoderBranches = registerMdxComponent(
  'CrosscoderBranches',
  CrosscoderBranchesComponent,
)

export default (() => CrosscoderBranches) satisfies (opts: undefined) => QuartzMdxComponent<Props>
