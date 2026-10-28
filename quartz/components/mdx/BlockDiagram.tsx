import type { FunctionalComponent } from 'preact'
import { MathText } from '../../util/math-text'
//@ts-ignore
import script from '../scripts/block-diagram.inline'
import style from '../styles/blockDiagram.scss'
import { CONFIGS } from './BlockDiagram/configs'
import { LABEL_H, layout, SR, SW } from './BlockDiagram/layout'
import { tex, type ConfigId, type Props, type Step, type Variant } from './BlockDiagram/model'
import { moveJunction, reduceChain } from './BlockDiagram/reduce-chain'
import { reduceLoop } from './BlockDiagram/reduce-loop'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

const reduceDiagram = (config: ConfigId, variant: Variant): Step[] => {
  const outName = CONFIGS[config].outName ?? 'C'
  if (variant.d.loop) return reduceLoop(variant, outName)
  if (variant.d.chain.some(e => e.k === 'sum')) return moveJunction(variant, outName)
  return reduceChain(variant, outName)
}

const stateOf = (id: string, lit: string[], done: string[]) =>
  lit.includes(id) ? 'lit' : done.includes(id) ? 'done' : 'idle'

const DiagramSvg: FunctionalComponent<{ step: Step; arrowId: string; label: string }> = ({
  step,
  arrowId,
  label,
}) => {
  const { shapes, w, h } = layout(step.d)
  return (
    <svg
      class="bd-svg"
      viewBox={`0 0 ${Math.ceil(w)} ${Math.ceil(h)}`}
      style={{ maxWidth: `${Math.ceil(w * 1.25)}px` }}
      preserveAspectRatio="xMidYMid meet"
      role="img"
      aria-label={label}
    >
      <defs>
        <marker
          id={arrowId}
          viewBox="0 0 10 10"
          refX="10"
          refY="5"
          markerWidth="7"
          markerHeight="7"
          markerUnits="userSpaceOnUse"
          orient="auto"
        >
          <path class="bd-arrowhead" d="M0,0 L10,5 L0,10 z" />
        </marker>
      </defs>
      {shapes.map(s => {
        if (s.t === 'wire')
          return (
            <path class="bd-wire" d={s.d} marker-end={s.arrow ? `url(#${arrowId})` : undefined} />
          )
        if (s.t === 'dot') return <circle class="bd-dot" cx={s.cx} cy={s.cy} r={3} />
        if (s.t === 'sum')
          return (
            <g class="bd-sum" data-state={stateOf(s.id, step.lit, step.done)}>
              <circle cx={s.cx} cy={s.cy} r={SR} />
              {s.signs.map(([sg, x, y]) => (
                <text class="bd-sign" x={x} y={y} text-anchor="middle">
                  {sg}
                </text>
              ))}
            </g>
          )
        if (s.t === 'sampler')
          return (
            <g class="bd-sampler" data-state={stateOf(s.id, step.lit, step.done)}>
              <rect
                class="bd-sampler-ground"
                x={s.dir === 1 ? s.x - 3 : s.x - SW + 3}
                y={s.y - 15}
                width={SW}
                height={19}
              />
              <path class="bd-wire" d={`M${s.x} ${s.y}L${s.x + s.dir * 22} ${s.y - 11}`} />
              <text class="bd-t" x={s.x + s.dir * 18} y={s.y + 20} text-anchor="middle">
                T
              </text>
            </g>
          )
        if (s.t === 'block')
          return (
            <g class="bd-block" data-state={stateOf(s.id, step.lit, step.done)}>
              <rect x={s.x} y={s.y} width={s.w} height={s.h} />
              <foreignObject x={s.x} y={s.y} width={s.w} height={s.h}>
                <div
                  class="bd-fo bd-fo--block"
                  dangerouslySetInnerHTML={{ __html: tex(`\\displaystyle ${s.tex}`) }}
                />
              </foreignObject>
            </g>
          )
        return (
          <foreignObject x={s.x} y={s.y} width={s.w} height={LABEL_H}>
            <div class="bd-fo bd-fo--label" dangerouslySetInnerHTML={{ __html: tex(s.tex) }} />
          </foreignObject>
        )
      })}
    </svg>
  )
}

let instance = 0

const BlockDiagramImpl: QuartzMdxComponent<Props> = ({
  caption,
  config = 'feedback',
  variant = 0,
}) => {
  const id = `bd-${++instance}`
  const cfg = CONFIGS[config] ?? CONFIGS.feedback
  const cid = CONFIGS[config] ? config : 'feedback'
  const v0 = variant >= 0 && variant < cfg.variants.length ? variant : 0
  const runs = cfg.variants.map(v => reduceDiagram(cid, v))

  return (
    <figure class="block-diagram" data-block-diagram data-bd-config={cid}>
      {cfg.variants.length > 1 ? (
        <fieldset class="bd-segments">
          <legend class="bd-legend">Configuration</legend>
          {cfg.variants.map((v, i) => (
            <label class="bd-segment">
              <input
                type="radio"
                name={`${id}-variant`}
                value={String(i)}
                checked={i === v0}
                data-bd-variant-input
              />
              <span>{v.label}</span>
            </label>
          ))}
        </fieldset>
      ) : null}

      {runs.map((steps, vi) => (
        <div class="bd-variant" data-bd-variant={String(vi)} hidden={vi !== v0}>
          <fieldset class="bd-segments bd-segments--steps">
            <legend class="bd-legend">Reduction step</legend>
            {steps.map((s, si) => (
              <label class="bd-segment">
                <input
                  type="radio"
                  name={`${id}-step-${vi}`}
                  value={String(si)}
                  checked={si === 0}
                  data-bd-step-input
                />
                <span>{`${si} ${s.name}`}</span>
              </label>
            ))}
          </fieldset>

          {steps.map((s, si) => (
            <div class="bd-step" data-bd-step={String(si)} hidden={si !== 0}>
              <DiagramSvg
                step={s}
                arrowId={`${id}-arrow-${vi}-${si}`}
                label={`Block diagram, ${cfg.variants[vi].label}, step ${si}: ${s.name}.`}
              />
              <p class="bd-text">
                <MathText text={s.text} mathClass="bd-math" />
              </p>
              {s.eq ? (
                <div
                  class="bd-eq"
                  data-bd-eq={s.eq}
                  dangerouslySetInnerHTML={{ __html: tex(s.eq, true) }}
                />
              ) : null}
              {s.rows.length > 0 ? (
                <dl class="bd-readout">
                  {s.rows.map(([label, value]) => (
                    <div>
                      <dt dangerouslySetInnerHTML={{ __html: tex(label) }} />
                      <dd
                        data-bd-out={value}
                        dangerouslySetInnerHTML={{ __html: tex(`\\displaystyle ${value}`) }}
                      />
                    </div>
                  ))}
                </dl>
              ) : null}
            </div>
          ))}
        </div>
      ))}

      {caption ? (
        <figcaption class="bd-caption">
          <MathText text={caption} mathClass="bd-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

BlockDiagramImpl.css = style
BlockDiagramImpl.afterDOMLoaded = script

export const BlockDiagram = registerMdxComponent('BlockDiagram', BlockDiagramImpl)
