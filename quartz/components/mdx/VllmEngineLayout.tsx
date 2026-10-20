import { MathText } from '../../util/math-text'
import style from '../styles/vllmEngineLayout.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string }

const BLOCKS = [1, 2, 3, 4, '…', 'N-1', 'N'] as const

const VllmEngineLayoutImpl: QuartzMdxComponent<Props> = ({ caption }) => (
  <figure class="vllm-engine-layout">
    <section class="vel-client">
      <h3 class="vel-title">engine core client</h3>
      <div class="vel-flow">
        <span class="vel-chip">requests in</span>
        <span class="vel-box">processor</span>
        <div class="vel-core">
          <span class="vel-box vel-box--exec">model executor</span>
          <div class="vel-sched">
            <span class="vel-box vel-box--sched">scheduler</span>
            <div class="vel-queues">
              <span class="vel-mini">waiting</span>
              <span class="vel-mini">running</span>
              <span class="vel-mini vel-mini--lit">KV cache manager</span>
            </div>
          </div>
        </div>
        <span class="vel-box">output processor</span>
        <span class="vel-chip">result out</span>
      </div>
    </section>

    <section class="vel-band vel-band--cpu">
      <header class="vel-band-head">
        <h3 class="vel-title">indexing structure</h3>
        <span class="vel-where">CPU</span>
      </header>
      <div class="vel-index">
        <span class="vel-mini vel-mini--lit">KV cache manager</span>
        <div class="vel-queue" aria-label="free block queue">
          {BLOCKS.map(b => (
            <span class="vel-block" key={String(b)}>
              {b === '…' ? '…' : `blk ${b}`}
            </span>
          ))}
        </div>
      </div>
    </section>

    <section class="vel-band vel-band--gpu">
      <header class="vel-band-head">
        <h3 class="vel-title">actual KV memory</h3>
        <span class="vel-where">GPU</span>
      </header>
      <p class="vel-note">paged KV cache memory</p>
      <div class="vel-pool" aria-label="physical KV blocks">
        {BLOCKS.map(b => (
          <span class="vel-phys" key={String(b)}>
            {b === '…' ? '…' : `blk ${b}`}
          </span>
        ))}
      </div>
    </section>

    {caption ? (
      <figcaption class="vel-caption">
        <MathText text={caption} mathClass="vel-math" />
      </figcaption>
    ) : null}
  </figure>
)

const VllmEngineLayoutComponent = VllmEngineLayoutImpl as QuartzMdxComponent<Props>
VllmEngineLayoutComponent.css = style

export const VllmEngineLayout = registerMdxComponent('VllmEngineLayout', VllmEngineLayoutComponent)

export default (() => VllmEngineLayout) satisfies (opts: undefined) => QuartzMdxComponent<Props>
