import { MathText } from '../../util/math-text'
import style from '../styles/processThreads.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string }

const SHARED = ['code', 'data', 'files'] as const

const ThreadCol = ({ n }: { n: number }) => (
  <div class="pt-thread">
    <div class="pt-state">
      <span>registers</span>
      <span>stack</span>
    </div>
    <div class="pt-wave" aria-hidden="true">
      <svg viewBox="0 0 20 48" class="pt-wave-svg">
        <path class="pt-wave-path" d="M10 2 C4 10, 16 18, 10 26 C4 34, 16 42, 10 46" />
      </svg>
    </div>
    <span class="pt-thread-label">thread {n}</span>
  </div>
)

const ProcessThreadsImpl: QuartzMdxComponent<Props> = ({ caption }) => (
  <figure class="process-threads">
    <div class="pt-grid">
      <section class="pt-card">
        <h3 class="pt-title">single-threaded process</h3>
        <div class="pt-shared">
          {SHARED.map(s => (
            <span class="pt-shared-cell" key={s}>
              {s}
            </span>
          ))}
        </div>
        <ThreadCol n={1} />
      </section>

      <section class="pt-card">
        <h3 class="pt-title">multi-threaded process</h3>
        <div class="pt-shared">
          {SHARED.map(s => (
            <span class="pt-shared-cell" key={s}>
              {s}
            </span>
          ))}
        </div>
        <div class="pt-threads">
          <ThreadCol n={1} />
          <ThreadCol n={2} />
          <ThreadCol n={3} />
        </div>
      </section>
    </div>
    {caption ? (
      <figcaption class="pt-caption">
        <MathText text={caption} mathClass="pt-math" />
      </figcaption>
    ) : null}
  </figure>
)

const ProcessThreadsComponent = ProcessThreadsImpl as QuartzMdxComponent<Props>
ProcessThreadsComponent.css = style

export const ProcessThreads = registerMdxComponent('ProcessThreads', ProcessThreadsComponent)

export default (() => ProcessThreads) satisfies (opts: undefined) => QuartzMdxComponent<Props>
