import katex from 'katex'
import { type FunctionalComponent } from 'preact'
import { customMacros, katexOptions } from '../../cfg'
import { MathText } from '../../util/math-text'
import style from '../styles/attentionFamilyMap.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Axis = 'memory' | 'kernel' | 'sparsity' | 'parallelism'

type Variant = {
  name: string
  slug: string
  badge: string
  badgeTex?: string
  pitch: string
  signatureTex: string
}

type Family = {
  id: string
  axis: Axis
  title: string
  optimises: string
  bottleneck: string
  variants: Variant[]
}

const FAMILIES: Family[] = [
  {
    id: 'cache-memory',
    axis: 'memory',
    title: 'cache + memory layout',
    optimises: 'shrinks KV bytes touched per decoded token',
    bottleneck: 'KV bandwidth at decode',
    variants: [
      {
        name: 'GQA',
        slug: 'GQA',
        badge: '1/r x KV',
        badgeTex: '\\tfrac{1}{r}\\!\\times\\! KV',
        pitch: 'r query heads share one KV head, slicing the cache by r without retraining.',
        signatureTex: '\\Theta(L\\,d_h\\,h/r)',
      },
      {
        name: 'MLA',
        slug: 'MLA',
        badge: '(d_c+d_h^R)',
        badgeTex: '(d_c+d_h^R)',
        pitch: 'jointly compress KV into a latent of width d_c plus a rotary residual.',
        signatureTex: 'K,V \\to c_{KV}\\in\\mathbb{R}^{d_c}',
      },
      {
        name: 'paged',
        slug: 'paged-attention',
        badge: 'block-paged',
        pitch: 'KV stored in fixed blocks with an OS-style page table; near-zero fragmentation.',
        signatureTex: 'KV[\\text{page}][\\text{slot}]',
      },
      {
        name: 'radix',
        slug: 'radix-attention',
        badge: 'prefix LRU',
        pitch:
          'KV blocks indexed by token-id radix tree; shared prefixes are reused across requests.',
        signatureTex: '\\mathrm{trie}(\\text{tokens})\\to KV',
      },
      {
        name: 'razor',
        slug: 'razor-attention',
        badge: 'score evict',
        pitch: 'per-token importance score drives eviction; keep what attention still queries.',
        signatureTex: 'KV\\setminus\\arg\\min_t s_t',
      },
    ],
  },
  {
    id: 'kernel',
    axis: 'kernel',
    title: 'kernel-level',
    optimises: 'fuses softmax + matmul so HBM is touched once per tile',
    bottleneck: 'HBM/SRAM traffic',
    variants: [
      {
        name: 'FlashAttention',
        slug: 'flash-attention',
        badge: 'O(L) IO',
        badgeTex: 'O(L)\\ \\text{IO}',
        pitch: 'tile Q,K,V into SRAM blocks and run an online softmax; one HBM read per tile.',
        signatureTex: 'O(L^2 d) \\text{ FLOPs},\\ O(L\\,d) \\text{ IO}',
      },
      {
        name: 'cascade',
        slug: 'cascade-attention',
        badge: 'top-k filter',
        pitch: 'coarse pass picks top-k blocks, fine pass attends only to survivors.',
        signatureTex: '\\text{score}\\to\\text{top-}k\\to\\text{attend}',
      },
    ],
  },
  {
    id: 'sparse',
    axis: 'sparsity',
    title: 'sparse / structured',
    optimises: 'replaces dense L by L mask with a structured pattern',
    bottleneck: 'quadratic attention cost',
    variants: [
      {
        name: 'sliding window',
        slug: 'sliding-window',
        badge: 'O(L*w)',
        badgeTex: 'O(L\\!\\cdot\\!w)',
        pitch: 'each query sees a window of radius w plus a handful of global tokens.',
        signatureTex: 'M_{ij}=0\\ \\text{if}\\ |i-j|\\le w',
      },
      {
        name: 'MFA',
        slug: 'MFA',
        badge: 'O(L*m*r)',
        badgeTex: 'O(L\\!\\cdot\\! m\\!\\cdot\\! r)',
        pitch: 'factor the attention map into m rank-r bases; quadratic cost collapses.',
        signatureTex: 'A \\approx \\sum_{i=1}^{m} U_i V_i^{\\!\\top}',
      },
    ],
  },
  {
    id: 'multi-device',
    axis: 'parallelism',
    title: 'multi-device',
    optimises: 'shards sequence across devices so context exceeds one GPU',
    bottleneck: 'context length > single-device HBM',
    variants: [
      {
        name: 'ring',
        slug: 'ring-attention',
        badge: 'Theta(p) rounds',
        badgeTex: '\\Theta(p)\\ \\text{rounds}',
        pitch: 'pipeline K,V around a ring; every device sees the full sequence in p hops.',
        signatureTex: 'p \\text{ devices}\\ \\to\\ p-1 \\text{ rotations}',
      },
      {
        name: 'tree',
        slug: 'tree-attention',
        badge: 'Theta(log p)',
        badgeTex: '\\Theta(\\log p)',
        pitch: 'reduce the softmax along a binary tree; same work, log-depth latency.',
        signatureTex: '(m,z,y)\\ \\text{associative merge}',
      },
    ],
  },
]

const AXIS_LABELS: Record<Axis, string> = {
  memory: 'memory',
  kernel: 'kernel',
  sparsity: 'sparsity',
  parallelism: 'parallelism',
}

function renderMath(tex: string): string {
  return katex.renderToString(tex, {
    ...katexOptions,
    displayMode: false,
    output: 'html',
    macros: customMacros,
    strict: false,
    throwOnError: false,
  })
}

const MathInline: FunctionalComponent<{ tex: string; cls?: string }> = ({ tex, cls }) => (
  <span
    class={`afm-math ${cls ?? ''}`.trim()}
    dangerouslySetInnerHTML={{ __html: renderMath(tex) }}
  />
)

const Chip: FunctionalComponent<{ variant: Variant }> = ({ variant }) => {
  const href = `/thoughts/${encodeURIComponent(variant.slug)}`
  return (
    <a class="afm-chip internal" href={href} data-slug={`thoughts/${variant.slug}`}>
      <span class="afm-chip-row">
        <span class="afm-chip-name">{variant.name}</span>
        <span class="afm-chip-badge">
          {variant.badgeTex ? <MathInline tex={variant.badgeTex} /> : variant.badge}
        </span>
      </span>
      <span class="afm-chip-detail" role="tooltip">
        <span class="afm-chip-pitch">{variant.pitch}</span>
        <MathInline tex={variant.signatureTex} cls="afm-chip-signature" />
      </span>
    </a>
  )
}

const Card: FunctionalComponent<{ family: Family }> = ({ family }) => (
  <article
    class={`afm-card afm-card--${family.axis}`}
    data-axis={family.axis}
    aria-labelledby={`afm-card-${family.id}`}
  >
    <header class="afm-card-head">
      <h3 id={`afm-card-${family.id}`} class="afm-card-title">
        {family.title}
      </h3>
      <p class="afm-card-pitch">{family.optimises}</p>
      <p class="afm-card-bottleneck">
        <span class="afm-card-bottleneck-label">bottleneck</span>
        <span class="afm-card-bottleneck-value">{family.bottleneck}</span>
      </p>
    </header>
    <ul class="afm-chip-list">
      {family.variants.map(v => (
        <li class="afm-chip-item" key={v.slug}>
          <Chip variant={v} />
        </li>
      ))}
    </ul>
  </article>
)

type Props = { caption?: string }

const AttentionFamilyMapImpl: QuartzMdxComponent<Props> = ({ caption }) => (
  <figure class="attention-family-map" data-attention-family-map>
    <nav class="afm-grid" aria-label="Attention variant families">
      {FAMILIES.map(f => (
        <Card key={f.id} family={f} />
      ))}
    </nav>
    <dl class="afm-legend" aria-label="Concern axis legend">
      {(Object.keys(AXIS_LABELS) as Axis[]).map(axis => (
        <div class="afm-legend-item" key={axis} data-axis={axis}>
          <span class="afm-legend-swatch" aria-hidden="true" />
          <dt class="afm-legend-term">{AXIS_LABELS[axis]}</dt>
          <dd class="afm-legend-desc">
            {axis === 'memory' && 'KV bytes per token'}
            {axis === 'kernel' && 'HBM/SRAM traffic'}
            {axis === 'sparsity' && 'edges in the mask'}
            {axis === 'parallelism' && 'sequence across devices'}
          </dd>
        </div>
      ))}
    </dl>
    {caption ? (
      <figcaption class="afm-caption">
        <MathText text={caption} mathClass="afm-math" />
      </figcaption>
    ) : null}
  </figure>
)

const AttentionFamilyMapComponent = AttentionFamilyMapImpl as QuartzMdxComponent<Props>
AttentionFamilyMapComponent.css = style

export const AttentionFamilyMap = registerMdxComponent(
  'AttentionFamilyMap',
  AttentionFamilyMapComponent,
)

export default (() => AttentionFamilyMap) satisfies (opts: undefined) => QuartzMdxComponent<Props>
