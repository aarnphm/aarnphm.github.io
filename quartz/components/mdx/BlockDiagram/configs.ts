import { MINUS, type Block, type Config, type ConfigId, type Sampler, type Variant } from './model'

// ---- configurations, drawn as in the course figures ----

const blk = (id: string, f: string, extra: Partial<Block> = {}): Block => ({
  k: 'block',
  id,
  f: [f],
  ...extra,
})
const sampler = (id: string, s: string, out?: string): Sampler => ({
  k: 'sampler',
  id,
  sig: s,
  out,
})

const feedbackVariant = (sgn: string, label: string): Variant => ({
  label,
  text: 'Forward block $H_1$, feedback block $H_2$, and a summing junction that adds or subtracts the fed-back $y_2$.',
  d: {
    input: 'u',
    output: 'y',
    chain: [
      { k: 'sum', id: 'sum', signs: ['+', sgn], out: 'u_1' },
      blk('h1', 'H_1', { out: 'y_1' }),
    ],
    loop: { inLabel: 'u_2', chain: [blk('h2', 'H_2', { out: 'y_2' })] },
  },
})

export const CONFIGS: Record<ConfigId, Config> = {
  feedback: {
    variants: [feedbackVariant(MINUS, 'negative'), feedbackVariant('+', 'positive')],
    outName: 'y',
  },
  'summing-junction': {
    variants: [
      {
        label: '(a) block to the left',
        text: 'The junction sits before $G(s)$, so $G(s)$ acts on $R(s) \\mp X(s)$. Moving $G(s)$ to the left past the junction must keep that output.',
        d: {
          input: 'R(s)',
          output: 'C(s)',
          chain: [
            { k: 'sum', id: 'sum', signs: ['+', '∓'], side: { input: 'X(s)', chain: [] } },
            blk('g', 'G(s)'),
          ],
        },
      },
      {
        label: '(b) block to the right',
        text: 'The junction sits after $G(s)$, so $X(s)$ reaches the output without $G(s)$. Moving $G(s)$ to the right past the junction must keep that output.',
        d: {
          input: 'R(s)',
          output: 'C(s)',
          chain: [
            blk('g', 'G(s)'),
            { k: 'sum', id: 'sum', signs: ['+', '∓'], side: { input: 'X(s)', chain: [] } },
          ],
        },
      },
    ],
    outName: 'C',
  },
  'sampler-cascade': {
    variants: [
      {
        label: 'sampler between',
        text: 'A second sampler turns $A(s)$ into the sequence $A^*(s)$ that drives $G_2(s)$.',
        d: {
          input: 'E(s)',
          output: 'C(s)',
          chain: [
            sampler('t1', 'E', 'E^*(s)'),
            blk('g1', 'G_1(s)', { sig: 'A', out: 'A(s)' }),
            sampler('t2', 'A', 'A^*(s)'),
            blk('g2', 'G_2(s)'),
          ],
        },
      },
      {
        label: 'no sampler between',
        text: 'Without the second sampler, $G_2(s)$ receives the whole continuous output of $G_1(s)$.',
        d: {
          input: 'E(s)',
          output: 'C(s)',
          chain: [sampler('t1', 'E', 'E^*(s)'), blk('g1', 'G_1(s)'), blk('g2', 'G_2(s)')],
        },
      },
    ],
    outName: 'C',
  },
  'open-loop': {
    variants: [
      {
        label: 'open loop',
        text: 'The digital controller $D(z)$ drives a zero-order hold and the plant $G_p(s)$. Hold and plant together form $G(s)$.',
        d: {
          input: 'E(s)',
          output: 'C(s)',
          chain: [
            sampler('t1', 'E', 'E(z)'),
            blk('d', 'D(z)', { z: true, sig: 'M', out: 'M(z)' }),
            blk('zoh', '\\frac{1-e^{-sT}}{s}', { out: '\\bar{M}(s)' }),
            blk('gp', 'G_p(s)', {
              merge: 'G(s)',
              zNote: '(1-z^{-1})\\mathcal{P}_T\\left\\{\\frac{G_p(s)}{s}\\right\\}',
            }),
          ],
        },
      },
    ],
    outName: 'C',
  },
  'sampled-feedback': {
    variants: [
      {
        label: 'sampled error',
        text: 'One sampler after the junction samples the error $E(s)$. $G(s)$ and $H(s)$ form one continuous path back to that sampler.',
        d: {
          input: 'R(s)',
          output: 'C(s)',
          chain: [
            { k: 'sum', id: 'sum', signs: ['+', MINUS], out: 'E(s)' },
            sampler('t1', 'E', 'E^*(s)'),
            blk('g', 'G(s)'),
          ],
          loop: { chain: [blk('h', 'H(s)')] },
        },
      },
    ],
    outName: 'C',
  },
  'sensing-device': {
    variants: [
      {
        label: 'sampled feedback',
        text: 'Only the feedback branch is sampled, before $H(s)$. The reference drives $G(s)$ continuously.',
        d: {
          input: 'R(s)',
          output: 'C(s)',
          chain: [{ k: 'sum', id: 'sum', signs: ['+', MINUS], out: 'E(s)' }, blk('g', 'G(s)')],
          loop: { chain: [sampler('t1', 'C', 'C^*(s)'), blk('h', 'H(s)')] },
        },
      },
    ],
    outName: 'C',
  },
  'digital-controller': {
    variants: [
      {
        label: 'two samplers',
        text: 'Synchronized samplers sit before $G_1(s)$ and before $G_2(s)$; $H(s)$ feeds the output back continuously.',
        d: {
          input: 'R(s)',
          output: 'C(s)',
          chain: [
            { k: 'sum', id: 'sum', signs: ['+', MINUS] },
            sampler('t1', 'E', 'E^*'),
            blk('g1', 'G_1(s)'),
            sampler('t2', 'U', 'U^*'),
            blk('g2', 'G_2(s)'),
          ],
          loop: { chain: [blk('h', 'H(s)')] },
        },
      },
    ],
    outName: 'C',
  },
  'pi-controller': {
    variants: [
      {
        label: 'PI controller',
        text: 'The error $E(s)$ drives two parallel paths: proportional $K_1$ and integral $K_2/s$. Their outputs add before the plant $G(s)$, and unity feedback closes the loop.',
        d: {
          input: 'R(s)',
          output: 'C(s)',
          chain: [
            { k: 'sum', id: 'sum', signs: ['+', MINUS], out: 'E(s)' },
            {
              k: 'par',
              id: 'gc',
              signs: ['+', '+'],
              branches: [blk('k1', 'K_1'), blk('k2', '\\frac{K_2}{s}')],
              merge: 'G_c(s)',
            },
            blk('g', 'G(s)'),
          ],
          loop: { chain: [] },
        },
      },
    ],
    outName: 'C',
  },
  proportional: {
    variants: [
      {
        label: 'proportional',
        text: 'The controller multiplies the error $E(s)$ by $K_p$ to give the control signal $U(s)$, which drives the plant $G_p(s)$. A unit-gain sensor feeds the output $Y(s)$ back.',
        d: {
          input: 'R(s)',
          output: 'Y(s)',
          chain: [
            { k: 'sum', id: 'sum', signs: ['+', MINUS], out: 'E(s)' },
            blk('kp', 'K_p', { out: 'U(s)' }),
            blk('gp', 'G_p(s)'),
          ],
          loop: { chain: [] },
        },
      },
    ],
    outName: 'Y',
  },
}
