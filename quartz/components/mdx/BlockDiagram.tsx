import katex from 'katex'
import { type FunctionalComponent } from 'preact'
import { customMacros, katexOptions } from '../../cfg'
import { MathText } from '../../util/math-text'
//@ts-ignore
import script from '../scripts/block-diagram.inline'
import style from '../styles/blockDiagram.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type ConfigId =
  | 'feedback'
  | 'summing-junction'
  | 'sampler-cascade'
  | 'open-loop'
  | 'sampled-feedback'
  | 'sensing-device'
  | 'digital-controller'
  | 'pi-controller'
  | 'proportional'
type Props = { caption?: string; config?: ConfigId; variant?: number }

// A block's factors are s-domain TeX terms such as G_1(s); `z` marks a block that is already a
// pulse transfer function, such as a digital controller D(z).
type Block = {
  k: 'block'
  id: string
  f: string[]
  z?: boolean
  // The name a series merge gives the product, such as G(s) for hold and plant.
  merge?: string
  // A closed form the note gives for this block's pulse transfer function.
  zNote?: string
  // Name of the signal this block outputs, when a sampler or the reduction refers to it.
  sig?: string
  out?: string
}
type Sampler = { k: 'sampler'; id: string; sig: string; out?: string }
type Sum = {
  k: 'sum'
  id: string
  signs: [string, string]
  side?: { input: string; chain: Block[] }
  out?: string
}
// Two blocks that take the same input and whose outputs meet at a junction; branches[0] stays on
// the main row, branches[1] runs above it.
type Par = {
  k: 'par'
  id: string
  signs: [string, string]
  branches: [Block, Block]
  merge: string
  out?: string
}
type El = Block | Sampler | Sum | Par
type Loop = { chain: El[]; inLabel?: string }
type Diagram = { input: string; output: string; chain: El[]; loop?: Loop }

type Row = [label: string, tex: string]
type Step = {
  d: Diagram
  name: string
  text: string
  eq: string
  lit: string[]
  done: string[]
  rows: Row[]
}
type Variant = { label: string; d: Diagram; text: string }
type Config = { variants: Variant[]; outName?: string }

const MINUS = '−'
const signTex = (s: string) =>
  s === MINUS ? '-' : s === '∓' ? '\\mp' : s === '±' ? '\\pm' : s === '+' ? '+' : s
const flip = (s: string) =>
  s === MINUS ? '+' : s === '+' ? '-' : s === '∓' ? '\\pm' : s === '±' ? '\\mp' : s

const zOf = (t: string) => t.replace(/\(s\)$/, '(z)')
// P_T{F(s)}: a lone named factor is written F(z), as the note does; a product stays inside P_T.
const PT = (fs: string[]) =>
  fs.length === 1 && fs[0].endsWith('(s)') ? zOf(fs[0]) : `\\mathcal{P}_T\\{${fs.join('')}\\}`
const blockTex = (b: Block) => b.f.join('')
const isBlock = (e: { k: string } | undefined): e is Block => e?.k === 'block'
const sig = (name: string, domain: 's' | 'z' | '*') =>
  domain === '*' ? `${name}^*(s)` : `${name}(${domain})`

// ---- chain rules: series of s-blocks, sampler segments, series of z-blocks ----

const reduceChain = (v: Variant, outName: string): Step[] => {
  const first: Step = { d: v.d, name: 'diagram', text: v.text, eq: '', lit: [], done: [], rows: [] }
  const steps: Step[] = [first]
  let chain = v.d.chain
  const done: string[] = []

  // Series: two adjacent continuous blocks multiply in signal order.
  for (let i = 0; i + 1 < chain.length; ) {
    const a = chain[i]
    const b = chain[i + 1]
    if (!isBlock(a) || !isBlock(b) || a.z || b.z) {
      i++
      continue
    }
    const product = [...a.f, ...b.f]
    const name = b.merge ?? a.merge
    const merged: Block = {
      k: 'block',
      id: `${a.id}${b.id}`,
      f: name ? [name] : product,
      out: b.out,
      sig: b.sig,
      zNote: b.zNote ?? a.zNote,
    }
    chain = [...chain.slice(0, i), merged, ...chain.slice(i + 2)]
    const prev = chain[i - 1]
    const input = prev?.k === 'sampler' ? sig(prev.sig, '*') : 'U(s)'
    steps.push({
      d: { ...v.d, chain },
      name: 'series',
      text: 'Series rule: no sampler separates the two blocks, so the continuous signal passes straight through and their transfer functions multiply.',
      eq: name
        ? `${name} = ${product.join('')}`
        : `${sig(b.sig ?? outName, 's')} = ${product.join('')}\\,${input}`,
      lit: [merged.id],
      done: [...done],
      rows: [],
    })
  }

  // Sampler segments: a block driven by a sample sequence relates input and output samples.
  const zBlocks: { tex: string; id: string }[] = []
  let current: string | null = null
  for (const e of chain) {
    if (e.k === 'sampler') {
      current = e.sig
      continue
    }
    if (!isBlock(e) || current === null) continue
    const outSig = e.sig ?? outName
    const z = e.z ? blockTex(e) : PT(e.f)
    const lit = [e.id, ...chain.filter(s => s.k === 'sampler' && s.sig === current).map(s => s.id)]
    steps.push({
      d: { ...v.d, chain },
      name: 'sampler',
      text: e.z
        ? 'A digital block maps the input sequence to an output sequence.'
        : 'Sampler rule: a sample sequence drives the continuous block up to the next sampler, so its pulse transfer function relates the samples, $\\mathcal{P}_T\\{F(s)\\} = \\mathcal{Z}\\{f(kT)\\}$.',
      eq: `${sig(outSig, 'z')} = ${z}${sig(current, 'z')}${e.zNote ? `,\\quad ${z} = ${e.zNote}` : ''}`,
      lit,
      done: [...done],
      rows: [],
    })
    done.push(...lit)
    zBlocks.push({ tex: z, id: e.id })
    current = outSig
  }

  const inSig = (chain.find(e => e.k === 'sampler') as Sampler | undefined)?.sig ?? 'E'
  // Successive z-blocks are written output side first: C(z) = G_2(z)G_1(z)E(z).
  const product = zBlocks
    .map(b => b.tex)
    .reverse()
    .join('')
  if (zBlocks.length > 1)
    steps.push({
      d: { ...v.d, chain },
      name: 'series',
      text: 'Series rule for sequences: substitute each output sequence into the next relation.',
      eq: `${sig(outName, 'z')} = ${product}${sig(inSig, 'z')}`,
      lit: zBlocks.map(b => b.id),
      done: [...done],
      rows: [],
    })

  const final: Diagram = {
    input: sig(inSig, 'z'),
    output: sig(outName, 'z'),
    chain: [{ k: 'block', id: 'eq', f: [product], z: true }],
  }
  steps.push({
    d: final,
    name: 'result',
    text: 'One block remains between the input and output samples.',
    eq: `\\frac{${sig(outName, 'z')}}{${sig(inSig, 'z')}} = ${product}`,
    lit: ['eq'],
    done: [],
    rows: [[`${sig(outName, 'z')}/${sig(inSig, 'z')}`, product]],
  })
  return steps
}

// ---- summing junction moved past a block, checked by expanding the output ----

type Term = { sign: string; f: string[]; input: string }
type Signal = { tex: string; compound: boolean; terms: Term[] }

const INVERSE = /^\\frac\{1\}\{(.+)\}$/
const cancel = (f: string[]): string[] => {
  const out: string[] = []
  for (const x of f) {
    const inv = x.match(INVERSE)?.[1]
    const j = out.findIndex(y => y === (inv ?? `\\frac{1}{${x}}`))
    if (j >= 0) out.splice(j, 1)
    else out.push(x)
  }
  return out
}

const applyBlock = (s: Signal, b: Block): Signal => ({
  tex: `${blockTex(b)}${s.compound ? `\\left(${s.tex}\\right)` : s.tex}`,
  compound: false,
  terms: s.terms.map(t => ({ ...t, f: [...b.f, ...t.f] })),
})

const evaluate = (input: string, chain: El[]): Signal => {
  let s: Signal = { tex: input, compound: false, terms: [{ sign: '+', f: [], input }] }
  for (const e of chain) {
    if (e.k === 'block') s = applyBlock(s, e)
    else if (e.k === 'sum' && e.side) {
      const side = evaluate(e.side.input, e.side.chain)
      s = {
        tex: `${s.tex} ${signTex(e.signs[1])} ${side.tex}`,
        compound: true,
        terms: [...s.terms, ...side.terms.map(t => ({ ...t, sign: e.signs[1] }))],
      }
    }
  }
  return s
}

const expand = (s: Signal) =>
  s.terms
    .map((t, i) => {
      const body = `${cancel(t.f).join('')}${t.input}`
      return i === 0 && t.sign === '+' ? body : ` ${signTex(t.sign)} ${body}`
    })
    .join('')

const moveJunction = (v: Variant, outName: string): Step[] => {
  const chain = v.d.chain
  const out = sig(outName, 's')
  const i = chain.findIndex(e => e.k === 'sum')
  const sum = chain[i] as Sum
  const ahead = isBlock(chain[i + 1])
  const g = (ahead ? chain[i + 1] : chain[i - 1]) as Block
  const side = sum.side ?? { input: 'X(s)', chain: [] }
  const moved: Sum = {
    ...sum,
    side: {
      input: side.input,
      chain: ahead
        ? [...side.chain, { ...g, id: `${g.id}x` }]
        : [...side.chain, { k: 'block', id: `${g.id}inv`, f: g.f.map(x => `\\frac{1}{${x}}`) }],
    },
  }
  const next = ahead
    ? [...chain.slice(0, i), g, moved, ...chain.slice(i + 2)]
    : [...chain.slice(0, i - 1), moved, g, ...chain.slice(i + 1)]
  const before = evaluate(v.d.input, chain)
  const after = evaluate(v.d.input, next)
  const same = expand(before) === expand(after)
  const rows = (s: Signal): Row[] => [[`${out},\\ \\text{expanded}`, expand(s)]]
  return [
    {
      d: v.d,
      name: 'diagram',
      text: v.text,
      eq: `${out} = ${before.tex}`,
      lit: [],
      done: [],
      rows: rows(before),
    },
    {
      d: { ...v.d, chain: next },
      name: 'move',
      text: ahead
        ? 'Move $G(s)$ to the left past the junction. $X(s)$ now joins after $G(s)$, so it passes through a copy of $G(s)$ first.'
        : 'Move $G(s)$ to the right past the junction. $X(s)$ now passes through $G(s)$ too, so it passes through $1/G(s)$ first.',
      eq: `${out} = ${after.tex}`,
      lit: [sum.id, `${g.id}${ahead ? 'x' : 'inv'}`],
      done: [],
      rows: [...rows(after), ['\\text{same output}', same ? '\\text{yes}' : '\\text{no}']],
    },
  ]
}

// ---- parallel rule: the branch transfer functions add ----

// A top-level \frac{N}{D} as [N, D]; any other term has no denominator.
const fracParts = (t: string): [string, string] | null => {
  if (!t.startsWith('\\frac{')) return null
  const n1 = readGroup(t, 5)
  const d1 = readGroup(t, n1 + 1)
  return d1 === t.length - 1 ? [t.slice(6, n1), t.slice(n1 + 2, d1)] : null
}

const reduceParallel = (d: Diagram): { d: Diagram; steps: Step[] } => {
  const i = d.chain.findIndex(e => e.k === 'par')
  if (i < 0) return { d, steps: [] }
  const p = d.chain[i] as Par
  const [a, b] = p.branches.map(blockTex)
  const op = signTex(p.signs[1])
  const forms = [`${a} ${op} ${b}`]
  const rows: Row[] = []
  const nb = fracParts(b)
  // A plain gain plus N/D shares the denominator D; factoring the gain out shows the zero.
  if (!fracParts(a) && nb) {
    const [n, den] = nb
    const factored = `\\frac{${a}\\left(${den} ${op} \\frac{${n}}{${a}}\\right)}{${den}}`
    forms.push(`\\frac{${a}${den} ${op} ${n}}{${den}}`, factored)
    rows.push([p.merge, factored])
    if (den === 's')
      rows.push(
        ['\\text{zero}', `s = ${op === '+' ? '-' : ''}\\frac{${n}}{${a}}`],
        ['\\text{pole}', 's = 0'],
      )
  } else rows.push([p.merge, forms[0]])
  const merged: Block = { k: 'block', id: p.id, f: [forms.at(-1)!] }
  const next = { ...d, chain: [...d.chain.slice(0, i), merged, ...d.chain.slice(i + 1)] }
  return {
    d: next,
    steps: [
      {
        d: next,
        name: 'parallel',
        text: 'Parallel rule: both paths take the same input and their outputs add at the junction, so their transfer functions add.',
        eq: `${p.merge} = ${forms.join(' = ')}`,
        lit: [p.id],
        done: [],
        rows,
      },
    ],
  }
}

// Clears fractions in a closed loop: F/(1 ∓ FH) with F = Nf/Df, H = Nh/Dh is NfDh/(DfDh ∓ NfNh).
const clearFractions = (fwd: string[], fb: string[], sgn: string): string | null => {
  const parts = (fs: string[]) => {
    const ps = fs.map(f => fracParts(f) ?? [f, ''])
    return [ps.map(x => x[0]).join(''), ps.map(x => x[1]).join('')]
  }
  const [nf, df] = parts(fwd)
  const [nh, dh] = parts(fb)
  if (!df && !dh) return null
  return `\\frac{${nf}${dh}}{${df}${dh} ${flip(sgn)} ${nf}${nh}}`
}

// ---- single loops: continuous feedback rule, or samplers that split the loop ----

const reduceLoop = (v: Variant, outName: string): Step[] => {
  const par = reduceParallel(v.d)
  const d = par.d
  const loop = d.loop!
  const sum = d.chain[0] as Sum
  const sgn = sum.signs[1]
  const steps: Step[] = [
    { d: v.d, name: 'diagram', text: v.text, eq: '', lit: [], done: [], rows: [] },
    ...par.steps,
  ]
  const all = [...d.chain, ...loop.chain]
  const ids = all.map(e => e.id)
  const samplers = all.filter((e): e is Sampler => e.k === 'sampler')

  if (samplers.length === 0) {
    const fwd = d.chain.filter(isBlock).flatMap(b => b.f)
    const fb = loop.chain.filter(isBlock).flatMap(b => b.f)
    const e = sum.out ?? 'e'
    const y = (d.chain.at(-1) as El).out ?? d.output
    // With unity feedback the junction subtracts the output itself.
    const unity = loop.chain.length === 0
    const yb = unity ? y : ((loop.chain.at(-1) as El).out ?? 'b')
    steps.push({
      d,
      name: 'loop',
      text: unity
        ? 'Write one relation per element. Unity feedback returns the output itself to the junction.'
        : 'Write one relation per element. The takeoff point copies the output into the feedback path.',
      eq: `${e} = ${d.input} ${signTex(sgn)} ${yb},\\quad ${y} = ${fwd.join('')}${e}${unity ? '' : `,\\quad ${yb} = ${fb.join('')}${y}`}`,
      lit: ids,
      done: [],
      rows: [],
    })
    const tf = `\\frac{${fwd.join('')}}{1 ${flip(sgn)} ${fwd.join('')}${fb.join('')}}`
    const cleared = clearFractions(fwd, fb, sgn)
    const result = cleared ?? tf
    steps.push({
      d: {
        input: d.input,
        output: d.output,
        chain: [{ k: 'block', id: 'eq', f: [result], z: true }],
      },
      name: 'feedback',
      text: `Feedback rule: eliminate $${e}$${unity ? '' : ` and $${yb}$`}. ${sgn === MINUS ? 'Negative' : 'Positive'} feedback puts $${flip(sgn)}$ in the denominator.${cleared ? ' Multiply above and below by the denominators to clear the inner fractions.' : ''}`,
      eq: `${d.output} = ${tf}\\,${d.input}${cleared ? ` = ${cleared}\\,${d.input}` : ''}`,
      lit: ['eq'],
      done: [],
      rows: [[`${d.output}/${d.input}`, result]],
    })
    return steps
  }

  // Walk the loop from the junction; samplers cut it into segments of continuous blocks.
  type Item = El | { k: 'takeoff'; id: 'takeoff' }
  const cyc: Item[] = [...d.chain, { k: 'takeoff', id: 'takeoff' }, ...loop.chain]
  const sIdx = cyc.flatMap((e, i) => (e.k === 'sampler' ? [i] : []))
  const K = sIdx.length
  const X = sIdx.map(i => (cyc[i] as Sampler).sig)
  const a: string[] = []
  let loopGain: string[] = []
  let b = ''
  let num: string[] = []
  let outputSampled = false
  const done: string[] = []
  const R = d.input.replace(/\(s\)$/, '')

  for (let j = 0; j < K; j++) {
    const from = sIdx[j]
    const seg: Item[] = []
    for (let i = (from + 1) % cyc.length; i !== sIdx[(j + 1) % K]; i = (i + 1) % cyc.length)
      seg.push(cyc[i])
    const sumAt = seg.findIndex(e => e.k === 'sum')
    const takeAt = seg.findIndex(e => e.k === 'takeoff')
    const blocks = (items: Item[]) => items.filter(isBlock).flatMap(e => e.f)
    const segIds = seg.filter(isBlock).map(e => e.id)
    const xj = X[j]
    const xNext = X[(j + 1) % K]

    if (takeAt >= 0 && (sumAt < 0 || takeAt < sumAt)) {
      const c = blocks(seg.slice(0, takeAt))
      const lit = [
        cyc[from].id,
        ...seg
          .slice(0, takeAt)
          .filter(isBlock)
          .map(e => e.id),
      ]
      steps.push({
        d,
        name: 'output',
        text: 'Sampler rule: the samples of the output follow from the sampled signal through the continuous blocks before the takeoff point.',
        eq: `${sig(outName, 'z')} = ${PT(c)}${sig(xj, 'z')}`,
        lit,
        done: [...done],
        rows: [],
      })
      done.push(...lit)
      num = [...a, PT(c)]
    }

    if (sumAt < 0) {
      const g = PT(blocks(seg))
      a.push(g)
      const lit = [cyc[from].id, ...segIds]
      steps.push({
        d,
        name: 'sampler',
        text: 'Sampler rule: the blocks between two samplers act on a sample sequence and are sampled again, so they give one pulse transfer function.',
        eq: `${sig(xNext, 'z')} = ${g}${sig(xj, 'z')}`,
        lit,
        done: [...done],
        rows: [],
      })
      done.push(...lit)
      continue
    }

    // The junction segment: the reference enters between this sampler and the next.
    const p1 = blocks(seg.slice(0, sumAt))
    const p2 = blocks(seg.slice(sumAt + 1))
    outputSampled = takeAt > sumAt
    const cont = `${sig(xNext, 's')} = ${p2.join('')}${d.input} ${signTex(sgn)} ${[...p2, ...p1].join('')}${sig(xj, '*')}`
    steps.push({
      d,
      name: 'loop',
      text: 'Follow the continuous path from the sampler around the loop. No sampler separates these blocks, so they combine before any sampling.',
      eq: cont,
      lit: [...segIds, sum.id],
      done: [...done],
      rows: [],
    })
    done.push(...segIds, sum.id)
    b = PT([...p2, `${R}(s)`])
    const back = PT([...p2, ...p1])
    loopGain = [...a, back]
    steps.push({
      d,
      name: 'sample',
      text: `Sample both sides. A product of continuous blocks is sampled as one: $${back}$, not a product of separate pulse transfer functions.`,
      eq: `${sig(xNext, 'z')} = ${b} ${signTex(sgn)} ${back}${sig(xj, 'z')}`,
      lit: [cyc[from].id, cyc[sIdx[(j + 1) % K]].id],
      done: [...done],
      rows: [],
    })
    done.push(cyc[from].id)
  }

  const den = `1 ${flip(sgn)} ${loopGain.join('')}`
  if (outputSampled) {
    const value = `\\frac{${b}}{${den}}`
    steps.push({
      d,
      name: 'solve',
      text: 'Solve for the output samples. The reference stays inside $\\mathcal{P}_T$, so no ratio to $R(z)$ exists without the continuous reference.',
      eq: `${sig(outName, 'z')} = ${value}`,
      lit: [],
      done: ids,
      rows: [
        [sig(outName, 'z'), value],
        [`${sig(outName, 'z')}/${R}(z)`, '\\text{none}'],
      ],
    })
    return steps
  }
  const tf = `\\frac{${num.join('')}}{${den}}`
  steps.push({
    d: {
      input: `${R}(z)`,
      output: sig(outName, 'z'),
      chain: [{ k: 'block', id: 'eq', f: [tf], z: true }],
    },
    name: 'solve',
    text: 'Eliminate the sampled signals. The numerator is the forward path from the reference samples to the output, the denominator one plus the loop gain.',
    eq: `\\frac{${sig(outName, 'z')}}{${R}(z)} = ${tf}`,
    lit: ['eq'],
    done: [],
    rows: [[`${sig(outName, 'z')}/${R}(z)`, tf]],
  })
  return steps
}

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

const CONFIGS: Record<ConfigId, Config> = {
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

const reduceDiagram = (config: ConfigId, variant: Variant): Step[] => {
  const outName = CONFIGS[config].outName ?? 'C'
  if (variant.d.loop) return reduceLoop(variant, outName)
  if (variant.d.chain.some(e => e.k === 'sum')) return moveJunction(variant, outName)
  return reduceChain(variant, outName)
}

// ---- drawing ----

const tex = (t: string, display = false): string =>
  katex.renderToString(t, {
    ...katexOptions,
    displayMode: display,
    output: 'html',
    macros: customMacros,
    strict: false,
    throwOnError: false,
  })

const readGroup = (t: string, open: number): number => {
  let depth = 0
  for (let i = open; i < t.length; i++) {
    if (t[i] === '\\') i++
    else if (t[i] === '{') depth++
    else if (t[i] === '}' && --depth === 0) return i
  }
  return t.length - 1
}

// Rough rendered width of a TeX label in figure units: fractions take their wider line.
const texLen = (t: string): number => {
  const at = t.indexOf('\\frac{')
  if (at >= 0) {
    const n0 = at + 5
    const n1 = readGroup(t, n0)
    const d1 = readGroup(t, n1 + 1)
    return (
      texLen(t.slice(0, at)) +
      Math.max(texLen(t.slice(n0 + 1, n1)), texLen(t.slice(n1 + 2, d1))) +
      texLen(t.slice(d1 + 1))
    )
  }
  return t
    .replace(/\\mathcal\{P\}_T/g, 'P')
    .replace(/\\(left|right|bigl|bigr|quad|,)/g, '')
    .replace(/_\{?[^{}]\}?|\^\{[^}]*\}|\^./g, '·')
    .replace(/\\[a-zA-Z]+/g, 'x')
    .replace(/[{}\\]/g, '').length
}
const isTall = (t: string) => t.includes('\\frac')

const Y0 = 46
const GAP = 52
const SR = 10
const SW = 30
const BH = 34
const BH_TALL = 50
const FB_GAP = 84
const UNITY_GAP = 46
// A parallel branch runs this far above the main row.
const PAR_RISE = 58
const CH = 7.4

const blockW = (b: Block) => Math.max(56, Math.round(texLen(blockTex(b)) * CH + 22))
const blockH = (b: Block) => (isTall(blockTex(b)) ? BH_TALL : BH)
const elW = (e: El) => (e.k === 'block' ? blockW(e) : e.k === 'sampler' ? SW : 2 * SR)

type Shape =
  | { t: 'wire'; d: string; arrow: boolean }
  | { t: 'label'; x: number; y: number; w: number; tex: string; anchor: 'mid' | 'start' | 'end' }
  | { t: 'block'; id: string; x: number; y: number; w: number; h: number; tex: string }
  | { t: 'sum'; id: string; cx: number; cy: number; signs: [string, number, number][] }
  | { t: 'sampler'; id: string; x: number; y: number; dir: 1 | -1 }
  | { t: 'dot'; cx: number; cy: number }

const LABEL_H = 22

const layout = (d: Diagram): { shapes: Shape[]; w: number; h: number } => {
  const shapes: Shape[] = []
  const Y = Y0 + (d.chain.some(e => e.k === 'par') ? PAR_RISE - 16 : 0)
  const label = (x0: number, x1: number, y: number, t?: string) => {
    if (!t) return
    const w = Math.max(x1 - x0, texLen(t) * CH + 12)
    shapes.push({
      t: 'label',
      x: (x0 + x1) / 2 - w / 2,
      y: y - LABEL_H - 3,
      w,
      tex: t,
      anchor: 'mid',
    })
  }
  let x = 6
  let lastLabel: string | undefined = d.input
  let sumX = 0
  let bottom = Y + BH_TALL / 2

  for (const e of d.chain) {
    const x1 = x + GAP
    if (e.k === 'sampler') {
      shapes.push({ t: 'wire', d: `M${x} ${Y}H${x1}`, arrow: false })
      label(x, x1, Y, lastLabel)
      shapes.push({ t: 'sampler', id: e.id, x: x1, y: Y, dir: 1 })
      x = x1 + SW
    } else if (e.k === 'sum') {
      const cx = x1 + SR
      shapes.push({ t: 'wire', d: `M${x} ${Y}H${x1}`, arrow: true })
      label(x, x1, Y, lastLabel)
      shapes.push({
        t: 'sum',
        id: e.id,
        cx,
        cy: Y,
        signs: [
          [e.signs[0], cx - SR - 7, Y + SR + 9],
          [e.signs[1], cx + SR + 7, Y + SR + 11],
        ],
      })
      sumX = cx
      if (e.side) {
        let y = Y + SR
        let yy = Y + SR + 36
        const parts: string[] = []
        for (const b of e.side.chain) {
          const h = blockH(b)
          const w = blockW(b)
          shapes.push({ t: 'wire', d: `M${cx} ${yy}V${y}`, arrow: true })
          shapes.push({ t: 'block', id: b.id, x: cx - w / 2, y: yy, w, h, tex: blockTex(b) })
          parts.push(b.id)
          y = yy + h
          yy = y + 36
        }
        shapes.push({ t: 'wire', d: `M${cx} ${yy}V${y}`, arrow: true })
        const lw = texLen(e.side.input) * CH + 12
        shapes.push({
          t: 'label',
          x: cx - lw / 2,
          y: yy + 2,
          w: lw,
          tex: e.side.input,
          anchor: 'mid',
        })
        bottom = Math.max(bottom, yy + LABEL_H + 4)
      }
      x = cx + SR
    } else if (e.k === 'par') {
      const [main, up] = e.branches
      const bw = Math.max(blockW(main), blockW(up))
      const hm = blockH(main)
      const hu = blockH(up)
      const yU = Y - PAR_RISE
      const bx = x1 + Math.round(GAP * 0.6)
      const cx = bx + bw + GAP + SR
      shapes.push({ t: 'wire', d: `M${x} ${Y}H${x1}`, arrow: false })
      label(x, x1, Y, lastLabel)
      shapes.push({ t: 'wire', d: `M${x1} ${Y}H${bx}`, arrow: true })
      shapes.push({ t: 'wire', d: `M${x1} ${Y}V${yU}H${bx}`, arrow: true })
      shapes.push({ t: 'dot', cx: x1, cy: Y })
      shapes.push({
        t: 'block',
        id: main.id,
        x: bx,
        y: Y - hm / 2,
        w: bw,
        h: hm,
        tex: blockTex(main),
      })
      shapes.push({ t: 'block', id: up.id, x: bx, y: yU - hu / 2, w: bw, h: hu, tex: blockTex(up) })
      shapes.push({ t: 'wire', d: `M${bx + bw} ${Y}H${cx - SR}`, arrow: true })
      shapes.push({ t: 'wire', d: `M${bx + bw} ${yU}H${cx}V${Y - SR}`, arrow: true })
      shapes.push({
        t: 'sum',
        id: `${e.id}-sum`,
        cx,
        cy: Y,
        signs: [
          [e.signs[0], cx - SR - 7, Y - 5],
          [e.signs[1], cx + 8, Y - SR - 5],
        ],
      })
      x = cx + SR
    } else {
      const w = blockW(e)
      const h = blockH(e)
      shapes.push({ t: 'wire', d: `M${x} ${Y}H${x1}`, arrow: true })
      label(x, x1, Y, lastLabel)
      shapes.push({ t: 'block', id: e.id, x: x1, y: Y - h / 2, w, h, tex: blockTex(e) })
      x = x1 + w
    }
    lastLabel = e.out
  }

  if (!d.loop) {
    const x1 = x + GAP
    shapes.push({ t: 'wire', d: `M${x} ${Y}H${x1}`, arrow: true })
    label(x, x1, Y, lastLabel ?? d.output)
    return { shapes, w: x1 + 8, h: bottom + 8 }
  }

  const fb = d.loop.chain
  const widths = fb.map(elW)
  const fbW = widths.reduce((s, w) => s + w, 0)
  // The feedback row needs a labelled wire gap on both sides of each element.
  const xt = Math.max(x + Math.round(GAP * 0.8), sumX + fbW + (fb.length + 1) * GAP)
  const xo = xt + 46
  shapes.push({ t: 'wire', d: `M${x} ${Y}H${xt}`, arrow: false })
  label(x, xt, Y, lastLabel)
  shapes.push({ t: 'wire', d: `M${xt} ${Y}H${xo}`, arrow: true })
  label(xt, xo, Y, d.output)
  shapes.push({ t: 'dot', cx: xt, cy: Y })

  const yF = Y + (fb.length ? FB_GAP : UNITY_GAP)
  const free = xt - sumX - fbW
  const gap = free / (fb.length + 1)
  let cur = xt
  let fbLabel = d.loop.inLabel
  let first = true
  for (const [i, e] of fb.entries()) {
    const right = cur - gap
    const start = first ? `M${xt} ${Y}V${yF}H${right}` : `M${cur} ${yF}H${right}`
    first = false
    const fl = (x0: number, x1: number, t?: string) => {
      if (!t) return
      const w = Math.max(x1 - x0, texLen(t) * CH + 12)
      shapes.push({
        t: 'label',
        x: (x0 + x1) / 2 - w / 2,
        y: yF - LABEL_H - 3,
        w,
        tex: t,
        anchor: 'mid',
      })
    }
    if (e.k === 'sampler') {
      shapes.push({ t: 'wire', d: start, arrow: false })
      fl(right, cur, fbLabel)
      shapes.push({ t: 'sampler', id: e.id, x: right, y: yF, dir: -1 })
      cur = right - SW
    } else if (e.k === 'block') {
      const h = blockH(e)
      shapes.push({ t: 'wire', d: start, arrow: true })
      fl(right, cur, fbLabel)
      shapes.push({
        t: 'block',
        id: e.id,
        x: right - widths[i],
        y: yF - h / 2,
        w: widths[i],
        h,
        tex: blockTex(e),
      })
      cur = right - widths[i]
      bottom = Math.max(bottom, yF + h / 2)
    }
    fbLabel = e.out
  }
  const back = fb.length ? `M${cur} ${yF}` : `M${xt} ${Y}V${yF}`
  shapes.push({ t: 'wire', d: `${back}H${sumX}V${Y + SR}`, arrow: true })
  if (fbLabel) {
    const w = Math.max(cur - sumX, texLen(fbLabel) * CH + 12)
    shapes.push({
      t: 'label',
      x: (sumX + cur) / 2 - w / 2,
      y: yF - LABEL_H - 3,
      w,
      tex: fbLabel,
      anchor: 'mid',
    })
  }
  bottom = fb.length ? Math.max(bottom, yF + BH_TALL / 2, yF + LABEL_H + 8) : yF + 8
  return { shapes, w: xo + 8, h: bottom + 8 }
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

export default (() => BlockDiagram) satisfies (opts: undefined) => QuartzMdxComponent<Props>
