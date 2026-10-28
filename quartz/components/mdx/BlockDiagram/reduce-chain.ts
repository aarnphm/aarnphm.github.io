import {
  blockTex,
  isBlock,
  PT,
  sig,
  signTex,
  type Block,
  type Diagram,
  type El,
  type Row,
  type Sampler,
  type Step,
  type Sum,
  type Variant,
} from './model'

// ---- chain rules: series of s-blocks, sampler segments, series of z-blocks ----

export const reduceChain = (v: Variant, outName: string): Step[] => {
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

export const moveJunction = (v: Variant, outName: string): Step[] => {
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
