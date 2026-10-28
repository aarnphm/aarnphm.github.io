import {
  blockTex,
  flip,
  isBlock,
  MINUS,
  PT,
  readGroup,
  sig,
  signTex,
  type Block,
  type Diagram,
  type El,
  type Par,
  type Row,
  type Sampler,
  type Step,
  type Sum,
  type Variant,
} from './model'

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

export const reduceLoop = (v: Variant, outName: string): Step[] => {
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
