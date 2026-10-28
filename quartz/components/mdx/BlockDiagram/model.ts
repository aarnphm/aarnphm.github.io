import katex from 'katex'
import { customMacros, katexOptions } from '../../../cfg'

export type ConfigId =
  | 'feedback'
  | 'summing-junction'
  | 'sampler-cascade'
  | 'open-loop'
  | 'sampled-feedback'
  | 'sensing-device'
  | 'digital-controller'
  | 'pi-controller'
  | 'proportional'
export type Props = { caption?: string; config?: ConfigId; variant?: number }

// A block's factors are s-domain TeX terms such as G_1(s); `z` marks a block that is already a
// pulse transfer function, such as a digital controller D(z).
export type Block = {
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
export type Sampler = { k: 'sampler'; id: string; sig: string; out?: string }
export type Sum = {
  k: 'sum'
  id: string
  signs: [string, string]
  side?: { input: string; chain: Block[] }
  out?: string
}
// Two blocks that take the same input and whose outputs meet at a junction; branches[0] stays on
// the main row, branches[1] runs above it.
export type Par = {
  k: 'par'
  id: string
  signs: [string, string]
  branches: [Block, Block]
  merge: string
  out?: string
}
export type El = Block | Sampler | Sum | Par
type Loop = { chain: El[]; inLabel?: string }
export type Diagram = { input: string; output: string; chain: El[]; loop?: Loop }

export type Row = [label: string, tex: string]
export type Step = {
  d: Diagram
  name: string
  text: string
  eq: string
  lit: string[]
  done: string[]
  rows: Row[]
}
export type Variant = { label: string; d: Diagram; text: string }
export type Config = { variants: Variant[]; outName?: string }

export const MINUS = '−'
export const signTex = (s: string) =>
  s === MINUS ? '-' : s === '∓' ? '\\mp' : s === '±' ? '\\pm' : s === '+' ? '+' : s
export const flip = (s: string) =>
  s === MINUS ? '+' : s === '+' ? '-' : s === '∓' ? '\\pm' : s === '±' ? '\\mp' : s

const zOf = (t: string) => t.replace(/\(s\)$/, '(z)')
// P_T{F(s)}: a lone named factor is written F(z), as the note does; a product stays inside P_T.
export const PT = (fs: string[]) =>
  fs.length === 1 && fs[0].endsWith('(s)') ? zOf(fs[0]) : `\\mathcal{P}_T\\{${fs.join('')}\\}`
export const blockTex = (b: Block) => b.f.join('')
export const isBlock = (e: { k: string } | undefined): e is Block => e?.k === 'block'
export const sig = (name: string, domain: 's' | 'z' | '*') =>
  domain === '*' ? `${name}^*(s)` : `${name}(${domain})`

// ---- drawing ----

export const tex = (t: string, display = false): string =>
  katex.renderToString(t, {
    ...katexOptions,
    displayMode: display,
    output: 'html',
    macros: customMacros,
    strict: false,
    throwOnError: false,
  })

export const readGroup = (t: string, open: number): number => {
  let depth = 0
  for (let i = open; i < t.length; i++) {
    if (t[i] === '\\') i++
    else if (t[i] === '{') depth++
    else if (t[i] === '}' && --depth === 0) return i
  }
  return t.length - 1
}

// Rough rendered width of a TeX label in figure units: fractions take their wider line.
export const texLen = (t: string): number => {
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
export const isTall = (t: string) => t.includes('\\frac')
