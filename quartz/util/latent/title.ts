import type { ElementContent } from 'hast'
import { h } from 'hastscript'

/**
 * Document titles carry TeX: arxiv titles keep `$\mathcal{O}(n)$`, and a title's footnote mark
 * comes through as `$^{*}$`. Markup renders the math as `span.math-inline` (KaTeX at build, and on
 * the client when the links or the reader change); svg labels, tooltips and status lines get a
 * plain reading. A footnote mark alone (`^{*}`, `^\dagger`, `^{1,2}`) is noise in both and drops.
 */

const MARK =
  /\s*\$\s*\^\s*\{?\s*(?:\*|\\ast|\\star|\\dagger|\\ddagger|\\S|\\P|\d+|[a-z]|[,\s])+\s*\}?\s*\$/g
const MATH = /\$(?![\s$])((?:\\.|[^\\$\n])+?)(?<!\s)\$(?!\d)/g

const unmark = (title: string) => title.replace(MARK, '').trim()

/** Title as inline runs: text and `span.math.math-inline` holding TeX. */
export function titleRuns(title: string): ElementContent[] {
  const text = unmark(title)
  const out: ElementContent[] = []
  let last = 0
  for (const m of text.matchAll(MATH)) {
    if (m.index > last) out.push({ type: 'text', value: text.slice(last, m.index) })
    out.push(h('span.math.math-inline', m[1]))
    last = m.index + m[0].length
  }
  if (last < text.length) out.push({ type: 'text', value: text.slice(last) })
  return out
}

const SYMBOL: Record<string, string> = {
  alpha: 'α',
  beta: 'β',
  gamma: 'γ',
  delta: 'δ',
  epsilon: 'ε',
  varepsilon: 'ε',
  theta: 'θ',
  lambda: 'λ',
  mu: 'μ',
  pi: 'π',
  sigma: 'σ',
  tau: 'τ',
  phi: 'φ',
  varphi: 'φ',
  omega: 'ω',
  Delta: 'Δ',
  Sigma: 'Σ',
  Omega: 'Ω',
  infty: '∞',
  times: '×',
  cdot: '·',
  leq: '≤',
  le: '≤',
  geq: '≥',
  ge: '≥',
  neq: '≠',
  approx: '≈',
  to: '→',
  rightarrow: '→',
  in: '∈',
  ell: 'ℓ',
  log: 'log',
  sqrt: '√',
}

const BLACKBOARD: Record<string, string> = { R: 'ℝ', N: 'ℕ', Z: 'ℤ', Q: 'ℚ', C: 'ℂ' }
const SUPERSCRIPT = '⁰¹²³⁴⁵⁶⁷⁸⁹'

/** Title as plain text: footnote marks gone, math as the characters it reads as. */
export function titleText(title: string): string {
  return unmark(title)
    .replace(MATH, (_, tex: string) =>
      tex
        .replace(/\\mathbb\s*\{?([RNZQC])\}?/g, (_, set: string) => BLACKBOARD[set])
        .replace(/\^\{?(\d)\}?/g, (_, digit: string) => SUPERSCRIPT[Number(digit)])
        .replace(/\\([A-Za-z]+)\s*/g, (_, name: string) => SYMBOL[name] ?? '')
        .replace(/[{}]/g, '')
        .replace(/\\([^A-Za-z])/g, '$1'),
    )
    .replace(/\s{2,}/g, ' ')
}
