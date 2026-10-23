import type { ProbeState } from './parts'
import { evalTerms, type ProbeSpec, type Series } from './model'

export const probeStatesFor = (
  probe: ProbeSpec | null,
  curve: Series | undefined,
  xPx: (v: number) => number,
  yPx: (v: number) => number,
  fmt: (v: number) => string,
): ProbeState[] => {
  if (!probe || !curve) return []
  const s = curve.spec
  const step = probe.step ?? 1
  const from = s.from ?? 0
  const to = s.to ?? 1
  const f = (x: number) => evalTerms(s.terms!, x)
  const fine = Array.from({ length: 2001 }, (_, i) => from + ((to - from) * i) / 2000)
  const argmin = fine.reduce((a, b) => (f(b) < f(a) ? b : a), from)
  const count = Math.floor((to - from) / step + 1e-9) + 1
  return Array.from({ length: count }, (_, i) => {
    const x = from + i * step
    const y = f(x)
    const regime =
      Math.abs(x - argmin) <= step / 2
        ? (probe.flat ?? 'minimum')
        : x < argmin
          ? (probe.falling ?? 'falling')
          : (probe.rising ?? 'rising')
    return { x, y, cx: xPx(x), cy: yPx(y), xs: x.toFixed(1), ys: fmt(y), regime }
  })
}
