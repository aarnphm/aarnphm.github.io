type ClientState = {
  theta: string
  bGate: string[]
  piGate: string[]
  fGate: number[]
  fTilde: string[]
  l0: string
  l1: string
  active: string
  zero: string
  ramp: string
  tx: number
  ty: number
  pointY: number[]
  valuetext: string
}

const OUTPUTS = ['theta', 'l0', 'l1', 'active'] as const

const setupGatedSAE = (root: HTMLElement): (() => void) | null => {
  const raw = root.getAttribute('data-gsae-states')
  const slider = root.querySelector<HTMLInputElement>('[data-gsae-theta]')
  if (!raw || !slider) return null
  const states = JSON.parse(raw) as ClientState[]
  const step = Number(slider.step)
  const q = <T extends Element>(sel: string) => root.querySelector<T>(sel)
  const text = (key: string, value: string) => {
    for (const el of root.querySelectorAll(`[data-gsae-text="${key}"]`)) el.textContent = value
  }
  const on = (key: string, value: boolean) => {
    for (const el of root.querySelectorAll(`[data-gsae-cell="${key}"]`))
      el.setAttribute('data-gsae-on', value ? '1' : '0')
  }

  // The server computes every threshold step; the script only swaps values and moves marks.
  const apply = () => {
    const s = states[Math.round(Number(slider.value) / step)]
    if (!s) return
    s.bGate.forEach((v, i) => text(`bgate-${i}`, v))
    s.piGate.forEach((v, i) => text(`pigate-${i}`, v))
    s.fGate.forEach((g, i) => {
      text(`fgate-${i}`, String(g))
      on(`fgate-${i}`, g === 1)
    })
    s.fTilde.forEach((v, i) => {
      text(`ftilde-${i}`, v)
      on(`ftilde-${i}`, v !== '0.00')
    })
    q('[data-gsae-zero]')?.setAttribute('d', s.zero)
    q('[data-gsae-ramp]')?.setAttribute('d', s.ramp)
    const line = q('[data-gsae-theta-line]')
    line?.setAttribute('x1', String(s.tx))
    line?.setAttribute('x2', String(s.tx))
    q('[data-gsae-closed]')?.setAttribute('cx', String(s.tx))
    const open = q('[data-gsae-open]')
    open?.setAttribute('cx', String(s.tx))
    open?.setAttribute('cy', String(s.ty))
    s.pointY.forEach((y, i) => {
      const p = q(`[data-gsae-point="${i}"]`)
      p?.setAttribute('cy', String(y))
      p?.setAttribute('data-gsae-on', s.fTilde[i] !== '0.00' ? '1' : '0')
      q(`[data-gsae-point-label="${i}"]`)?.setAttribute('y', String(y - 9))
    })
    for (const key of OUTPUTS) {
      const out = q(`[data-gsae-out="${key}"]`)
      if (out) out.textContent = s[key]
    }
    slider.setAttribute('aria-valuetext', s.valuetext)
  }

  slider.addEventListener('input', apply)
  apply()
  return () => slider.removeEventListener('input', apply)
}

document.addEventListener('nav', () => {
  for (const root of document.querySelectorAll<HTMLElement>('[data-gated-sae]')) {
    const cleanup = setupGatedSAE(root)
    if (cleanup) window.addCleanup?.(cleanup)
  }
})
