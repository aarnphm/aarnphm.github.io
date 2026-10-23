import { hbView } from './heavy-ball-model'
import { registerMdxContent } from './mdx-content'

const OUTPUTS = ['alpha', 'beta', 'rho', 'roots', 'settle'] as const

const setupHeavyBallRate = (root: HTMLElement): (() => void) | null => {
  const alpha = root.querySelector<HTMLInputElement>('[data-hbr-alpha]')
  const beta = root.querySelector<HTMLInputElement>('[data-hbr-beta]')
  const curve = root.querySelector<SVGPathElement>('[data-hbr-curve]')
  const guide = root.querySelector<SVGLineElement>('[data-hbr-guide]')
  const mark = root.querySelector<SVGCircleElement>('[data-hbr-mark]')
  const band = root.querySelector<SVGRectElement>('[data-hbr-band]')
  const traj = root.querySelector<SVGPathElement>('[data-hbr-traj]')
  const points = Array.from(root.querySelectorAll<SVGCircleElement>('[data-hbr-point]'))
  if (!alpha || !beta || !curve || !guide || !mark || !band || !traj) return null

  let lastBeta = ''
  const apply = () => {
    const v = hbView(Number(alpha.value), Number(beta.value))
    if (beta.value !== lastBeta) {
      curve.setAttribute('d', v.curve)
      band.setAttribute('width', String(v.bandW))
      lastBeta = beta.value
    }
    guide.setAttribute('x1', String(v.markX))
    guide.setAttribute('x2', String(v.markX))
    guide.setAttribute('y2', String(v.markY))
    mark.setAttribute('cx', String(v.markX))
    mark.setAttribute('cy', String(v.markY))
    traj.setAttribute('d', v.traj)
    points.forEach((p, i) => {
      p.setAttribute('cx', String(v.points[i].x))
      p.setAttribute('cy', String(v.points[i].y))
    })
    for (const key of OUTPUTS) {
      const out = root.querySelector(`[data-hbr-out="${key}"]`)
      if (out) out.textContent = v[key]
    }
    alpha.setAttribute('aria-valuetext', v.valuetextAlpha)
    beta.setAttribute('aria-valuetext', v.valuetextBeta)
  }

  root.addEventListener('input', apply)
  apply()
  return () => root.removeEventListener('input', apply)
}

registerMdxContent('[data-heavy-ball-rate]', root => {
  const cleanup = setupHeavyBallRate(root)
  if (cleanup) window.addCleanup?.(cleanup)
})
