export {}

type MlaKey = 'd' | 'nh' | 'dh' | 'dc' | 'dr'
type MlaState = Record<MlaKey, number>

const mlaKeys: MlaKey[] = ['d', 'nh', 'dh', 'dc', 'dr']
const mlaLabels: Record<MlaKey, string> = {
  d: 'model dim',
  nh: 'heads',
  dh: 'per-head',
  dc: 'latent',
  dr: 'RoPE',
}

const mlaFmt = (n: number): string => {
  if (n >= 1000) return `${(n / 1000).toFixed(n % 1000 === 0 ? 0 : 1)}k`
  return String(n)
}

const mlaDimHtml = (key: string, state: MlaState): string => {
  const ndh = state.nh * state.dh
  if (key === 'dc') return `<span class="mla-dyn-math">d<sub>c</sub>=${state.dc}</span>`
  if (key === 'dr') return `<span class="mla-dyn-math">d<sub>h</sub><sup>R</sup>=${state.dr}</span>`
  if (key === 'kc' || key === 'vc') {
    return `<span class="mla-dyn-math">n<sub>h</sub>d<sub>h</sub>=${ndh}</span>`
  }
  if (key === 'concat') {
    return `<span class="mla-dyn-math">d<sub>h</sub>+d<sub>h</sub><sup>R</sup>=${state.dh + state.dr}</span>`
  }
  return ''
}

const mlaReadState = (root: HTMLElement): MlaState => {
  const s = { d: 0, nh: 0, dh: 0, dc: 0, dr: 0 }
  for (const k of mlaKeys)
    s[k] = Number(root.dataset[`mla${k.charAt(0).toUpperCase()}${k.slice(1)}`] ?? '0')
  return s
}

const mlaWriteState = (root: HTMLElement, state: MlaState) => {
  for (const k of mlaKeys) {
    root.dataset[`mla${k.charAt(0).toUpperCase()}${k.slice(1)}`] = String(state[k])
  }
}

const mlaUpdateDims = (root: HTMLElement, state: MlaState) => {
  for (const key of ['dc', 'dr', 'kc', 'vc', 'concat']) {
    const el = root.querySelector<HTMLElement>(`[data-mla-dim="${key}"]`)
    if (el) el.innerHTML = mlaDimHtml(key, state)
  }
}

const mlaUpdateReadout = (root: HTMLElement, state: MlaState) => {
  const mha = 2 * state.nh * state.dh
  const mla = state.dc + state.dr
  const ratio = mla > 0 ? mha / mla : 0

  const mhaEl = root.querySelector<HTMLElement>('[data-mla-mha]')
  if (mhaEl) mhaEl.textContent = mlaFmt(mha)

  const mlaEl = root.querySelector<HTMLElement>('[data-mla-mla]')
  if (mlaEl) mlaEl.textContent = mlaFmt(mla)

  const ratioEl = root.querySelector<HTMLElement>('[data-mla-ratio]')
  if (ratioEl) ratioEl.textContent = `${ratio.toFixed(ratio >= 10 ? 0 : 1)}x`
}

const mlaUpdateValueLabels = (root: HTMLElement, state: MlaState) => {
  for (const k of mlaKeys) {
    const valEl = root.querySelector<HTMLElement>(`[data-mla-value="${k}"]`)
    if (valEl) valEl.textContent = String(state[k])
    const input = root.querySelector<HTMLInputElement>(`[data-mla-input="${k}"]`)
    if (input) {
      input.setAttribute('aria-valuenow', String(state[k]))
      input.setAttribute('aria-valuetext', `${mlaLabels[k]} ${state[k]}`)
    }
  }
}

const mlaRender = (root: HTMLElement, state: MlaState) => {
  mlaWriteState(root, state)
  mlaUpdateValueLabels(root, state)
  mlaUpdateDims(root, state)
  mlaUpdateReadout(root, state)
}

const mlaSetup = () => {
  const roots = document.querySelectorAll<HTMLElement>('[data-mla-latent-path]')
  for (const root of roots) {
    if (root.dataset.mlaBound === 'true') continue
    root.dataset.mlaBound = 'true'

    const state = mlaReadState(root)
    mlaRender(root, state)

    const handlers: Array<() => void> = []
    for (const k of mlaKeys) {
      const input = root.querySelector<HTMLInputElement>(`[data-mla-input="${k}"]`)
      if (!input) continue
      const handler = () => {
        state[k] = Number(input.value)
        mlaRender(root, state)
      }
      input.addEventListener('input', handler)
      handlers.push(() => input.removeEventListener('input', handler))
    }

    window.addCleanup?.(() => {
      for (const off of handlers) off()
      delete root.dataset.mlaBound
    })
  }
}

document.addEventListener('nav', mlaSetup)
