import katex from 'katex'

export {}

type MlaKey = 'd' | 'nh' | 'dh' | 'dc' | 'dr'
type MlaState = Record<MlaKey, number>

const mlaKeys: MlaKey[] = ['d', 'nh', 'dh', 'dc', 'dr']
const mlaLabels: Record<MlaKey, string> = {
  d: 'model dimension',
  nh: 'attention heads',
  dh: 'per-head dimension',
  dc: 'KV latent dimension',
  dr: 'RoPE duplicate dimension',
}

const mlaRenderMath = (tex: string): string =>
  katex.renderToString(tex, {
    displayMode: false,
    output: 'html',
    strict: false,
    throwOnError: false,
  })

const mlaFmtTex = (n: number): string => {
  if (n >= 1000) return `${(n / 1000).toFixed(n % 1000 === 0 ? 0 : 1)}\\mathrm{k}`
  return String(n)
}

const mlaDimTex = (key: string, state: MlaState): string => {
  const ndh = state.nh * state.dh
  if (key === 'cq') return 'd_c^Q'
  if (key === 'dc') return `d_c=${state.dc}`
  if (key === 'dr') return `d_h^R=${state.dr}`
  if (key === 'kc' || key === 'vc') {
    return `n_h d_h=${ndh}`
  }
  if (key === 'qc') {
    return `n_h d_h=${ndh}`
  }
  if (key === 'qr') {
    return `n_h d_h^R=${state.nh * state.dr}`
  }
  if (key === 'concat') {
    return `d_h+d_h^R=${state.dh + state.dr}`
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
  for (const key of ['cq', 'dc', 'dr', 'qc', 'qr', 'kc', 'vc', 'concat']) {
    for (const el of root.querySelectorAll<HTMLElement>(`[data-mla-dim="${key}"]`)) {
      el.innerHTML = mlaRenderMath(mlaDimTex(key, state))
    }
  }
}

const mlaUpdateReadout = (root: HTMLElement, state: MlaState) => {
  const mha = 2 * state.nh * state.dh
  const mla = state.dc + state.dr
  const ratio = mla > 0 ? mha / mla : 0

  const mhaEl = root.querySelector<HTMLElement>('[data-mla-mha]')
  if (mhaEl) mhaEl.innerHTML = mlaRenderMath(mlaFmtTex(mha))

  const mlaEl = root.querySelector<HTMLElement>('[data-mla-mla]')
  if (mlaEl) mlaEl.innerHTML = mlaRenderMath(mlaFmtTex(mla))

  const ratioEl = root.querySelector<HTMLElement>('[data-mla-ratio]')
  if (ratioEl) ratioEl.innerHTML = mlaRenderMath(`${ratio.toFixed(ratio >= 10 ? 0 : 1)}\\times`)
}

const mlaUpdateValueLabels = (root: HTMLElement, state: MlaState) => {
  for (const k of mlaKeys) {
    const valEl = root.querySelector<HTMLElement>(`[data-mla-value="${k}"]`)
    if (valEl) valEl.innerHTML = mlaRenderMath(String(state[k]))
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
