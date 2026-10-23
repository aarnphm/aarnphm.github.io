import katex from 'katex'
import { buildSelect } from '../controls/select'
import {
  ACC_DTYPE_BYTES,
  ACC_VALUES,
  accBytesTex,
  accCompute,
  accMetricTex,
  accRatioTex,
  type AccDType,
  type AccMetrics,
  type AccNumKey,
  type AccState,
} from './attention-cost-model'
import { registerMdxContent } from './mdx-content'

type AccKey = AccNumKey | 'dt'

const accLabels: Record<AccNumKey, string> = {
  dm: 'model dim',
  nl: 'layers',
  nh: 'heads',
  sl: 'seq length',
  bs: 'batch',
}

const accDtypes = Object.keys(ACC_DTYPE_BYTES) as AccDType[]

const accSnap = (value: number, options: number[]): number =>
  options.reduce((best, v) => (Math.abs(value - v) < Math.abs(value - best) ? v : best))

const accReadState = (root: HTMLElement): AccState => ({
  dm: Number(root.dataset.accDm ?? '4096'),
  nl: Number(root.dataset.accNl ?? '32'),
  nh: Number(root.dataset.accNh ?? '32'),
  sl: Number(root.dataset.accSl ?? '4096'),
  bs: Number(root.dataset.accBs ?? '1'),
  dt: ((root.dataset.accDt as AccDType) ?? 'bf16') as AccDType,
})

const accWriteState = (root: HTMLElement, s: AccState) => {
  root.dataset.accDm = String(s.dm)
  root.dataset.accNl = String(s.nl)
  root.dataset.accNh = String(s.nh)
  root.dataset.accSl = String(s.sl)
  root.dataset.accBs = String(s.bs)
  root.dataset.accDt = s.dt
}

const accTex = (tex: string): string =>
  tex ? katex.renderToString(tex, { output: 'html', throwOnError: false, strict: false }) : ''

const accSetTex = (root: HTMLElement, sel: string, tex: string) => {
  const el = root.querySelector<HTMLElement>(sel)
  if (el) el.innerHTML = accTex(tex)
}

const accUpdateInputs = (root: HTMLElement, s: AccState) => {
  for (const k of Object.keys(ACC_VALUES) as AccNumKey[]) {
    const idx = ACC_VALUES[k].indexOf(s[k])
    const input = root.querySelector<HTMLInputElement>(`[data-acc-input="${k}"]`)
    if (input && idx >= 0) {
      input.value = String(idx)
      input.setAttribute('aria-valuenow', String(s[k]))
      input.setAttribute('aria-valuetext', `${accLabels[k]} ${s[k]}`)
    }
    const value = root.querySelector<HTMLElement>(`[data-acc-value="${k}"]`)
    if (value) {
      value.innerHTML = accTex(String(s[k]))
      value.setAttribute('aria-label', `${accLabels[k]} ${s[k]}, type a value`)
    }
  }
}

const accCards = ['dh', 'params', 'flops', 'kvtok', 'kvtotal', 'ratio']

const accUpdateCards = (root: HTMLElement, m: AccMetrics) => {
  for (const k of accCards) {
    accSetTex(root, `[data-acc-num="${k}"]`, accMetricTex(k, m))
  }
  const warning = root.querySelector<HTMLElement>('[data-acc-warning]')
  if (warning) warning.hidden = Number.isInteger(m.dh)
}

const accFlashCard = (root: HTMLElement, k: AccKey) => {
  const cardMap: Record<AccKey, string[]> = {
    dm: ['params', 'flops', 'kvtok', 'kvtotal', 'ratio', 'dh'],
    nl: ['params', 'flops', 'kvtotal', 'ratio'],
    nh: ['dh'],
    sl: ['flops', 'kvtotal', 'ratio'],
    bs: ['kvtotal', 'ratio'],
    dt: ['kvtok', 'kvtotal', 'ratio'],
  }
  for (const card of root.querySelectorAll<HTMLElement>('[data-acc-card]')) {
    if (!cardMap[k].includes(card.dataset.accCard ?? '')) continue
    card.classList.add('acc-card--flash')
    window.setTimeout(() => card.classList.remove('acc-card--flash'), 480)
  }
}

const accUpdateBar = (root: HTMLElement, m: AccMetrics) => {
  const total = m.paramBytes + m.ffnBytes + m.kvtotal
  const parts: [string, number][] = [
    ['attn', m.paramBytes],
    ['ffn', m.ffnBytes],
    ['kv', m.kvtotal],
  ]
  for (const [k, bytes] of parts) {
    const el = root.querySelector<HTMLElement>(`[data-acc-bar="${k}"]`)
    if (el) el.style.width = `${((bytes / total) * 100).toFixed(1)}%`
    accSetTex(root, `[data-acc-barval="${k}"]`, accBytesTex(bytes))
  }
}

const accUpdateSummary = (root: HTMLElement, m: AccMetrics) => {
  accSetTex(root, '[data-acc-sumkv]', accBytesTex(m.kvtotal))
  accSetTex(root, '[data-acc-sumparams]', accBytesTex(m.paramBytes))
  accSetTex(root, '[data-acc-sumratio]', accRatioTex(m.ratio))
}

const accRender = (root: HTMLElement, s: AccState) => {
  const m = accCompute(s)
  accWriteState(root, s)
  accUpdateInputs(root, s)
  accUpdateCards(root, m)
  accUpdateBar(root, m)
  accUpdateSummary(root, m)
}

registerMdxContent('[data-acc-root]', root => {
  if (root.dataset.accBound === 'true') return
  root.dataset.accBound = 'true'

  const state = accReadState(root)
  accRender(root, state)

  const offHandlers: Array<() => void> = []

  for (const k of Object.keys(ACC_VALUES) as AccNumKey[]) {
    const input = root.querySelector<HTMLInputElement>(`[data-acc-input="${k}"]`)
    if (!input) continue
    const opts = ACC_VALUES[k]
    const handler = () => {
      const idx = Math.min(opts.length - 1, Math.max(0, Math.round(Number(input.value))))
      let next = opts[idx]
      if (k === 'nh') {
        const dh = state.dm / next
        if (!Number.isInteger(dh)) {
          const divisors = opts.filter(h => Number.isInteger(state.dm / h))
          if (divisors.length > 0) next = accSnap(next, divisors)
        }
      }
      state[k] = next
      accRender(root, state)
      accFlashCard(root, k)
    }
    input.addEventListener('input', handler)
    offHandlers.push(() => input.removeEventListener('input', handler))

    const valueEl = root.querySelector<HTMLElement>(`[data-acc-value="${k}"]`)
    const editEl = root.querySelector<HTMLInputElement>(`[data-acc-edit="${k}"]`)
    if (valueEl && editEl) {
      const startEdit = () => {
        editEl.value = String(state[k])
        valueEl.hidden = true
        editEl.hidden = false
        editEl.focus()
        editEl.select()
      }
      const commitEdit = () => {
        if (editEl.hidden) return
        const raw = Number(editEl.value)
        if (Number.isFinite(raw) && raw > 0) {
          let next = accSnap(raw, opts)
          if (k === 'nh') {
            const divisors = opts.filter(h => Number.isInteger(state.dm / h))
            if (divisors.length > 0) next = accSnap(next, divisors)
          }
          state[k] = next
          accRender(root, state)
          accFlashCard(root, k)
        }
        editEl.hidden = true
        valueEl.hidden = false
        valueEl.focus({ preventScroll: true })
      }
      const cancelEdit = () => {
        editEl.hidden = true
        valueEl.hidden = false
      }
      const onEditKey = (e: KeyboardEvent) => {
        if (e.key === 'Enter') {
          e.preventDefault()
          commitEdit()
        } else if (e.key === 'Escape') {
          e.preventDefault()
          cancelEdit()
        }
      }
      valueEl.addEventListener('click', startEdit)
      editEl.addEventListener('keydown', onEditKey)
      editEl.addEventListener('blur', commitEdit)
      offHandlers.push(() => {
        valueEl.removeEventListener('click', startEdit)
        editEl.removeEventListener('keydown', onEditKey)
        editEl.removeEventListener('blur', commitEdit)
      })
    }
  }

  const dtHost = root.querySelector<HTMLElement>('[data-acc-dtype]')
  if (dtHost) {
    const select = buildSelect<AccDType>({
      id: `${root.id || 'acc'}-dtype-menu`,
      label: 'kv dtype',
      groups: [
        { options: accDtypes.map(v => ({ value: v, label: `${v} (${ACC_DTYPE_BYTES[v]} B)` })) },
      ],
      selected: state.dt,
      onSelect: v => {
        state.dt = v
        accRender(root, state)
        accFlashCard(root, 'dt')
      },
    })
    dtHost.replaceChildren(select.element)
    offHandlers.push(select.mount())
  }

  window.addCleanup?.(() => {
    for (const off of offHandlers) off()
    delete root.dataset.accBound
  })
})
