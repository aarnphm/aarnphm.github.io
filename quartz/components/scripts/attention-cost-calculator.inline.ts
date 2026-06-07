import katex from 'katex'

type AccDType = 'bf16' | 'fp16' | 'fp32' | 'int8' | 'int4'
type AccNumKey = 'dm' | 'nl' | 'nh' | 'sl' | 'bs'
type AccKey = AccNumKey | 'dt'

type AccState = { dm: number; nl: number; nh: number; sl: number; bs: number; dt: AccDType }

const accDmValues = [128, 256, 512, 1024, 2048, 4096, 8192]
const accNlValues = [1, 2, 4, 6, 8, 12, 16, 24, 32, 40, 48, 64, 80, 96]
const accNhValues = [1, 2, 4, 8, 16, 32, 64]
const accSlValues = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
const accBsValues = [1, 2, 4, 8, 16, 32, 64, 128, 256]

const accValueTable: Record<AccNumKey, number[]> = {
  dm: accDmValues,
  nl: accNlValues,
  nh: accNhValues,
  sl: accSlValues,
  bs: accBsValues,
}

const accDtypeBytes: Record<AccDType, number> = { bf16: 2, fp16: 2, fp32: 4, int8: 1, int4: 0.5 }

const accLabels: Record<AccKey, string> = {
  dm: 'model dim',
  nl: 'layers',
  nh: 'heads',
  sl: 'seq length',
  bs: 'batch',
  dt: 'dtype',
}

const accSnap = (value: number, options: number[]): number => {
  let best = options[0]
  let bestDiff = Math.abs(value - best)
  for (const v of options) {
    const diff = Math.abs(value - v)
    if (diff < bestDiff) {
      best = v
      bestDiff = diff
    }
  }
  return best
}

const accFmtBytes = (n: number): string => {
  if (n < 1024) return `${n.toFixed(0)} B`
  const units = ['KB', 'MB', 'GB', 'TB', 'PB']
  let v = n / 1024
  let i = 0
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i++
  }
  return `${v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2)} ${units[i]}`
}

const accFmtCount = (n: number): string => {
  if (n < 1000) return n.toString()
  const units = ['K', 'M', 'B', 'T', 'P']
  let v = n / 1000
  let i = 0
  while (v >= 1000 && i < units.length - 1) {
    v /= 1000
    i++
  }
  return `${v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2)}${units[i]}`
}

const accSignedDelta = (cur: number, prev: number): string => {
  if (!isFinite(prev) || prev === 0 || prev === cur) return ''
  const diff = cur - prev
  if (diff === 0) return ''
  const abs = Math.abs(diff)
  const sign = diff > 0 ? '+' : '-'
  return `${sign}${accFmtBytes(abs)}`
}

const accSignedCountDelta = (cur: number, prev: number): string => {
  if (!isFinite(prev) || prev === 0 || prev === cur) return ''
  const diff = cur - prev
  if (diff === 0) return ''
  const abs = Math.abs(diff)
  const sign = diff > 0 ? '+' : '-'
  return `${sign}${accFmtCount(abs)}`
}

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

type AccMetrics = {
  dh: number
  dhInteger: boolean
  params: number
  flops: number
  kvtok: number
  kvtotal: number
  ratio: number
  ffnParams: number
  paramBytes: number
}

const accCompute = (s: AccState): AccMetrics => {
  const dh = s.dm / s.nh
  const dhInteger = Number.isInteger(dh)
  const bytes = accDtypeBytes[s.dt]
  const params = s.nl * 4 * s.dm * s.dm
  const ffnParams = s.nl * 8 * s.dm * s.dm
  const flops = 4 * s.sl * s.dm
  const kvtok = 2 * s.dm * bytes
  const kvtotal = s.bs * s.sl * s.nl * 2 * s.dm * bytes
  const paramBytes = params * bytes
  const ratio = paramBytes > 0 ? kvtotal / paramBytes : 0
  return { dh, dhInteger, params, flops, kvtok, kvtotal, ratio, ffnParams, paramBytes }
}

const accSetText = (root: HTMLElement, sel: string, text: string) => {
  const el = root.querySelector<HTMLElement>(sel)
  if (el) el.textContent = text
}

const accTex = (v: string): string => {
  const m = v.match(/^([+\-\d.,\s]*)(.*)$/)
  const num = m ? m[1] : v
  const unit = m ? m[2] : ''
  const tex = unit ? `${num}\\text{${unit}}` : num
  try {
    return katex.renderToString(tex, {
      displayMode: false,
      output: 'html',
      throwOnError: false,
      strict: false,
    })
  } catch {
    return v
  }
}

const accSetTex = (root: HTMLElement, sel: string, text: string) => {
  const el = root.querySelector<HTMLElement>(sel)
  if (el) el.innerHTML = accTex(text)
}

const accUpdateInputs = (root: HTMLElement, s: AccState) => {
  for (const k of Object.keys(accValueTable) as AccNumKey[]) {
    const opts = accValueTable[k]
    const idx = opts.indexOf(s[k])
    const input = root.querySelector<HTMLInputElement>(`[data-acc-input="${k}"]`)
    if (input && idx >= 0) {
      input.value = String(idx)
      input.setAttribute('aria-valuenow', String(s[k]))
      input.setAttribute('aria-valuetext', `${accLabels[k]} ${s[k]}`)
    }
    accSetTex(root, `[data-acc-value="${k}"]`, String(s[k]))
  }
  const dtSelect = root.querySelector<HTMLSelectElement>('[data-acc-input="dt"]')
  if (dtSelect) dtSelect.value = s.dt
}

const accUpdateCards = (root: HTMLElement, m: AccMetrics, prev: AccMetrics | null) => {
  accSetTex(root, '[data-acc-num="dh"]', m.dhInteger ? m.dh.toString() : m.dh.toFixed(2))
  accSetTex(root, '[data-acc-num="params"]', accFmtCount(m.params))
  accSetTex(root, '[data-acc-num="flops"]', accFmtCount(m.flops))
  accSetTex(root, '[data-acc-num="kvtok"]', accFmtBytes(m.kvtok))
  accSetTex(root, '[data-acc-num="kvtotal"]', accFmtBytes(m.kvtotal))
  accSetTex(root, '[data-acc-num="ratio"]', `${m.ratio.toFixed(m.ratio >= 10 ? 0 : 2)}x`)

  if (prev) {
    accSetText(
      root,
      '[data-acc-delta="dh"]',
      m.dh === prev.dh ? '' : `${m.dh > prev.dh ? '+' : '-'}${Math.abs(m.dh - prev.dh).toFixed(0)}`,
    )
    accSetText(root, '[data-acc-delta="params"]', accSignedCountDelta(m.params, prev.params))
    accSetText(root, '[data-acc-delta="flops"]', accSignedCountDelta(m.flops, prev.flops))
    accSetText(root, '[data-acc-delta="kvtok"]', accSignedDelta(m.kvtok, prev.kvtok))
    accSetText(root, '[data-acc-delta="kvtotal"]', accSignedDelta(m.kvtotal, prev.kvtotal))
    accSetText(
      root,
      '[data-acc-delta="ratio"]',
      m.ratio === prev.ratio
        ? ''
        : `${m.ratio > prev.ratio ? '+' : '-'}${Math.abs(m.ratio - prev.ratio).toFixed(2)}x`,
    )
  }

  const warning = root.querySelector<HTMLElement>('[data-acc-warning]')
  if (warning) warning.hidden = m.dhInteger
}

const accFlashCard = (root: HTMLElement, k: AccKey) => {
  const cardMap: Partial<Record<AccKey, string[]>> = {
    dm: ['params', 'flops', 'kvtok', 'kvtotal', 'ratio', 'dh'],
    nl: ['params', 'kvtotal', 'ratio'],
    nh: ['dh'],
    sl: ['flops', 'kvtotal', 'ratio'],
    bs: ['kvtotal', 'ratio'],
    dt: ['kvtok', 'kvtotal', 'ratio'],
  }
  const targets = cardMap[k] ?? []
  for (const card of root.querySelectorAll<HTMLElement>('[data-acc-card]')) {
    const key = card.dataset.accCard ?? ''
    if (targets.includes(key)) {
      card.classList.add('acc-card--flash')
      window.setTimeout(() => card.classList.remove('acc-card--flash'), 480)
    }
  }
}

const accUpdateBar = (root: HTMLElement, m: AccMetrics) => {
  const totalParams = Math.max(1, m.params + m.ffnParams)
  const attnPct = (m.params / totalParams) * 100
  const ffnPct = (m.ffnParams / totalParams) * 100
  const totalBytes = Math.max(1, m.paramBytes + m.kvtotal)
  const kvPct = (m.kvtotal / totalBytes) * 100
  const paramBytesPct = (m.paramBytes / totalBytes) * 100

  const setBar = (k: string, pct: number) => {
    const el = root.querySelector<HTMLElement>(`[data-acc-bar="${k}"]`)
    if (el) el.style.width = `${pct.toFixed(1)}%`
  }
  setBar('attn', attnPct)
  setBar('ffn', ffnPct)
  setBar('kv', kvPct)
  setBar('paramBytes', paramBytesPct)

  accSetTex(root, '[data-acc-barval="attn"]', accFmtCount(m.params))
  accSetTex(root, '[data-acc-barval="ffn"]', accFmtCount(m.ffnParams))
  accSetTex(root, '[data-acc-barval="kv"]', accFmtBytes(m.kvtotal))
}

const accUpdateSummary = (root: HTMLElement, m: AccMetrics) => {
  accSetText(root, '[data-acc-sumkv]', accFmtBytes(m.kvtotal))
  accSetText(root, '[data-acc-sumparams]', accFmtBytes(m.paramBytes))
  accSetText(root, '[data-acc-sumratio]', `${m.ratio.toFixed(m.ratio >= 10 ? 0 : 2)}x`)
}

const accRender = (root: HTMLElement, s: AccState, prev: AccMetrics | null): AccMetrics => {
  const m = accCompute(s)
  accWriteState(root, s)
  accUpdateInputs(root, s)
  accUpdateCards(root, m, prev)
  accUpdateBar(root, m)
  accUpdateSummary(root, m)
  return m
}

const accSetup = () => {
  const roots = document.querySelectorAll<HTMLElement>('[data-acc-root]')
  for (const root of roots) {
    if (root.dataset.accBound === 'true') continue
    root.dataset.accBound = 'true'

    const state = accReadState(root)
    let prev: AccMetrics | null = null
    prev = accRender(root, state, prev)

    const offHandlers: Array<() => void> = []

    for (const k of Object.keys(accValueTable) as AccNumKey[]) {
      const input = root.querySelector<HTMLInputElement>(`[data-acc-input="${k}"]`)
      if (!input) continue
      const opts = accValueTable[k]
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
        prev = accRender(root, state, prev)
        accFlashCard(root, k)
      }
      input.addEventListener('input', handler)
      offHandlers.push(() => input.removeEventListener('input', handler))
    }

    const dtSelect = root.querySelector<HTMLSelectElement>('[data-acc-input="dt"]')
    if (dtSelect) {
      const handler = () => {
        const v = dtSelect.value as AccDType
        if (v in accDtypeBytes) {
          state.dt = v
          prev = accRender(root, state, prev)
          accFlashCard(root, 'dt')
        }
      }
      dtSelect.addEventListener('change', handler)
      offHandlers.push(() => dtSelect.removeEventListener('change', handler))
    }

    window.addCleanup?.(() => {
      for (const off of offHandlers) off()
      delete root.dataset.accBound
    })
  }
}

document.addEventListener('nav', accSetup)
