import katex from 'katex'

const mhaViewH = 420
const mhaBandTop = 36
const mhaBandBot = mhaViewH - 36
const mhaBandH = mhaBandBot - mhaBandTop
const mhaProjW = 52
const mhaProjH = 14
const mhaProjGap = 4
const mhaScoreBox = 52
const mhaHeadOutW = 56
const mhaHeadOutH = 22
const mhaColX = { input: 60, proj: 200, score: 332, head: 472, concat: 612, wo: 700 } as const

const mhaHeadMid = (i: number, h: number): number => {
  const slot = mhaBandH / h
  return mhaBandTop + slot * i + slot / 2
}

const mhaFmtBytes = (n: number): string => {
  if (n >= 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MiB`
  if (n >= 1024) return `${(n / 1024).toFixed(1)} KiB`
  return `${n} B`
}

const mhaFmtCompact = (n: number): string => {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}G`
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k`
  return String(n)
}

const mhaSetAttr = (el: Element | null, name: string, value: string) => {
  if (el) el.setAttribute(name, value)
}

const mhaTex = (v: string): string => {
  const m = v.match(/^([\d.,\s]*)(.*)$/)
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

const mhaSetText = (root: HTMLElement, selector: string, text: string) => {
  const el = root.querySelector(selector)
  if (el) el.innerHTML = mhaTex(text)
}

const mhaSetLayout = (root: HTMLElement, h: number, showPattern: boolean) => {
  const headGroups = root.querySelectorAll<SVGGElement>('[data-mha-head]')
  for (const g of headGroups) {
    const i = Number(g.dataset.mhaHead)
    const visible = i < h
    g.dataset.mhaHidden = visible ? 'false' : 'true'
    if (!visible) continue

    const mid = mhaHeadMid(i, h)
    const projTopY = mid - (mhaProjH * 3 + mhaProjGap * 2) / 2

    const projs = g.querySelectorAll<SVGGElement>('[data-mha-projs] > g')
    projs.forEach((projGroup, k) => {
      const py = projTopY + k * (mhaProjH + mhaProjGap)
      const rect = projGroup.querySelector<SVGRectElement>('rect.mha-proj')
      const fo = projGroup.querySelector<SVGForeignObjectElement>('foreignObject')
      mhaSetAttr(rect, 'y', String(py))
      mhaSetAttr(fo, 'y', String(py))
    })

    const projLink = g.querySelector<SVGLineElement>('.mha-link--proj-score')
    mhaSetAttr(projLink, 'y1', String(mid))
    mhaSetAttr(projLink, 'y2', String(mid))

    const scoreGroup = g.querySelector<SVGGElement>('[data-mha-score]')
    if (scoreGroup) {
      scoreGroup.dataset.mhaScoreVisible = showPattern ? 'true' : 'false'
      const scoreY = mid - mhaScoreBox / 2
      const frame = scoreGroup.querySelector<SVGRectElement>('.mha-score-frame')
      mhaSetAttr(frame, 'y', String(scoreY))
      const cells = scoreGroup.querySelectorAll<SVGRectElement>('.mha-score-cell')
      const cellPx = mhaScoreBox / 4
      cells.forEach((cell, idx) => {
        const r = Math.floor(idx / 4)
        cell.setAttribute('y', String(scoreY + r * cellPx))
      })
      const labelFO = scoreGroup.querySelector<SVGForeignObjectElement>('foreignObject')
      mhaSetAttr(labelFO, 'y', String(scoreY - 14))
    }

    const collapsedGroup = g.querySelector<SVGGElement>('[data-mha-score-collapsed]')
    if (collapsedGroup) {
      collapsedGroup.dataset.mhaCollapsedVisible = showPattern ? 'false' : 'true'
      const collapsedRect = collapsedGroup.querySelector<SVGRectElement>('.mha-box--collapsed')
      const collapsedFO = collapsedGroup.querySelector<SVGForeignObjectElement>('foreignObject')
      mhaSetAttr(collapsedRect, 'y', String(mid - 11))
      mhaSetAttr(collapsedFO, 'y', String(mid - 11))
    }

    const outLink = g.querySelector<SVGLineElement>('.mha-link--score-out')
    mhaSetAttr(outLink, 'y1', String(mid))
    mhaSetAttr(outLink, 'y2', String(mid))

    const headRect = g.querySelector<SVGRectElement>('[data-mha-head-rect]')
    const headFO = g.querySelector<SVGForeignObjectElement>('[data-mha-head-label]')
    const headOutY = mid - mhaHeadOutH / 2
    mhaSetAttr(headRect, 'y', String(headOutY))
    mhaSetAttr(headFO, 'y', String(headOutY))
  }

  const fanoutLines = root.querySelectorAll<SVGPathElement>('[data-mha-fanout-line]')
  for (const line of fanoutLines) {
    const i = Number(line.dataset.mhaFanoutLine)
    const visible = i < h
    line.dataset.mhaHidden = visible ? 'false' : 'true'
    if (!visible) continue
    const mid = mhaHeadMid(i, h)
    line.setAttribute(
      'd',
      `M ${mhaColX.input + 36} ${mhaViewH / 2} C ${mhaColX.input + 70} ${mhaViewH / 2}, ${mhaColX.proj - 60} ${mid}, ${mhaColX.proj - mhaProjW / 2 - 4} ${mid}`,
    )
  }

  const concatLines = root.querySelectorAll<SVGPathElement>('[data-mha-concat-line]')
  for (const line of concatLines) {
    const i = Number(line.dataset.mhaConcatLine)
    const visible = i < h
    line.dataset.mhaHidden = visible ? 'false' : 'true'
    if (!visible) continue
    const mid = mhaHeadMid(i, h)
    line.setAttribute(
      'd',
      `M ${mhaColX.head + mhaHeadOutW / 2} ${mid} C ${mhaColX.head + 60} ${mid}, ${mhaColX.concat - 30} ${mhaViewH / 2}, ${mhaColX.concat - 22} ${mhaViewH / 2}`,
    )
  }
}

const mhaSetReadout = (root: HTMLElement, h: number, dm: number, seq: number) => {
  const dh = h > 0 ? Math.round(dm / h) : 0
  const params = 4 * dm * dm
  const cachePerToken = 2 * dm * 2
  const flops = 4 * seq * seq * dm

  mhaSetText(root, '[data-mha-dh]', String(dh))
  mhaSetText(root, '[data-mha-params]', mhaFmtCompact(params))
  mhaSetText(root, '[data-mha-cache]', mhaFmtBytes(cachePerToken))
  mhaSetText(root, '[data-mha-flops]', mhaFmtCompact(flops))

  const ticks = root.querySelectorAll<HTMLElement>('[data-mha-tick]')
  for (const tick of ticks) {
    tick.dataset.mhaActive = Number(tick.dataset.mhaTick) === h ? 'true' : 'false'
  }

  const valueEl = root.querySelector<HTMLElement>('[data-mha-heads-value]')
  if (valueEl) valueEl.textContent = `h=${h}`

  const slider = root.querySelector<HTMLInputElement>('[data-mha-heads-input]')
  if (slider) {
    slider.setAttribute('aria-valuenow', String(h))
    slider.setAttribute('aria-valuetext', `${h} heads`)
  }

  const canvas = root.querySelector<SVGElement>('[data-mha-canvas]')
  mhaSetAttr(
    canvas,
    'aria-label',
    `Multi-head attention with ${h} head${h === 1 ? '' : 's'}: input x splits into per-head Q, K, V projections, each head runs an independent softmax over a per-head attention pattern, outputs are concatenated and projected by W_O, then added back to the residual stream.`,
  )
}

const mhaParseOptions = (raw: string | undefined, fallback: number[]): number[] => {
  if (!raw) return fallback
  const parsed = raw
    .split(',')
    .map(s => Number(s.trim()))
    .filter(n => Number.isFinite(n) && n > 0)
  return parsed.length > 0 ? parsed : fallback
}

const mhaSetup = () => {
  const roots = document.querySelectorAll<HTMLElement>('[data-multi-head-attention]')
  for (const root of roots) {
    if (root.dataset.mhaBound === 'true') continue
    root.dataset.mhaBound = 'true'

    const options = mhaParseOptions(root.dataset.mhaOptions, [1, 2, 4, 8])
    const seq = Number(root.dataset.mhaSeq ?? '4')
    let h = Number(root.dataset.mhaHeads ?? String(options[0]))
    let dm = Number(root.dataset.mhaDm ?? '512')
    let showPattern = root.dataset.mhaShowPattern !== 'false'

    const slider = root.querySelector<HTMLInputElement>('[data-mha-heads-input]')
    const dmInput = root.querySelector<HTMLInputElement>('[data-mha-dm-input]')
    const toggle = root.querySelector<HTMLButtonElement>('[data-mha-pattern-toggle]')

    const render = () => {
      mhaSetLayout(root, h, showPattern)
      mhaSetReadout(root, h, dm, seq)
    }

    render()

    const handlers: Array<() => void> = []

    if (slider) {
      const onSlide = () => {
        const idx = Number(slider.value)
        const next = options[idx] ?? options[0]
        h = next
        root.dataset.mhaHeads = String(h)
        render()
      }
      slider.addEventListener('input', onSlide)
      handlers.push(() => slider.removeEventListener('input', onSlide))
    }

    if (dmInput) {
      const onDm = () => {
        const raw = Number(dmInput.value)
        if (!Number.isFinite(raw) || raw <= 0) return
        dm = Math.round(raw)
        root.dataset.mhaDm = String(dm)
        render()
      }
      dmInput.addEventListener('input', onDm)
      handlers.push(() => dmInput.removeEventListener('input', onDm))
    }

    if (toggle) {
      const onToggle = () => {
        showPattern = !showPattern
        root.dataset.mhaShowPattern = showPattern ? 'true' : 'false'
        toggle.classList.toggle('is-active', showPattern)
        toggle.setAttribute('aria-pressed', showPattern ? 'true' : 'false')
        render()
      }
      toggle.addEventListener('click', onToggle)
      handlers.push(() => toggle.removeEventListener('click', onToggle))
    }

    window.addCleanup(() => {
      for (const off of handlers) off()
      delete root.dataset.mhaBound
    })
  }
}

document.addEventListener('nav', mhaSetup)
