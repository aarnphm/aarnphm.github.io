const VIEW_W = 520
const BOX_W = 36
const ROW_PAD = 28

const headCenterX = (i: number, h: number) => {
  const span = VIEW_W - 2 * ROW_PAD
  return ROW_PAD + (span / h) * (i + 0.5)
}

const kvCenterX = (g: number, nk: number) => {
  const span = VIEW_W - 2 * ROW_PAD
  return ROW_PAD + (span / nk) * (g + 0.5)
}

const ratioName = (r: number, h: number): string => {
  if (r === 1) return 'MHA'
  if (r === h) return 'MQA'
  return `GQA r=${r}`
}

const setKVLayout = (root: HTMLElement, h: number, r: number) => {
  const nk = h / r
  const span = (VIEW_W - 2 * ROW_PAD) / nk
  const w = Math.max(BOX_W, span - 12)

  const kvs = root.querySelectorAll<SVGGElement>('[data-kvg-kv]')
  for (const kv of kvs) {
    const g = Number(kv.dataset.kvgKv)
    if (g < nk) {
      kv.dataset.kvgHidden = 'false'
      const cx = kvCenterX(g, nk)
      const rect = kv.querySelector<SVGRectElement>('rect.kvg-box--kv')
      const fo = kv.querySelector<SVGForeignObjectElement>('foreignObject')
      if (rect) {
        rect.setAttribute('x', String(cx - w / 2))
        rect.setAttribute('width', String(w))
      }
      if (fo) {
        fo.setAttribute('x', String(cx - w / 2))
        fo.setAttribute('width', String(w))
      }
    } else {
      kv.dataset.kvgHidden = 'true'
    }
  }

  const queries = root.querySelectorAll<SVGGElement>('[data-kvg-query]')
  for (const q of queries) {
    const i = Number(q.dataset.kvgQuery)
    q.dataset.kvgGroup = String(Math.floor(i / r))
  }

  const links = root.querySelectorAll<SVGLineElement>('[data-kvg-link]')
  for (const link of links) {
    const i = Number(link.dataset.kvgLink)
    const g = Math.floor(i / r)
    link.dataset.kvgGroup = String(g)
    link.setAttribute('x1', String(headCenterX(i, h)))
    link.setAttribute('x2', String(kvCenterX(g, nk)))
  }
}

const setReadout = (root: HTMLElement, h: number, r: number) => {
  const nk = h / r
  const regime = ratioName(r, h)

  for (const el of root.querySelectorAll<HTMLElement>('[data-kvg-regime-r]')) {
    el.dataset.kvgActive = el.dataset.kvgRegimeR === String(r) ? 'true' : 'false'
  }

  for (const el of root.querySelectorAll<HTMLElement>('[data-kvg-nk-r]')) {
    el.dataset.kvgActive = el.dataset.kvgNkR === String(r) ? 'true' : 'false'
  }

  for (const el of root.querySelectorAll<HTMLElement>('[data-kvg-bw-r]')) {
    el.dataset.kvgActive = el.dataset.kvgBwR === String(r) ? 'true' : 'false'
  }

  for (const el of root.querySelectorAll<HTMLElement>('[data-kvg-cache-nk]')) {
    el.dataset.kvgActive = el.dataset.kvgCacheNk === String(nk) ? 'true' : 'false'
  }

  const ticks = root.querySelectorAll<HTMLElement>('[data-kvg-ratio-label]')
  for (const tick of ticks) {
    tick.dataset.kvgActive = Number(tick.dataset.kvgRatioLabel) === r ? 'true' : 'false'
  }

  const canvas = root.querySelector<SVGElement>('[data-kvg-canvas]')
  if (canvas) {
    canvas.setAttribute(
      'aria-label',
      `Query heads on top connecting to ${nk} shared KV box${nk === 1 ? '' : 'es'} below; current grouping ${regime}.`,
    )
  }
}

const setupKVHeadGrouping = () => {
  const roots = document.querySelectorAll<HTMLElement>('[data-kv-head-grouping]')
  for (const root of roots) {
    if (root.dataset.kvgBound === 'true') continue
    root.dataset.kvgBound = 'true'

    const h = Number(root.dataset.heads ?? '8')
    const ratios = (root.dataset.ratios ?? '1,2,4,8').split(',').map(Number)
    const slider = root.querySelector<HTMLInputElement>('[data-kvg-ratio]')
    if (!slider) continue

    const handleInput = () => {
      const idx = Number(slider.value)
      const r = ratios[idx] ?? 1
      slider.setAttribute('aria-valuenow', String(r))
      slider.setAttribute('aria-valuetext', ratioName(r, h))
      setReadout(root, h, r)
      setKVLayout(root, h, r)
    }

    slider.addEventListener('input', handleInput)
    handleInput()

    window.addCleanup(() => {
      slider.removeEventListener('input', handleInput)
      delete root.dataset.kvgBound
    })
  }
}

document.addEventListener('nav', setupKVHeadGrouping)
