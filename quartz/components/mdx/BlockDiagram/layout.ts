import { blockTex, isTall, texLen, type Block, type Diagram, type El } from './model'

const Y0 = 46
const GAP = 52
export const SR = 10
export const SW = 30
const BH = 34
const BH_TALL = 50
const FB_GAP = 84
const UNITY_GAP = 46
// A parallel branch runs this far above the main row.
const PAR_RISE = 58
const CH = 7.4

const blockW = (b: Block) => Math.max(56, Math.round(texLen(blockTex(b)) * CH + 22))
const blockH = (b: Block) => (isTall(blockTex(b)) ? BH_TALL : BH)
const elW = (e: El) => (e.k === 'block' ? blockW(e) : e.k === 'sampler' ? SW : 2 * SR)

type Shape =
  | { t: 'wire'; d: string; arrow: boolean }
  | { t: 'label'; x: number; y: number; w: number; tex: string; anchor: 'mid' | 'start' | 'end' }
  | { t: 'block'; id: string; x: number; y: number; w: number; h: number; tex: string }
  | { t: 'sum'; id: string; cx: number; cy: number; signs: [string, number, number][] }
  | { t: 'sampler'; id: string; x: number; y: number; dir: 1 | -1 }
  | { t: 'dot'; cx: number; cy: number }

export const LABEL_H = 22

export const layout = (d: Diagram): { shapes: Shape[]; w: number; h: number } => {
  const shapes: Shape[] = []
  const Y = Y0 + (d.chain.some(e => e.k === 'par') ? PAR_RISE - 16 : 0)
  const label = (x0: number, x1: number, y: number, t?: string) => {
    if (!t) return
    const w = Math.max(x1 - x0, texLen(t) * CH + 12)
    shapes.push({
      t: 'label',
      x: (x0 + x1) / 2 - w / 2,
      y: y - LABEL_H - 3,
      w,
      tex: t,
      anchor: 'mid',
    })
  }
  let x = 6
  let lastLabel: string | undefined = d.input
  let sumX = 0
  let bottom = Y + BH_TALL / 2

  for (const e of d.chain) {
    const x1 = x + GAP
    if (e.k === 'sampler') {
      shapes.push({ t: 'wire', d: `M${x} ${Y}H${x1}`, arrow: false })
      label(x, x1, Y, lastLabel)
      shapes.push({ t: 'sampler', id: e.id, x: x1, y: Y, dir: 1 })
      x = x1 + SW
    } else if (e.k === 'sum') {
      const cx = x1 + SR
      shapes.push({ t: 'wire', d: `M${x} ${Y}H${x1}`, arrow: true })
      label(x, x1, Y, lastLabel)
      shapes.push({
        t: 'sum',
        id: e.id,
        cx,
        cy: Y,
        signs: [
          [e.signs[0], cx - SR - 7, Y + SR + 9],
          [e.signs[1], cx + SR + 7, Y + SR + 11],
        ],
      })
      sumX = cx
      if (e.side) {
        let y = Y + SR
        let yy = Y + SR + 36
        const parts: string[] = []
        for (const b of e.side.chain) {
          const h = blockH(b)
          const w = blockW(b)
          shapes.push({ t: 'wire', d: `M${cx} ${yy}V${y}`, arrow: true })
          shapes.push({ t: 'block', id: b.id, x: cx - w / 2, y: yy, w, h, tex: blockTex(b) })
          parts.push(b.id)
          y = yy + h
          yy = y + 36
        }
        shapes.push({ t: 'wire', d: `M${cx} ${yy}V${y}`, arrow: true })
        const lw = texLen(e.side.input) * CH + 12
        shapes.push({
          t: 'label',
          x: cx - lw / 2,
          y: yy + 2,
          w: lw,
          tex: e.side.input,
          anchor: 'mid',
        })
        bottom = Math.max(bottom, yy + LABEL_H + 4)
      }
      x = cx + SR
    } else if (e.k === 'par') {
      const [main, up] = e.branches
      const bw = Math.max(blockW(main), blockW(up))
      const hm = blockH(main)
      const hu = blockH(up)
      const yU = Y - PAR_RISE
      const bx = x1 + Math.round(GAP * 0.6)
      const cx = bx + bw + GAP + SR
      shapes.push({ t: 'wire', d: `M${x} ${Y}H${x1}`, arrow: false })
      label(x, x1, Y, lastLabel)
      shapes.push({ t: 'wire', d: `M${x1} ${Y}H${bx}`, arrow: true })
      shapes.push({ t: 'wire', d: `M${x1} ${Y}V${yU}H${bx}`, arrow: true })
      shapes.push({ t: 'dot', cx: x1, cy: Y })
      shapes.push({
        t: 'block',
        id: main.id,
        x: bx,
        y: Y - hm / 2,
        w: bw,
        h: hm,
        tex: blockTex(main),
      })
      shapes.push({ t: 'block', id: up.id, x: bx, y: yU - hu / 2, w: bw, h: hu, tex: blockTex(up) })
      shapes.push({ t: 'wire', d: `M${bx + bw} ${Y}H${cx - SR}`, arrow: true })
      shapes.push({ t: 'wire', d: `M${bx + bw} ${yU}H${cx}V${Y - SR}`, arrow: true })
      shapes.push({
        t: 'sum',
        id: `${e.id}-sum`,
        cx,
        cy: Y,
        signs: [
          [e.signs[0], cx - SR - 7, Y - 5],
          [e.signs[1], cx + 8, Y - SR - 5],
        ],
      })
      x = cx + SR
    } else {
      const w = blockW(e)
      const h = blockH(e)
      shapes.push({ t: 'wire', d: `M${x} ${Y}H${x1}`, arrow: true })
      label(x, x1, Y, lastLabel)
      shapes.push({ t: 'block', id: e.id, x: x1, y: Y - h / 2, w, h, tex: blockTex(e) })
      x = x1 + w
    }
    lastLabel = e.out
  }

  if (!d.loop) {
    const x1 = x + GAP
    shapes.push({ t: 'wire', d: `M${x} ${Y}H${x1}`, arrow: true })
    label(x, x1, Y, lastLabel ?? d.output)
    return { shapes, w: x1 + 8, h: bottom + 8 }
  }

  const fb = d.loop.chain
  const widths = fb.map(elW)
  const fbW = widths.reduce((s, w) => s + w, 0)
  // The feedback row needs a labelled wire gap on both sides of each element.
  const xt = Math.max(x + Math.round(GAP * 0.8), sumX + fbW + (fb.length + 1) * GAP)
  const xo = xt + 46
  shapes.push({ t: 'wire', d: `M${x} ${Y}H${xt}`, arrow: false })
  label(x, xt, Y, lastLabel)
  shapes.push({ t: 'wire', d: `M${xt} ${Y}H${xo}`, arrow: true })
  label(xt, xo, Y, d.output)
  shapes.push({ t: 'dot', cx: xt, cy: Y })

  const yF = Y + (fb.length ? FB_GAP : UNITY_GAP)
  const free = xt - sumX - fbW
  const gap = free / (fb.length + 1)
  let cur = xt
  let fbLabel = d.loop.inLabel
  let first = true
  for (const [i, e] of fb.entries()) {
    const right = cur - gap
    const start = first ? `M${xt} ${Y}V${yF}H${right}` : `M${cur} ${yF}H${right}`
    first = false
    const fl = (x0: number, x1: number, t?: string) => {
      if (!t) return
      const w = Math.max(x1 - x0, texLen(t) * CH + 12)
      shapes.push({
        t: 'label',
        x: (x0 + x1) / 2 - w / 2,
        y: yF - LABEL_H - 3,
        w,
        tex: t,
        anchor: 'mid',
      })
    }
    if (e.k === 'sampler') {
      shapes.push({ t: 'wire', d: start, arrow: false })
      fl(right, cur, fbLabel)
      shapes.push({ t: 'sampler', id: e.id, x: right, y: yF, dir: -1 })
      cur = right - SW
    } else if (e.k === 'block') {
      const h = blockH(e)
      shapes.push({ t: 'wire', d: start, arrow: true })
      fl(right, cur, fbLabel)
      shapes.push({
        t: 'block',
        id: e.id,
        x: right - widths[i],
        y: yF - h / 2,
        w: widths[i],
        h,
        tex: blockTex(e),
      })
      cur = right - widths[i]
      bottom = Math.max(bottom, yF + h / 2)
    }
    fbLabel = e.out
  }
  const back = fb.length ? `M${cur} ${yF}` : `M${xt} ${Y}V${yF}`
  shapes.push({ t: 'wire', d: `${back}H${sumX}V${Y + SR}`, arrow: true })
  if (fbLabel) {
    const w = Math.max(cur - sumX, texLen(fbLabel) * CH + 12)
    shapes.push({
      t: 'label',
      x: (sumX + cur) / 2 - w / 2,
      y: yF - LABEL_H - 3,
      w,
      tex: fbLabel,
      anchor: 'mid',
    })
  }
  bottom = fb.length ? Math.max(bottom, yF + BH_TALL / 2, yF + LABEL_H + 8) : yF + 8
  return { shapes, w: xo + 8, h: bottom + 8 }
}
