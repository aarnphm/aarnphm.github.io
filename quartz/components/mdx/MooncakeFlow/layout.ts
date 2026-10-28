// Two drawings of the same model. `wide` puts the conductor beside the instances; `tall` stacks
// everything in one column for frames under 600px, so labels keep 12px without a sideways scroll.

export type Box = { x: number; y: number; w: number; h: number }
export type Link = { key?: string; points: [number, number][] }
export type MooncakeLayout = {
  id: 'wide' | 'tall'
  viewW: number
  viewH: number
  prompt: { x: number; y: number; w: number; gap: number }
  conductorY: number
  scheds: Box[]
  prefill: Box[]
  decode: Box[]
  rdma: { x: number; y: number; r: number }
  rdmaLabel: { x: number; y: number; anchor: 'start' | 'end' }
  links: Link[]
}

export const PREFILL_H = 176
export const DECODE_H = 168
// The messenger strip sits at the bottom of a prefill box and the top of a decode box.
export const PREFILL_MSG_Y = PREFILL_H - 26
export const DECODE_MSG_Y = 26
export const MSG_H = 18

const rdmaWide = { x: 402, y: 230, r: 9 }

const wide: MooncakeLayout = {
  id: 'wide',
  viewW: 640,
  viewH: 452,
  prompt: { x: 12, y: 28, w: 15, gap: 2 },
  conductorY: 62,
  scheds: [72, 208, 328].map(y => ({ x: 12, y, w: 148, h: 44 })),
  prefill: [172, 410].map(x => ({ x, y: 12, w: 222, h: PREFILL_H })),
  decode: [172, 410].map(x => ({ x, y: 272, w: 222, h: DECODE_H })),
  rdma: rdmaWide,
  rdmaLabel: { x: 418, y: 234, anchor: 'start' },
  links: [
    {
      key: 'sched-prefill',
      points: [
        [160, 94],
        [172, 94],
      ],
    },
    {
      points: [
        [160, 230],
        [393, 230],
      ],
    },
    {
      key: 'sched-decode',
      points: [
        [160, 350],
        [172, 350],
      ],
    },
    {
      key: 'link-p0',
      points: [
        [283, 188],
        [402, 221],
      ],
    },
    {
      key: 'link-p1',
      points: [
        [521, 188],
        [402, 221],
      ],
    },
    {
      key: 'link-d0',
      points: [
        [402, 239],
        [283, 272],
      ],
    },
    {
      key: 'link-d1',
      points: [
        [402, 239],
        [521, 272],
      ],
    },
  ],
}

// Spines run down the right margin, one per instance, so no link crosses another box.
const TALL_P = [210, 398]
const TALL_D = [626, 806]
const rdmaTall = { x: 287, y: 600, r: 9 }
const rdmaTop = rdmaTall.y - Math.sqrt(rdmaTall.r ** 2 - 9)
const rdmaBottom = rdmaTall.y + Math.sqrt(rdmaTall.r ** 2 - 9)
const pMsg = (y: number) => y + PREFILL_MSG_Y + MSG_H / 2
const dMsg = (y: number) => y + DECODE_MSG_Y + MSG_H / 2

const tall: MooncakeLayout = {
  id: 'tall',
  viewW: 300,
  viewH: 986,
  prompt: { x: 12, y: 24, w: 30, gap: 3 },
  conductorY: 62,
  scheds: [70, 114, 158].map(y => ({ x: 12, y, w: 276, h: 40 })),
  prefill: TALL_P.map(y => ({ x: 12, y, w: 262, h: PREFILL_H })),
  decode: TALL_D.map(y => ({ x: 12, y, w: 262, h: DECODE_H })),
  rdma: rdmaTall,
  rdmaLabel: { x: 272, y: 604, anchor: 'end' },
  links: [
    {
      key: 'link-p0',
      points: [
        [274, pMsg(TALL_P[0])],
        [284, pMsg(TALL_P[0])],
        [284, rdmaTop],
      ],
    },
    {
      key: 'link-p1',
      points: [
        [274, pMsg(TALL_P[1])],
        [290, pMsg(TALL_P[1])],
        [290, rdmaTop],
      ],
    },
    {
      key: 'link-d0',
      points: [
        [284, rdmaBottom],
        [284, dMsg(TALL_D[0])],
        [274, dMsg(TALL_D[0])],
      ],
    },
    {
      key: 'link-d1',
      points: [
        [290, rdmaBottom],
        [290, dMsg(TALL_D[1])],
        [274, dMsg(TALL_D[1])],
      ],
    },
  ],
}

export const MOONCAKE_LAYOUTS = [wide, tall] as const
