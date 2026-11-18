import { docOfRow, partOf, textOf, type Runtime } from '../runtime'

/** a chunk's rendered text aligned to its raw text (both in UTF-16 units) */
export interface Aligned {
  row: number
  raw: string
  nodes: { node: Text; off: number }[]
  rawToDom: Int32Array
  domToRaw: Int32Array
}

// ---- alignment ------------------------------------------------------------------------------

const SPACE = /\s/

/** the chunk's own text nodes: the locus (chunk number, page) is not passage text */
function textNodes(chunk: HTMLElement): Text[] {
  const out: Text[] = []
  const walker = document.createTreeWalker(chunk, NodeFilter.SHOW_TEXT, {
    acceptNode: node =>
      (node.parentElement?.closest('.lat-locus, .katex-mathml, button') ?? null)
        ? NodeFilter.FILTER_REJECT
        : NodeFilter.FILTER_ACCEPT,
  })
  for (let n = walker.nextNode(); n; n = walker.nextNode()) out.push(n as Text)
  return out
}

/**
 * Raw markdown and rendered text, aligned by their non-space characters. A raw character maps
 * where the next three raw characters appear within a short window of the rendered text, so
 * markup that never renders (`**`, `[[target|`, `$`) is skipped instead of dragging the match
 * forward onto a later word.
 */
export function align(chunk: HTMLElement, row: number, raw: string): Aligned {
  const nodes: { node: Text; off: number }[] = []
  let dom = ''
  for (const node of textNodes(chunk)) {
    const t = node.data
    for (let i = 0; i < t.length; i++) {
      nodes.push({ node, off: i })
      dom += t[i]
    }
  }
  const rawToDom = new Int32Array(raw.length).fill(-1)
  const domToRaw = new Int32Array(dom.length).fill(-1)
  const rawIdx: number[] = []
  const domIdx: number[] = []
  for (let i = 0; i < raw.length; i++) if (!SPACE.test(raw[i])) rawIdx.push(i)
  for (let i = 0; i < dom.length; i++) if (!SPACE.test(dom[i])) domIdx.push(i)
  const R = rawIdx.map(i => raw[i])
  const D = domIdx.map(i => dom[i])
  const WINDOW = 48
  let j = 0
  for (let i = 0; i < R.length && j < D.length; i++) {
    let hit = -1
    if (R[i] === D[j]) hit = j
    else
      for (let k = j + 1; k < Math.min(D.length, j + WINDOW); k++) {
        if (D[k] === R[i] && D[k + 1] === R[i + 1] && D[k + 2] === R[i + 2]) {
          hit = k
          break
        }
      }
    if (hit < 0) continue
    rawToDom[rawIdx[i]] = domIdx[hit]
    domToRaw[domIdx[hit]] = rawIdx[i]
    j = hit + 1
  }
  return { row, raw, nodes, rawToDom, domToRaw }
}

export function domRange(al: Aligned, s: number, e: number): Range | null {
  let a = -1
  let b = -1
  for (let i = s; i < e; i++) {
    const d = al.rawToDom[i]
    if (d < 0) continue
    if (a < 0) a = d
    b = d
  }
  if (a < 0) return null
  const range = document.createRange()
  range.setStart(al.nodes[a].node, al.nodes[a].off)
  range.setEnd(al.nodes[b].node, al.nodes[b].off + 1)
  return range
}

/** dom (node, offset) -> index into al.nodes */
function domIndex(al: Aligned, node: Node, off: number, end: boolean): number {
  if (node.nodeType !== Node.TEXT_NODE) {
    const child = node.childNodes[Math.min(off, node.childNodes.length - 1)]
    if (!child) return end ? al.nodes.length : 0
    const walker = document.createTreeWalker(child, NodeFilter.SHOW_TEXT)
    const first = (child.nodeType === Node.TEXT_NODE ? child : walker.nextNode()) as Text | null
    if (!first) return end ? al.nodes.length : 0
    node = first
    off = 0
  }
  for (let i = 0; i < al.nodes.length; i++) {
    const n = al.nodes[i]
    if (n.node === node && n.off >= off) return i
  }
  return end ? al.nodes.length : 0
}

/** the raw range a dom selection covers, through the nearest aligned characters */
export function rawRange(al: Aligned, range: Range): [number, number] | null {
  const a = domIndex(al, range.startContainer, range.startOffset, false)
  const b = domIndex(al, range.endContainer, range.endOffset, true)
  let s = -1
  let e = -1
  for (let i = a; i < Math.min(b, al.domToRaw.length); i++) {
    const r = al.domToRaw[i]
    if (r < 0) continue
    if (s < 0) s = r
    e = r + 1
  }
  return s < 0 ? null : [s, e]
}

export async function chunkText(rt: Runtime, row: number): Promise<string | null> {
  const doc = docOfRow(rt, row)
  if (!doc) return null
  const head = await textOf(rt, doc.d)
  const c = row - doc.r0
  const chunks = await partOf(rt, doc.d, head.part ? Math.floor(c / head.part) : 0)
  return chunks.find(ch => ch.r === row)?.text ?? null
}
