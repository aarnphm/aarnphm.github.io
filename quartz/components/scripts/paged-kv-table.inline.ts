export {}

type PktSeqId = 'a' | 'b'

type PktBlock = { blk: number; filled: number; phys: number | null }

type PktSeqState = { id: PktSeqId; tokens: number; blocks: PktBlock[] }

type PktState = { seqs: PktSeqState[]; blockSize: number; physicalSlots: number }

const PKT_BLOCK_TONE: Record<PktSeqId, 'salmon' | 'sage'> = { a: 'salmon', b: 'sage' }

const pktClone = (state: PktState): PktState => ({
  seqs: state.seqs.map(s => ({
    id: s.id,
    tokens: s.tokens,
    blocks: s.blocks.map(b => ({ ...b })),
  })),
  blockSize: state.blockSize,
  physicalSlots: state.physicalSlots,
})

const pktReadInitial = (root: HTMLElement): PktState | null => {
  const raw = root.dataset.pktInitial
  if (!raw) return null
  try {
    return JSON.parse(raw) as PktState
  } catch {
    return null
  }
}

const pktOccupantOf = (state: PktState, phys: number): { seq: PktSeqId; blk: number } | null => {
  for (const seq of state.seqs) {
    for (const block of seq.blocks) {
      if (block.phys === phys) return { seq: seq.id, blk: block.blk }
    }
  }
  return null
}

const pktFirstFree = (state: PktState): number | null => {
  for (let i = 0; i < state.physicalSlots; i++) {
    if (!pktOccupantOf(state, i)) return i
  }
  return null
}

const pktSeqState = (state: PktState, id: PktSeqId): PktSeqState | undefined =>
  state.seqs.find(s => s.id === id)

const pktTotalAllocatedBlocks = (state: PktState): number => {
  let n = 0
  for (const seq of state.seqs) {
    for (const block of seq.blocks) {
      if (block.phys !== null) n += 1
    }
  }
  return n
}

const pktTotalTokens = (state: PktState): number => {
  let n = 0
  for (const seq of state.seqs) n += seq.tokens
  return n
}

const pktFormatFrag = (state: PktState): string => {
  const alloc = pktTotalAllocatedBlocks(state)
  const tokens = pktTotalTokens(state)
  if (alloc === 0) return '0%'
  const denom = alloc * state.blockSize
  const frag = 1 - tokens / denom
  const pct = Math.max(0, frag) * 100
  return pct < 10 ? `${pct.toFixed(1)}%` : `${pct.toFixed(0)}%`
}

const pktRenderLogical = (root: HTMLElement, state: PktState) => {
  const wraps = root.querySelectorAll<HTMLElement>('[data-pkt-seq-blocks]')
  for (const wrap of wraps) {
    const seqId = wrap.dataset.pktSeqBlocks as PktSeqId | undefined
    if (!seqId) continue
    const seq = pktSeqState(state, seqId)
    if (!seq) continue
    const existing = Array.from(wrap.querySelectorAll<HTMLElement>('[data-pkt-lblock]'))
    while (existing.length > seq.blocks.length) {
      const last = existing.pop()
      last?.remove()
    }
    for (let i = 0; i < seq.blocks.length; i++) {
      const block = seq.blocks[i]
      let el = existing[i]
      if (!el) {
        const btn = document.createElement('button')
        btn.type = 'button'
        btn.classList.add('pkt-lblock', `pkt-lblock--${PKT_BLOCK_TONE[seqId]}`)
        btn.dataset.pktLblock = ''
        btn.dataset.pktSeq = seqId
        btn.innerHTML = `
          <span class="pkt-lblock-tag">${seqId}<sub>${block.blk}</sub></span>
          <span class="pkt-lblock-cells" aria-hidden="true"></span>
          <span class="pkt-lblock-meta" data-pkt-lblock-meta></span>
        `
        wrap.appendChild(btn)
        el = btn
        existing.push(el)
      }
      el.dataset.pktBlk = String(block.blk)
      el.dataset.pktMapped = block.phys !== null ? 'true' : 'false'
      const filled = block.filled
      el.setAttribute(
        'aria-label',
        `seq ${seqId.toUpperCase()} logical block ${block.blk}, ${filled} of ${state.blockSize} tokens, ${
          block.phys !== null ? `mapped to physical slot ${block.phys}` : 'paged out'
        }`,
      )
      const cells = el.querySelector<HTMLElement>('.pkt-lblock-cells')
      if (cells) {
        const haveCount = cells.children.length
        if (haveCount !== state.blockSize) {
          cells.innerHTML = ''
          for (let k = 0; k < state.blockSize; k++) {
            const span = document.createElement('span')
            span.className = 'pkt-tok'
            span.dataset.pktTok = String(block.blk * state.blockSize + k)
            cells.appendChild(span)
          }
        }
        Array.from(cells.children).forEach((tok, k) => {
          tok.classList.toggle('is-filled', k < filled)
        })
      }
      const meta = el.querySelector<HTMLElement>('[data-pkt-lblock-meta]')
      if (meta) meta.textContent = block.phys !== null ? `to p${block.phys}` : 'paged out'
    }
  }
}

const pktRenderPageTable = (root: HTMLElement, state: PktState) => {
  const tbody = root.querySelector<HTMLElement>('[data-pkt-pt-body]')
  if (!tbody) return
  tbody.innerHTML = ''
  for (const seq of state.seqs) {
    for (const block of seq.blocks) {
      if (block.phys === null) continue
      const tr = document.createElement('tr')
      tr.className = 'pkt-pt-row'
      tr.dataset.pktPtRow = ''
      tr.dataset.pktSeq = seq.id
      tr.dataset.pktBlk = String(block.blk)
      tr.dataset.pktPhys = String(block.phys)
      tr.innerHTML = `
        <td class="pkt-pt-key"><span class="pkt-pt-id">${seq.id}<sub>${block.blk}</sub></span></td>
        <td class="pkt-pt-arrow" aria-hidden="true">to</td>
        <td class="pkt-pt-val" data-pkt-pt-val>p${block.phys}</td>
      `
      tbody.appendChild(tr)
    }
  }
}

const pktRenderPhysical = (root: HTMLElement, state: PktState, landingPhys?: number) => {
  const slots = root.querySelectorAll<HTMLElement>('[data-pkt-pblock]')
  for (const slot of slots) {
    const idx = Number(slot.dataset.pktPhys ?? '-1')
    const occ = pktOccupantOf(state, idx)
    slot.classList.remove('pkt-pblock--salmon', 'pkt-pblock--sage', 'pkt-pblock--free')
    slot.classList.remove('is-landing')
    if (occ) {
      slot.classList.add(`pkt-pblock--${PKT_BLOCK_TONE[occ.seq]}`)
      slot.dataset.pktOccupied = 'true'
      slot.dataset.pktOccSeq = occ.seq
      slot.dataset.pktOccBlk = String(occ.blk)
    } else {
      slot.classList.add('pkt-pblock--free')
      slot.dataset.pktOccupied = 'false'
      slot.dataset.pktOccSeq = ''
      slot.dataset.pktOccBlk = ''
    }
    const label = slot.querySelector<HTMLElement>('[data-pkt-pblock-label]')
    if (label) label.textContent = occ ? `${occ.seq}${occ.blk}` : 'free'
    const evict = slot.querySelector<HTMLButtonElement>('[data-pkt-evict]')
    if (evict) evict.disabled = !occ
    if (landingPhys === idx) {
      void slot.offsetWidth
      slot.classList.add('is-landing')
    }
  }
}

const pktRenderReadout = (root: HTMLElement, state: PktState) => {
  const seqA = pktSeqState(state, 'a')
  const seqB = pktSeqState(state, 'b')
  const tokA = root.querySelector<HTMLElement>('[data-pkt-tokens-a]')
  const tokB = root.querySelector<HTMLElement>('[data-pkt-tokens-b]')
  if (tokA && seqA) tokA.textContent = String(seqA.tokens)
  if (tokB && seqB) tokB.textContent = String(seqB.tokens)
  const alloc = root.querySelector<HTMLElement>('[data-pkt-alloc]')
  if (alloc) alloc.textContent = String(pktTotalAllocatedBlocks(state))
  const frag = root.querySelector<HTMLElement>('[data-pkt-frag]')
  if (frag) frag.textContent = pktFormatFrag(state)
}

const pktClearActive = (root: HTMLElement) => {
  for (const el of root.querySelectorAll('.is-active')) el.classList.remove('is-active')
}

const pktHighlight = (root: HTMLElement, state: PktState, seq: PktSeqId, blk: number) => {
  pktClearActive(root)
  const phys = pktSeqState(state, seq)?.blocks.find(b => b.blk === blk)?.phys
  root
    .querySelector<HTMLElement>(`[data-pkt-lblock][data-pkt-seq="${seq}"][data-pkt-blk="${blk}"]`)
    ?.classList.add('is-active')
  root
    .querySelector<HTMLElement>(`[data-pkt-pt-row][data-pkt-seq="${seq}"][data-pkt-blk="${blk}"]`)
    ?.classList.add('is-active')
  if (phys !== null && phys !== undefined) {
    root
      .querySelector<HTMLElement>(`[data-pkt-pblock][data-pkt-phys="${phys}"]`)
      ?.classList.add('is-active')
  }
}

const pktRender = (root: HTMLElement, state: PktState, landingPhys?: number) => {
  pktRenderLogical(root, state)
  pktRenderPageTable(root, state)
  pktRenderPhysical(root, state, landingPhys)
  pktRenderReadout(root, state)
}

const pktBindRoot = (root: HTMLElement) => {
  if (root.dataset.pktBound === 'true') return
  const initial = pktReadInitial(root)
  if (!initial) return
  root.dataset.pktBound = 'true'
  let state = pktClone(initial)
  pktRender(root, state)

  const onClick = (event: Event) => {
    const target = event.target as HTMLElement | null
    if (!target) return

    const evictBtn = target.closest<HTMLButtonElement>('[data-pkt-evict]')
    if (evictBtn) {
      const slot = evictBtn.closest<HTMLElement>('[data-pkt-pblock]')
      if (!slot) return
      const phys = Number(slot.dataset.pktPhys ?? '-1')
      const occ = pktOccupantOf(state, phys)
      if (!occ) return
      const seq = pktSeqState(state, occ.seq)
      if (!seq) return
      const block = seq.blocks.find(b => b.blk === occ.blk)
      if (!block) return
      block.phys = null
      pktRender(root, state)
      return
    }

    const appendBtn = target.closest<HTMLButtonElement>('[data-pkt-append]')
    if (appendBtn) {
      const seqId = appendBtn.dataset.pktSeq as PktSeqId | undefined
      if (!seqId) return
      const seq = pktSeqState(state, seqId)
      if (!seq) return
      let landing: number | undefined
      const last = seq.blocks[seq.blocks.length - 1]
      if (last && last.phys !== null && last.filled < state.blockSize) {
        last.filled += 1
        seq.tokens += 1
      } else {
        const free = pktFirstFree(state)
        if (free === null) return
        const newBlkIdx = (last?.blk ?? -1) + 1
        seq.blocks.push({ blk: newBlkIdx, filled: 1, phys: free })
        seq.tokens += 1
        landing = free
      }
      pktRender(root, state, landing)
      return
    }

    const lblock = target.closest<HTMLElement>('[data-pkt-lblock]')
    if (lblock) {
      const seqId = lblock.dataset.pktSeq as PktSeqId | undefined
      const blk = Number(lblock.dataset.pktBlk ?? '-1')
      if (!seqId || !Number.isInteger(blk)) return
      pktHighlight(root, state, seqId, blk)
      return
    }

    const resetBtn = target.closest<HTMLButtonElement>('[data-pkt-reset]')
    if (resetBtn) {
      state = pktClone(initial)
      pktRender(root, state)
      pktClearActive(root)
      return
    }
  }

  root.addEventListener('click', onClick)
  window.addCleanup(() => {
    root.removeEventListener('click', onClick)
    delete root.dataset.pktBound
  })
}

const pktSetup = () => {
  const roots = document.querySelectorAll<HTMLElement>('[data-paged-kv-table]')
  for (const root of roots) pktBindRoot(root)
}

document.addEventListener('nav', pktSetup)
