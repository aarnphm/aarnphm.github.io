import type { Runtime } from './runtime'
import { linkItems, renderDocumentLinks } from './document-list'
import { renderRail } from './search'

// ---- links pane ----------------------------------------------------------------------------

export async function pinLinks(
  rt: Runtime,
  label: string,
  rows: [number, number][],
): Promise<void> {
  if (rt.signal.aborted) return
  // a feature's cluster knows no kinds: the box's `filter:` narrows it here
  const kinds = rt.filter?.kinds
  const items = linkItems(
    rt,
    kinds ? rows.filter(([r]) => kinds.includes(rt.docByRow[r]?.s)) : rows,
  )
  // the pinned list survives focus changes, like a typed query; Escape in the box clears it
  rt.query = { text: label, items }
  renderDocumentLinks(
    rt,
    { r: -1, d: -1, s: 'note', t: label, c: 0, p: null, l: '', x: '' },
    items,
    label,
  )
  rt.links.dataset.query = label
  renderRail(rt, items)
}
