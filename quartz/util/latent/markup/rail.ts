import type { Element, ElementContent } from 'hast'
import { h } from 'hastscript'
import {
  LABEL_ROW,
  LABEL_X,
  LANE_X,
  LEADER_X,
  OPEN_WIDTH,
  RAIL_HEIGHT,
  RAIL_WIDTH,
  TRACK_X,
  UNDATED_FIRST,
  UNDATED_ROW,
  layoutRail,
  placeLabels,
  thinYears,
  truncate,
} from '../rail'
import { titleText } from '../title'
import { KIND_ORDER, type LatentKind, type LatentTrackDoc } from '../types'

export interface RailModel {
  docs: LatentTrackDoc[]
  titles: (d: number) => string
  open: number
  focus: number
  /** neighbour doc → best score */
  neighbours: Map<number, number>
  /** neighbour doc → how many of the focus row's neighbours it holds; ribbon width */
  weights?: Map<number, number>
  focusChunkIndex: number
  scopeFlags?: Uint8Array
  /** viewBox height in rail units; RAIL_HEIGHT unless the client has measured its pane */
  height?: number
}

/** The rail as an inline SVG. Collapsed documents are one path per store. */
export const RAIL_BIN = 4
const BAND_WIDTH = 7

export function railSvg(model: RailModel): Element {
  const height = model.height ?? RAIL_HEIGHT
  const layout = layoutRail(model.docs, height)
  const children: ElementContent[] = []
  children.push(h('line.lat-axis', { x1: TRACK_X, x2: TRACK_X, y1: 10, y2: height - 8 }))
  for (const tick of thinYears(layout.years)) {
    children.push(
      h('text.lat-year', { x: TRACK_X - 5, y: tick.y.toFixed(1) }, String(tick.year)),
      h('line.lat-tick', {
        x1: TRACK_X - 3,
        x2: TRACK_X + 3,
        y1: tick.y.toFixed(1),
        y2: tick.y.toFixed(1),
      }),
    )
  }
  // dated documents: y is rank, so time density is already in the year ticks' spacing; the band
  // shows which kinds a slice of time holds, one thin stacked segment per 4-unit bin
  const bins = Math.ceil(height / RAIL_BIN)
  const counts = new Map<LatentKind, Uint16Array>()
  const undated = new Map<LatentKind, number>()
  const totals = new Uint16Array(bins)
  for (let i = 0; i < model.docs.length; i++) {
    const doc = model.docs[i]
    if (doc.day === 0) {
      undated.set(doc.kind, (undated.get(doc.kind) ?? 0) + 1)
      continue
    }
    let row = counts.get(doc.kind)
    if (!row) {
      row = new Uint16Array(bins)
      counts.set(doc.kind, row)
    }
    const b = Math.min(bins - 1, Math.floor(layout.y[i] / RAIL_BIN))
    row[b]++
    totals[b]++
  }
  const segments = new Map<LatentKind, string[]>()
  for (let b = 0; b < bins; b++) {
    if (!totals[b]) continue
    let x = TRACK_X + 3
    for (const kind of KIND_ORDER) {
      const count = counts.get(kind)?.[b] ?? 0
      if (!count) continue
      const w = (BAND_WIDTH * count) / totals[b]
      const list = segments.get(kind) ?? []
      list.push(`M${x.toFixed(1)} ${b * RAIL_BIN}h${w.toFixed(1)}v${RAIL_BIN}h-${w.toFixed(1)}z`)
      segments.set(kind, list)
      x += w
    }
  }
  for (const [kind, list] of segments) {
    children.push(h('path.lat-density', { dataKind: kind, d: list.join('') }))
  }
  // undated documents: one row per kind, a bar as long as its share and the count
  if (undated.size > 0) {
    const total = [...undated.values()].reduce((a, b) => a + b, 0)
    children.push(
      h('line.lat-tick', {
        x1: TRACK_X - 3,
        x2: TRACK_X + 3,
        y1: layout.undatedTop.toFixed(1),
        y2: layout.undatedTop.toFixed(1),
      }),
      h(
        'text.lat-year.lat-undated',
        { x: TRACK_X + 6, y: layout.undatedTop.toFixed(1) },
        `undated · ${total.toLocaleString('en-US')}`,
      ),
    )
    for (const kind of layout.undatedKinds) {
      const count = undated.get(kind) ?? 0
      const y = layout.undatedTop + UNDATED_FIRST + layout.undatedKinds.indexOf(kind) * UNDATED_ROW
      const w = Math.max(2, (count / total) * 36)
      children.push(
        h('rect.lat-density', {
          dataKind: kind,
          x: TRACK_X + 3,
          y: (y - 2.5).toFixed(1),
          width: w.toFixed(1),
          height: 5,
        }),
        h(
          'text.lat-undated-count',
          { x: (TRACK_X + 3 + w + 4).toFixed(1), y: y.toFixed(1) },
          `${kind} ${count.toLocaleString('en-US')}`,
        ),
      )
    }
  }
  // the open document: a bracket at its slot, its chunk rows fanned to the right. The fan and the
  // neighbour labels stay above the undated band, even when the open document sits in it.
  const floor = undated.size > 0 ? layout.undatedTop - 16 : height - 12
  // undated documents share band rows with the kind counts, so brackets and leaders for them
  // anchor just above the band header instead of crossing the counts
  const anchorY = (i: number) => (model.docs[i].day === 0 ? layout.undatedTop - 8 : layout.y[i])
  const open = model.docs[model.open]
  if (open) {
    const oy = anchorY(model.open)
    const rows = Math.min(open.n, 48)
    const rowH = 3
    const first = Math.max(0, Math.min(open.n - rows, model.focusChunkIndex - Math.floor(rows / 2)))
    // neighbour labels share the fan's column: those above the open slot go above the fan, the
    // rest below it, and the fan leaves each group its rows
    const targets = [...model.neighbours.entries()]
      .filter(([d]) => d !== model.open)
      .sort((a, b) => anchorY(a[0]) - anchorY(b[0]))
    const aboveCount = targets.filter(([d]) => anchorY(d) < oy).length
    const belowCount = targets.length - aboveCount
    const top = Math.max(
      14 + (aboveCount > 0 ? 22 + (aboveCount - 1) * LABEL_ROW : 0),
      Math.min(
        floor - 6 - rows * rowH - (belowCount > 0 ? belowCount * LABEL_ROW : 0),
        oy - (model.focusChunkIndex - first) * rowH,
      ),
    )
    const chunkMarks: string[] = []
    for (let i = 0; i < rows; i++) {
      const y = (top + i * rowH).toFixed(1)
      chunkMarks.push(`M${LEADER_X} ${y}h${Math.min(OPEN_WIDTH, 6 + ((i * 37) % 23))}`)
    }
    const focusY = top + (model.focusChunkIndex - first) * rowH
    children.push(
      h('path.lat-bracket', {
        d: `M${LANE_X[open.kind]} ${oy.toFixed(1)}H${LEADER_X - 6}V${top.toFixed(1)}M${LEADER_X - 6} ${oy.toFixed(1)}V${(top + rows * rowH).toFixed(1)}`,
      }),
      h('path.lat-open', { dataKind: open.kind, d: chunkMarks.join('') }),
      h('line.lat-cursor', {
        x1: LEADER_X - 10,
        x2: RAIL_WIDTH,
        y1: focusY.toFixed(1),
        y2: focusY.toFixed(1),
      }),
      h(
        'text.lat-open-title',
        { x: LEADER_X, y: (top - 4).toFixed(1) },
        truncate(titleText(model.titles(model.open)), 26),
      ),
    )
    // ribbons to neighbour documents: width is how many of the focus row's neighbours the
    // document holds, opacity its best score against the others; labels de-overlapped
    const fanTop = top - 22
    const fanBottom = top + rows * rowH + 6
    const sources = targets.map(([d]) => anchorY(d))
    const labelY = [
      ...placeLabels(sources.slice(0, aboveCount), LABEL_ROW, 12, fanTop),
      ...placeLabels(sources.slice(aboveCount), LABEL_ROW, fanBottom, floor),
    ]
    let lo = 255
    let hi = 0
    for (const [, score] of targets) {
      lo = Math.min(lo, score)
      hi = Math.max(hi, score)
    }
    targets.forEach(([d, score], i) => {
      const doc = model.docs[d]
      const ty = anchorY(d)
      const ly = labelY[i]
      const out = model.scopeFlags && model.scopeFlags[d] === 0
      const count = model.weights?.get(d) ?? 1
      const width = Math.min(5, 0.6 + 0.9 * (count - 1))
      const opacity = 0.35 + 0.55 * (hi > lo ? (score - lo) / (hi - lo) : 1)
      const title = titleText(model.titles(d))
      children.push(
        h('path.lat-leader', {
          dataKind: doc.kind,
          dataOut: out ? '' : undefined,
          strokeWidth: width.toFixed(1),
          opacity: (out ? opacity * 0.5 : opacity).toFixed(2),
          d: `M${LANE_X[doc.kind] + 4} ${ty.toFixed(1)}C${LEADER_X - 20} ${ty.toFixed(1)} ${LEADER_X - 20} ${ly.toFixed(1)} ${LABEL_X - 4} ${ly.toFixed(1)}`,
        }),
        h(
          'text.lat-label',
          {
            x: LABEL_X,
            y: (ly + 3.5).toFixed(1),
            dataDoc: String(d),
            dataKind: doc.kind,
            dataOut: out ? '' : undefined,
          },
          [
            h(
              'title',
              `${title} · ${(score / 255).toFixed(2)}${count > 1 ? ` · ${count} passages` : ''}`,
            ),
            truncate(title, count > 1 ? 21 : 25),
            ...(count > 1 ? [h('tspan.lat-label-count', { dx: 3 }, `×${count}`)] : []),
          ],
        ),
      )
    })
  }
  return h(
    'svg.lat-track',
    {
      viewBox: `0 0 ${RAIL_WIDTH} ${height}`,
      preserveAspectRatio: 'none',
      style: `--lat-rail-height: ${height}`,
      role: 'img',
      ariaLabel: 'documents in time order',
    },
    children,
  )
}
