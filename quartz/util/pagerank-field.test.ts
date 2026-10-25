import assert from 'node:assert/strict'
import test from 'node:test'
import { createNoteRanker, type NoteRankMetadata } from './note-pagerank'
import { fieldLayout } from './pagerank-field'
import { pageRankModel } from './pagerank-view'

/** A garden-shaped corpus: most notes unlinked, links drawn to a few popular notes. */
function corpus(size: number) {
  let seed = 7
  const random = () => ((seed = (seed * 16807) % 2147483647) - 1) / 2147483646
  const data: Record<string, NoteRankMetadata> = {}
  for (let i = 0; i < size; i++) {
    const links: string[] = []
    if (random() < 0.4)
      for (let k = Math.floor(random() * 6); k > 0; k--)
        links.push(`n${Math.floor(size * random() ** 3)}`)
    data[`n${i}`] = { links, filePath: `content/n${i}.md`, fileName: `n${i}` }
  }
  return pageRankModel(createNoteRanker(data).graph(), slug => slug)
}

const model = corpus(1900)

test('a sparse corpus keeps every linked note within half a cell of its score', () => {
  const sparse = corpus(120)
  const field = fieldLayout(sparse, 1100)
  assert.equal(field.spill, 0)
  for (const node of sparse.nodes)
    if (node.inbound > 0)
      assert.ok(Math.abs(field.x[node.id] - field.scale(node.times)) <= field.cell / 2 + 1e-3)
})

for (const width of [1100, 254]) {
  test(`field at ${width}px places every note once without overlap`, () => {
    const field = fieldLayout(model, width)
    const { x, y, radius, cell } = field
    assert.equal(field.overflow, 0)
    for (let a = 0; a < x.length; a++)
      for (let b = a + 1; b < x.length; b++)
        assert.ok(
          Math.hypot(x[a] - x[b], y[a] - y[b]) >= 2 * radius,
          `${model.nodes[a].slug} overlaps ${model.nodes[b].slug}`,
        )
    for (let id = 0; id < x.length; id++) {
      assert.ok(x[id] >= radius && x[id] <= width - radius, 'inside the canvas')
      assert.ok(y[id] - radius >= 0 && y[id] + radius < field.axisY, 'above the axis')
    }
    assert.ok(cell >= 2 * radius)
  })

  test(`field at ${width}px keeps linked notes on their score and unlinked notes in the block`, () => {
    const field = fieldLayout(model, width)
    const unlinked = model.nodes.filter(node => node.inbound === 0)
    assert.equal(field.blockCount, unlinked.length)
    assert.ok(field.blockCount > 0)
    const column = (px: number) => Math.round((px - field.axisX0) / field.cell)
    const filled = new Map<number, number>()
    for (const node of model.nodes)
      if (node.inbound > 0) {
        const c = column(field.x[node.id])
        filled.set(c, (filled.get(c) ?? 0) + 1)
      }
    for (const node of model.nodes) {
      const [px, py] = [field.x[node.id], field.y[node.id]]
      if (node.inbound === 0) {
        assert.ok(px >= field.blockX0 && px <= field.blockX1 && px < field.axisX0)
        assert.ok(py >= field.blockTop)
      } else {
        // A dot leaves its score column only when every column nearer to it is full.
        const ideal = column(field.scale(node.times))
        const distance = Math.abs(column(px) - ideal)
        for (let c = ideal - distance + 1; c < ideal + distance; c++)
          if (c >= 0 && c <= column(field.axisX1))
            assert.equal(filled.get(c), field.rows, `${node.slug} skipped column ${c}`)
        assert.ok(px >= field.axisX0 - 1e-3 && px <= field.axisX1 + 1e-3)
      }
      assert.equal(field.nearest(px, py, 8), node.id)
    }
    assert.equal(field.nearest(-50, -50, 8), -1)
  })
}

test('field ticks stay in range and never collide', () => {
  for (const width of [1100, 600, 324, 254]) {
    const field = fieldLayout(model, width)
    assert.ok(field.ticks.some(tick => tick.label === '1×'))
    for (const [index, tick] of field.ticks.entries()) {
      assert.ok(tick.x >= field.axisX0 - 1e-3 && tick.x <= field.axisX1 + 1e-3, tick.label)
      const previous = field.ticks[index - 1]
      if (previous)
        assert.ok(tick.x - previous.x >= ((tick.label.length + previous.label.length) / 2) * 6.6)
    }
  }
})
