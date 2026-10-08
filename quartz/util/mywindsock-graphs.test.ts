import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import { parseMyWindsockGraphs } from './mywindsock-graphs'

const archive = async (activityId: number): Promise<unknown> =>
  JSON.parse(
    await readFile(
      new URL(`../../content/triathlon/wind/${activityId}.json`, import.meta.url),
      'utf8',
    ),
  )

// The browser cannot prove that projection preserves the source archive, or that
// a second activity's valid W′ values survive the rejection of a corrupt curve.
test('rejects the captured October 6 W′ overflow without rewriting its source', async () => {
  const input = await archive(20480919077)
  const before = JSON.stringify(input)
  const result = parseMyWindsockGraphs(input, 20480919077)
  assert.ok(result)
  const graph = result.graphs.find(graph => graph.key === 'wprime')
  assert.ok(graph)
  assert.equal(graph.state, 'invalid')
  assert.equal(graph.plots.length, 0)
  assert.ok(graph.note?.includes('W′ balance'))
  assert.equal(JSON.stringify(input), before)
})

test('keeps the captured October 7 W′ values and their source units', async () => {
  const result = parseMyWindsockGraphs(await archive(20491396921), 20491396921)
  assert.ok(result)
  const graph = result.graphs.find(graph => graph.key === 'wprime')
  assert.ok(graph)
  assert.equal(graph.state, 'captured')
  const plot = graph.plots[0]
  assert.ok(plot)
  const balance = plot.series.find(series => series.label === 'Joules')
  assert.ok(balance)
  assert.ok(balance.points.length > 1)
  assert.ok(balance.points.every(point => point.y === 20_000))
  assert.equal(plot.axes[balance.axis]?.label, 'Joules')
})

test('projects inline chart names and provider categories from the captured report', async () => {
  const result = parseMyWindsockGraphs(await archive(20480919077), 20480919077)
  assert.ok(result)
  const ranking = result.graphs.find(graph => graph.key === 'inline:pointsgraph')
  assert.equal(ranking?.label, 'Activity Weather Rankings')
  assert.equal(ranking?.category, 'Summary')
  assert.equal(
    result.graphs.find(graph => graph.key === 'virt_grade')?.category,
    'Feels Like Elevation™',
  )
  assert.equal(result.graphs.find(graph => graph.key === 'grade')?.category, 'Course')
  assert.equal(result.graphs.find(graph => graph.key === 'wprime')?.category, 'Power')
  assert.equal(result.graphs.find(graph => graph.key === 'delta_compare')?.state, 'unavailable')
})
