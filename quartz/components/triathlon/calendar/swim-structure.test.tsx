import type { Element, RootContent } from 'hast'
import { fromHtml } from 'hast-util-from-html'
import assert from 'node:assert/strict'
import test from 'node:test'
import renderToString from 'preact-render-to-string'
import type { TrainingPeaksCalendarWorkout } from '../../../util/trainingpeaks-calendar'
import { createTriathlonFormatter } from '../runtime/formatter'
import { parseSwimDescription } from './swim-structure'
import { TrainingStructure } from './TrainingStructure'

// The day-page replay covers visibility, focus and layout. These cases cover text variants
// absent from that page: wrapped cycles, per-rep rests, nested sets and ambiguous counts.
const workout: TrainingPeaksCalendarWorkout = {
  id: 'swim-intervals',
  source: 'trainingpeaks',
  date: '2026-10-27',
  title: 'Short activations',
  sport: 'swim',
  workoutTypeId: 1,
  description: `400m pull
4 x 50m build in the 50 on 60
40 x 25m as 3x hard, 1x easy on 35
200m very easy
12 x 25m as 1x hard, 1x easy on 30
200m very easy
4 x 100m pull steady stroke out
200m very easy swim down`,
  planned: { distanceMeters: 2400, durationSeconds: 2700, tss: null },
  actual: { distanceMeters: null, durationSeconds: null, tss: null },
  status: 'planned',
  startTime: null,
  startTimePlanned: null,
  order: null,
}

const formatter = createTriathlonFormatter({
  locale: 'en',
  distance: 'metric',
  powerSamples: 'recorded',
})

function buttons(description: string): Element[] {
  const html = renderToString(
    <TrainingStructure workout={{ ...workout, description }} zones={{}} formatter={formatter} />,
  )
  const found: Element[] = []
  const visit = (nodes: RootContent[]): void => {
    for (const node of nodes) {
      if (node.type !== 'element') continue
      if (node.tagName === 'button') found.push(node)
      visit(node.children)
    }
  }
  visit(fromHtml(html, { fragment: true }).children)
  return found
}

test('written swim intervals render hard/easy cycles and retain every send-off', () => {
  const bars = buttons(workout.description)
  const labels = bars.map(bar => String(bar.properties.ariaLabel))
  assert.equal(bars.length, 64)
  for (let cycle = 0; cycle < 10; cycle++) {
    const reps = labels.slice(5 + cycle * 4, 9 + cycle * 4)
    assert.deepEqual(
      reps.map(label => label.includes(' · hard ·')),
      [true, true, true, false],
    )
    assert.match(reps[3], / · easy · /)
    for (const label of reps) assert.match(label, /send-off interval: @ 0:35/)
  }
  for (let rep = 0; rep < 12; rep++) {
    assert.match(labels[46 + rep], rep % 2 === 0 ? / · hard · / : / · easy · /)
    assert.match(labels[46 + rep], /send-off interval: @ 0:30/)
  }
  assert.equal(parseSwimDescription(workout.description)?.meters, 2900)
})

test('the rendered source distinguishes written volume from the provider headline', () => {
  const html = renderToString(
    <TrainingStructure workout={workout} zones={{}} formatter={formatter} />,
  )
  assert.match(html, /2,900 m from workout text/)
  assert.match(html, /2,400 m planned/)
})

test('counted alternation accepts x, multiplication signs and wrapped instructions', () => {
  for (const description of [
    '8 x 25m as 3 hard, 1 easy on 35',
    '8 × 25m as 3× hard, 1× easy on 35',
    '8 x 25m on 35\nas 3x hard, 1x easy',
  ]) {
    const labels = buttons(description).map(bar => String(bar.properties.ariaLabel))
    assert.equal(labels.length, 8)
    assert.equal(labels.filter(label => label.includes(' · hard ·')).length, 6)
    assert.equal(labels.filter(label => label.includes(' · easy ·')).length, 2)
    assert.ok(labels.every(label => label.includes('send-off interval: @ 0:35')))
  }
})

test('existing split reps, timed sets and explicit per-rep rests keep their quantities', () => {
  const alternate = parseSwimDescription('20 x 50m as\n1 HARD (5sec), 1 EASY (20sec), repeat')
  assert.equal(alternate?.meters, 1000)
  const [block] = alternate?.blocks ?? []
  assert.ok(block?.kind === 'set')
  assert.equal(block.repeat, 10)
  assert.deepEqual(
    block.steps.map(step => step.rest?.min),
    [5, 20],
  )
  assert.ok(block.steps.every(step => step.rest?.interval === false))
  assert.equal(parseSwimDescription('2(2 x 50) superman')?.meters, 200)
  assert.equal(parseSwimDescription('4 x 50m as 25m drill/25m swim')?.meters, 200)
  assert.equal(parseSwimDescription('4 x 2min hard/30sec float')?.seconds, 600)
  assert.equal(parseSwimDescription('400m easy\n4 x 1min hard')?.unit, null)
  assert.equal(parseSwimDescription('technique practice'), null)
})

test('an incomplete or unknown cycle does not invent repetitions or lose volume', () => {
  for (const description of ['7 x 25m as 3x hard, 1x easy', '8 x 25m as 3x foo, 1x bar']) {
    const swim = parseSwimDescription(description)
    const [block] = swim?.blocks ?? []
    assert.ok(block?.kind === 'set')
    assert.equal(block.repeat, 1)
    assert.equal(block.steps.length, 1)
    assert.equal(swim?.meters, description.startsWith('7') ? 175 : 200)
  }
})
