import assert from 'node:assert/strict'
import test from 'node:test'
import type { TriathlonPresentation } from '../../../util/triathlon-presentation'
import { createTriathlonFormatter } from './formatter'

test('presentation formatting is deterministic across locale and distance systems', () => {
  const metric = createTriathlonFormatter({
    locale: 'en',
    distance: 'metric',
    powerSamples: 'recorded',
  })
  const imperial = createTriathlonFormatter({
    locale: 'fr',
    distance: 'imperial',
    powerSamples: 'exclude-zero',
  })

  assert.equal(metric.distance(10, 'run'), '10.0 km')
  assert.equal(metric.distance(1.5, 'swim'), '1,500 m')
  assert.equal(metric.elevation(100), '100 m')
  assert.equal(metric.temperature(20), '20.0°C')
  assert.equal(metric.weight(70), '70.0 kg')
  assert.equal(metric.pace(300), '5:00 /km')
  assert.equal(metric.shortDate('2026-08-10'), 'Aug 10')

  assert.equal(imperial.distance(10, 'run'), '6.2 mi')
  assert.equal(imperial.elevation(100), '328 ft')
  assert.equal(imperial.temperature(20), '68.0°F')
  assert.equal(imperial.weight(70), '154.3 lb')
  assert.equal(imperial.pace(300), '8:03 /mi')
  assert.match(imperial.shortDate('2026-08-10'), /^10 août$/)
  assert.equal(imperial.text('fitness'), 'condition')
})

test('formatter snapshots its presentation input', () => {
  const presentation: TriathlonPresentation = {
    locale: 'en',
    distance: 'metric',
    powerSamples: 'recorded',
  }
  const formatter = createTriathlonFormatter(presentation)
  assert.notEqual(formatter.presentation, presentation)
  assert.equal(Object.isFrozen(formatter.presentation), true)
  assert.equal(Object.isFrozen(formatter), true)
})
