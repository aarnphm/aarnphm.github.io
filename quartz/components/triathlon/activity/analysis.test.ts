import assert from 'node:assert/strict'
import test from 'node:test'
import { analysisChartSelectionBounds } from '../../../util/triathlon-card'
import { activityScrubCursorX, activityScrubElapsedIndexAt, activityScrubIndexAt } from './analysis'

const routeLessSamples = [
  { d: 0, elapsedS: 0 },
  { d: 520, elapsedS: 520 },
  { d: 1_040, elapsedS: 1_040 },
  { d: 1_560, elapsedS: 1_560 },
]

test('route-less activity scrub selects the nearest elapsed-axis sample', () => {
  assert.equal(activityScrubIndexAt(routeLessSamples, -20), 0)
  assert.equal(activityScrubIndexAt(routeLessSamples, 510), 1)
  assert.equal(activityScrubIndexAt(routeLessSamples, 780), 1)
  assert.equal(activityScrubIndexAt(routeLessSamples, 800), 2)
  assert.equal(activityScrubIndexAt(routeLessSamples, 2_000), 3)
})

test('linked activity charts synchronize samples by elapsed time', () => {
  const distanceSamples = [
    { d: 0, elapsedS: 90 },
    { d: 4.8, elapsedS: 600 },
    { d: 10.2, elapsedS: 1_100 },
    { d: 15.4, elapsedS: 1_700 },
  ]
  assert.equal(activityScrubElapsedIndexAt(distanceSamples, 0), 0)
  assert.equal(activityScrubElapsedIndexAt(distanceSamples, 850), 1)
  assert.equal(activityScrubElapsedIndexAt(distanceSamples, 900), 2)
  assert.equal(activityScrubElapsedIndexAt(distanceSamples, 2_000), 3)
  assert.equal(activityScrubElapsedIndexAt([], 900), -1)
})

test('linked run/walk cursor projects the hovered time onto its own elapsed axis', () => {
  const sample = { d: 1.5, elapsedS: 900 }
  assert.equal(activityScrubCursorX(sample, { startDistanceKm: 0, endDistanceKm: 2 }), 75)
  assert.equal(activityScrubCursorX(sample, { startElapsedS: 0, endElapsedS: 1_800 }), 50)
  assert.equal(activityScrubCursorX(sample, { startElapsedS: 600, endElapsedS: 1_800 }), 25)
  assert.equal(activityScrubCursorX(sample, { startElapsedS: 1_000, endElapsedS: 1_800 }), 0)
  assert.equal(activityScrubCursorX(sample, { startElapsedS: 0, endElapsedS: 600 }), 100)
  assert.equal(activityScrubCursorX(sample, { startElapsedS: 900, endElapsedS: 900 }), 0)
})

const selectedLap = {
  startDistanceKm: 0.5,
  endDistanceKm: 1.2,
  startElapsedS: 900,
  endElapsedS: 1_500,
}

test('projects a selected lap independently onto distance and elapsed chart axes', () => {
  assert.deepEqual(
    analysisChartSelectionBounds(selectedLap, { startDistanceKm: 0, endDistanceKm: 2 }),
    { x: 25, width: 35 },
  )
  assert.deepEqual(
    analysisChartSelectionBounds(selectedLap, { startElapsedS: 0, endElapsedS: 3_000 }),
    { x: 30, width: 20 },
  )
})

test('projects lap highlights into zoomed SVG coordinates and inset environment plots', () => {
  assert.deepEqual(
    analysisChartSelectionBounds(
      selectedLap,
      { startDistanceKm: 0.4, endDistanceKm: 1.4 },
      { x: 20, width: 50 },
    ),
    { x: 25, width: 35 },
  )
  const bounds = analysisChartSelectionBounds(
    selectedLap,
    { startElapsedS: 0, endElapsedS: 3_000 },
    { x: 2, width: 96 },
  )
  assert.ok(Math.abs(bounds.x - 30.8) < 1e-9)
  assert.ok(Math.abs(bounds.width - 19.2) < 1e-9)
})

test('clips lap highlights to each chart domain and handles empty domains', () => {
  assert.deepEqual(
    analysisChartSelectionBounds(selectedLap, { startDistanceKm: 0.8, endDistanceKm: 1 }),
    { x: 0, width: 100 },
  )
  assert.deepEqual(
    analysisChartSelectionBounds(selectedLap, { startElapsedS: 0, endElapsedS: 600 }),
    { x: 100, width: 0 },
  )
  assert.deepEqual(
    analysisChartSelectionBounds(selectedLap, { startElapsedS: 1_800, endElapsedS: 3_000 }),
    { x: 0, width: 0 },
  )
  assert.deepEqual(
    analysisChartSelectionBounds(selectedLap, { startElapsedS: 900, endElapsedS: 900 }),
    { x: 0, width: 0 },
  )
})
