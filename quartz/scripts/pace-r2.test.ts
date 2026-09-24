import assert from 'node:assert/strict'
import { test } from 'node:test'
import {
  LEGACY_VAL_SPACE,
  type LiveLatest,
  livePointerVal,
  parseLiveLatest,
  regressesLiveMae,
} from './pace-r2'

test('regressesLiveMae allows a same-space result within tolerance', () => {
  const live: LiveLatest = { version: 1, datasetHash: '', valMae: 0.05, valSpace: 'ratio' }
  assert.equal(regressesLiveMae(live, 0.05, 'ratio', 0.15), false)
})

test('legacy hr pointer still gates a ratio-space regression', () => {
  const live = parseLiveLatest({ version: 2, val: { mae: 0.05 } })
  assert.ok(live)
  assert.equal(regressesLiveMae(live, 0.09, 'ratio', 0.15), true)
})

test('legacy pace pointer skips the gate for the ratio to velocity migration', () => {
  const live = parseLiveLatest({ version: 2, val: { mae: 0.05 } })
  assert.ok(live)
  assert.equal(regressesLiveMae(live, 0.24, 'velocity', 0.15), false)
})

test('livePointerVal writes the passed mae, not the source mae', () => {
  assert.equal(livePointerVal({ mae: 0.05, valSpace: 'ratio' }, 0.24).mae, 0.24)
})

test('livePointerVal defaults a legacy source manifest to ratio', () => {
  assert.equal(
    livePointerVal({ mae: 0.05, nll: 1, coverage90: 0.9 }, 0.05).valSpace,
    LEGACY_VAL_SPACE,
  )
})

test('a reverted velocity pointer round-trips and still gates same-space', () => {
  const pointerVal = livePointerVal(
    { mae: 0.2, nll: 1, coverage90: 0.9, valSpace: 'velocity' },
    0.2,
  )
  assert.equal(pointerVal.valSpace, 'velocity')
  const live = parseLiveLatest({ version: 5, val: pointerVal })
  assert.ok(live)
  assert.equal(regressesLiveMae(live, 0.3, 'velocity', 0.15), true)
})
