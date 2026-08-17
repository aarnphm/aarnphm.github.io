import assert from 'node:assert/strict'
import test from 'node:test'
import type { PowerRankEffort, PowerRankInterval } from '../../../../plugins/stores/power-rank'
import {
  powerRankEffortLabel,
  powerRankProgressLabel,
  powerRankProgressNextLabel,
  powerRankRangeRows,
  powerSkillAtSeconds,
} from './power'

const effort: PowerRankEffort = {
  watts: 280,
  wattsPerKg: 3.25,
  level: 2,
  levelName: 'intermediate',
  percentile: 34,
  nextLevel: 3,
  nextLevelName: 'athletic',
  nextWatts: 312,
  wattsToNext: 32,
}

test('keeps the radar effort readout concise', () => {
  assert.equal(powerRankEffortLabel(effort), '280 W · 3.25 W/kg')
  assert.equal(powerRankEffortLabel(null), 'no data')
})

test('moves rank and next-level context into the progress row', () => {
  assert.equal(powerRankProgressLabel(effort), 'Intermediate · 34%')
  assert.equal(powerRankProgressNextLabel(effort), '32 W to Athletic')
  assert.equal(powerRankProgressLabel(null), 'no data')
  assert.equal(powerRankProgressNextLabel(null), '')
  assert.equal(powerRankProgressNextLabel({ ...effort, wattsToNext: null }), 'top level')
})

test('maps power-curve durations to Strava skill bands', () => {
  assert.equal(powerSkillAtSeconds(15), 'sprint')
  assert.equal(powerSkillAtSeconds(60), 'sprint')
  assert.equal(powerSkillAtSeconds(61), 'attack')
  assert.equal(powerSkillAtSeconds(600), 'attack')
  assert.equal(powerSkillAtSeconds(601), 'climb')
})

test('builds the complete descending level table for a hovered duration', () => {
  const interval: PowerRankInterval = {
    durationS: 300,
    skill: 'attack',
    thresholds: [
      { level: 1, name: 'aspiring', percentile: 1, watts: 178, wattsPerKg: 2.06 },
      { level: 2, name: 'intermediate', percentile: 31, watts: 267, wattsPerKg: 3.09 },
      { level: 3, name: 'athletic', percentile: 46, watts: 305, wattsPerKg: 3.53 },
    ],
    efforts: { 'six-weeks': effort, year: null },
  }

  assert.deepEqual(powerRankRangeRows(interval, 'six-weeks'), [
    {
      level: 3,
      label: 'Athletic',
      percentile: 46,
      thresholdWatts: 305,
      bestWatts: null,
      gapWatts: 25,
      current: false,
    },
    {
      level: 2,
      label: 'Intermediate',
      percentile: 31,
      thresholdWatts: 267,
      bestWatts: 280,
      gapWatts: null,
      current: true,
    },
    {
      level: 1,
      label: 'Aspiring',
      percentile: 1,
      thresholdWatts: 178,
      bestWatts: null,
      gapWatts: null,
      current: false,
    },
  ])
})
