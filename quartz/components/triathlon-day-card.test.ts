import assert from 'node:assert/strict'
import test from 'node:test'
import {
  triathlonDayProps,
  triathlonEmbedAnchor,
  triathlonEmbedAnchorFromSource,
} from './triathlon-day-card'

test('parses triathlon embed sport and activity exclusions', () => {
  assert.deepEqual(
    triathlonEmbedAnchor('["2026-07-26","cycling","filter=19471122670&19476629599"]'),
    { date: '2026-07-26', sport: 'bike', excludedActivityIds: ['19471122670', '19476629599'] },
  )
})

test('rejects malformed triathlon embed filters', () => {
  assert.equal(triathlonEmbedAnchor('["2026-07-26","filter=19471122670&&19476629599"]'), null)
  assert.equal(triathlonEmbedAnchor('["2026-07-26","filter="]'), null)
  assert.equal(triathlonEmbedAnchor('["2026-07-26","unknown"]'), null)
})

test('recovers activity exclusions from source when the cached anchor is slugged', () => {
  assert.deepEqual(
    triathlonEmbedAnchorFromSource(
      '["2026-07-26","filter1947112267019476629599"]',
      [
        '![[triathlon#2026-07-20#cycling]]',
        '![[triathlon#2026-07-26#filter=19471122670&19476629599]]',
      ].join('\n'),
    ),
    { date: '2026-07-26', excludedActivityIds: ['19471122670', '19476629599'] },
  )
})

test('carries activity exclusions into hydrated day-card props', () => {
  assert.deepEqual(
    triathlonDayProps({ excludedActivityIds: ['19471122670', '19476629599'] }, '2026-07-26'),
    { 'data-triathlon-date': '2026-07-26', 'data-triathlon-filter': '19471122670&19476629599' },
  )
})
