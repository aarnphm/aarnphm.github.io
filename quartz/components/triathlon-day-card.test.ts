import type { Element } from 'hast'
import { h } from 'hastscript'
import assert from 'node:assert/strict'
import test from 'node:test'
import {
  filterTriathlonTraceElements,
  triathlonDayProps,
  triathlonEmbedAnchor,
  triathlonEmbedAnchorFromSource,
  triathlonEmbedDayHref,
} from './triathlon-day-card'

test('parses triathlon embed sport and activity exclusions', () => {
  assert.deepEqual(
    triathlonEmbedAnchor('["2026-07-26","cycling","filter=19471122670&19476629599"]'),
    { date: '2026-07-26', sport: 'bike', excludedActivityIds: ['19471122670', '19476629599'] },
  )
})

test('parses ampersand-separated triathlon trace settings', () => {
  assert.deepEqual(
    triathlonEmbedAnchor(
      '["2026-08-10","cycling","settings=matched-rides:false&power-balance:true"]',
    ),
    {
      date: '2026-08-10',
      sport: 'bike',
      settings: { 'matched-rides': false, 'power-balance': true },
    },
  )
})

test('rejects malformed triathlon embed options', () => {
  assert.equal(triathlonEmbedAnchor('["2026-02-29","cycling"]'), null)
  assert.equal(triathlonEmbedAnchor('["2026-07-26","filter=19471122670&&19476629599"]'), null)
  assert.equal(triathlonEmbedAnchor('["2026-07-26","filter="]'), null)
  assert.equal(triathlonEmbedAnchor('["2026-07-26","settings=matched rides:false"]'), null)
  assert.equal(triathlonEmbedAnchor('["2026-07-26","settings=matched-rides:0"]'), null)
  assert.equal(
    triathlonEmbedAnchor('["2026-07-26","settings=matched-rides:false&matched-rides:true"]'),
    null,
  )
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

test('recovers trace settings from source when the cached anchor is slugged', () => {
  assert.deepEqual(
    triathlonEmbedAnchorFromSource(
      '["2026-08-10","cycling","settingsmatched-ridesfalsematched-runstrue"]',
      '![[triathlon#2026-08-10#cycling#settings=matched-rides:false&matched-runs:true]]',
    ),
    {
      date: '2026-08-10',
      sport: 'bike',
      settings: { 'matched-rides': false, 'matched-runs': true },
    },
  )
})

test('routes triathlon embeds to their generated day pages', () => {
  assert.equal(
    triathlonEmbedDayHref('../../../..', '2026-07-26'),
    '../../../../triathlon/on/2026/07/26',
  )
  assert.equal(triathlonEmbedDayHref('.', '2026-02-29'), null)
})

test('carries activity exclusions into hydrated day-card props', () => {
  assert.deepEqual(
    triathlonDayProps(
      { excludedActivityIds: ['19471122670', '19476629599'], embedded: true },
      '2026-07-26',
    ),
    {
      'data-triathlon-date': '2026-07-26',
      'data-triathlon-filter': '19471122670&19476629599',
      'data-triathlon-embedded': '1',
    },
  )
})

test('carries trace settings into hydrated day-card props', () => {
  assert.deepEqual(
    triathlonDayProps(
      { settings: { 'matched-rides': false, 'power-balance': true }, embedded: true },
      '2026-08-10',
    ),
    {
      'data-triathlon-date': '2026-08-10',
      'data-triathlon-settings': 'matched-rides:false&power-balance:true',
      'data-triathlon-embedded': '1',
    },
  )
})

test('filters disabled kebab-case trace blocks from server markup', () => {
  const root: Element = h('div', [
    h('section', { 'data-tri-trace': 'matched-rides' }, 'rides'),
    h('div', [h('section', { 'data-tri-trace': 'power-balance' }, 'balance')]),
    h('section', { 'data-tri-trace': 'matched-runs' }, 'runs'),
  ])

  filterTriathlonTraceElements(root, {
    'matched-rides': false,
    'power-balance': false,
    'matched-runs': true,
  })

  assert.deepEqual(
    root.children
      .filter((child): child is Element => child.type === 'element')
      .map(child => child.properties.dataTriTrace ?? child.tagName),
    ['div', 'matched-runs'],
  )
  const nested = root.children[0]
  assert.equal(nested.type, 'element')
  if (nested.type === 'element') assert.deepEqual(nested.children, [])
})
