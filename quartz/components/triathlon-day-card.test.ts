import type { Element } from 'hast'
import { toHtml } from 'hast-util-to-html'
import { h } from 'hastscript'
import assert from 'node:assert/strict'
import test from 'node:test'
import {
  serializeTriathlonTraceSettings,
  TRIATHLON_TRACE_DISPLAY_SETTINGS,
} from '../util/triathlon-trace-settings'
import {
  filterTriathlonTraceElements,
  resolveTriathlonEmbedDate,
  triathlonDayProps,
  triathlonCardFactory,
  triathlonEmbedAnchor,
  triathlonEmbedAnchorFromBlockRef,
  triathlonEmbedAnchorFromSource,
  triathlonEmbedDayHref,
} from './triathlon-day-card'

const SIMPLIFIED_SETTINGS_VALUE = serializeTriathlonTraceSettings(
  TRIATHLON_TRACE_DISPLAY_SETTINGS.simplified,
)

test('renders inline math in server-side triathlon card nodes', () => {
  const rendered = toHtml(
    triathlonCardFactory.math('tri-elev-range', '61.0% $\\mathrm{SmO}_2$ avg'),
  )
  assert.match(rendered, /class="tri-math"/)
  assert.match(rendered, /class="katex"/)
  assert.doesNotMatch(rendered, /\$\\mathrm\{SmO\}_2\$/)
})

test('parses triathlon embed sport and activity exclusions', () => {
  assert.deepEqual(
    triathlonEmbedAnchor('["2026-07-26","cycling","filter=19471122670&19476629599"]'),
    { date: '2026-07-26', sport: 'bike', excludedActivityIds: ['19471122670', '19476629599'] },
  )
})

test('parses a direct triathlon activity ID block reference', () => {
  assert.deepEqual(triathlonEmbedAnchor('["19731411847"]'), { activityId: '19731411847' })
  assert.deepEqual(triathlonEmbedAnchorFromBlockRef('#19731411847'), { activityId: '19731411847' })
  assert.deepEqual(triathlonEmbedAnchorFromBlockRef('#2026-08-13'), { date: '2026-08-13' })
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

test('parses dated analytics embeds with settings and activity filters', () => {
  assert.deepEqual(triathlonEmbedAnchor('["2026-08-16","analytics"]'), {
    date: '2026-08-16',
    analytics: true,
  })
  assert.deepEqual(
    triathlonEmbedAnchorFromSource(
      '["2026-08-16","analytics"]',
      '![[triathlon#2026-08-16#analytics]]',
    ),
    { date: '2026-08-16', analytics: true },
  )
  assert.deepEqual(
    triathlonEmbedAnchor('["2026-08-16","analytics","settings=display:simplified"]'),
    { date: '2026-08-16', settings: TRIATHLON_TRACE_DISPLAY_SETTINGS.simplified, analytics: true },
  )
  assert.deepEqual(triathlonEmbedAnchor('["2026-08-16","analytics","settings=display:detailed"]'), {
    date: '2026-08-16',
    settings: TRIATHLON_TRACE_DISPLAY_SETTINGS.detailed,
    analytics: true,
  })
  assert.deepEqual(
    triathlonEmbedAnchor('["2026-08-18","analytics","settings=display:simplified&expanded:true"]'),
    {
      date: '2026-08-18',
      settings: { ...TRIATHLON_TRACE_DISPLAY_SETTINGS.simplified, expanded: true },
      analytics: true,
    },
  )
  assert.deepEqual(triathlonEmbedAnchor('["2026-08-16","analytics","cycling"]'), {
    date: '2026-08-16',
    sport: 'bike',
    analytics: true,
  })
  assert.deepEqual(triathlonEmbedAnchor('["2026-08-16","analytics","filter=19771722076"]'), {
    date: '2026-08-16',
    excludedActivityIds: ['19771722076'],
    analytics: true,
  })
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
  assert.equal(triathlonEmbedAnchor('["2026-07-26","analytics","analytics"]'), null)
  assert.equal(triathlonEmbedAnchor('["2026-07-26","analytics","cycling","bike"]'), null)
  assert.equal(triathlonEmbedAnchor('["19731411847","cycling"]'), null)
})

test('resolves a direct activity embed to the activity date', () => {
  const anchor = triathlonEmbedAnchor('["19731411847"]')
  assert.ok(anchor)
  assert.equal(
    resolveTriathlonEmbedDate(anchor, { details: { 19731411847: { date: '2026-08-13' } } }),
    '2026-08-13',
  )
  assert.equal(resolveTriathlonEmbedDate(anchor, { details: {} }), null)
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
  assert.equal(
    triathlonEmbedDayHref('../../../..', '2026-08-13', '19731411847'),
    '../../../../triathlon/on/2026/08/13#tri-activity-19731411847',
  )
  assert.equal(triathlonEmbedDayHref('.', '2026-02-29'), null)
})

test('carries a direct activity selection into hydrated day-card props', () => {
  assert.deepEqual(triathlonDayProps({ activityId: '19731411847', embedded: true }, '2026-08-13'), {
    'data-triathlon-date': '2026-08-13',
    'data-triathlon-activity-id': '19731411847',
    'data-triathlon-embedded': '1',
  })
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

test('carries analytics mode, cycling, and settings into hydrated day-card props', () => {
  assert.deepEqual(
    triathlonDayProps(
      {
        sport: 'bike',
        settings: { ...TRIATHLON_TRACE_DISPLAY_SETTINGS.simplified, expanded: true },
        analytics: true,
        embedded: true,
      },
      '2026-08-16',
    ),
    {
      'data-triathlon-date': '2026-08-16',
      'data-triathlon-sport': 'bike',
      'data-triathlon-settings': `${SIMPLIFIED_SETTINGS_VALUE}&expanded:true`,
      'data-triathlon-analytics': '1',
      'data-triathlon-embedded': '1',
    },
  )
  assert.deepEqual(
    triathlonDayProps(
      { settings: TRIATHLON_TRACE_DISPLAY_SETTINGS.detailed, embedded: true },
      '2026-08-16',
    ),
    { 'data-triathlon-date': '2026-08-16', 'data-triathlon-embedded': '1' },
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
