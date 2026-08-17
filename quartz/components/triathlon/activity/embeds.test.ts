import assert from 'node:assert/strict'
import test from 'node:test'
import { dayExtrasFromDataset } from './embeds'

test('restores activity selection, exclusions, and trace settings from the embed dataset', () => {
  const dataset: DOMStringMap = {
    triathlonActivityId: '19731411847',
    triathlonFilter: '19471122670&19476629599',
    triathlonSettings: 'matched-rides:false&power-balance:true',
    triathlonAnalytics: '1',
    triathlonEmbedded: '1',
  }

  assert.deepEqual(dayExtrasFromDataset(dataset), {
    location: undefined,
    event: undefined,
    sport: undefined,
    activityId: '19731411847',
    excludedActivityIds: ['19471122670', '19476629599'],
    settings: { 'matched-rides': false, 'power-balance': true },
    analytics: true,
    expanded: false,
    embedded: true,
    dateHref: undefined,
  })
})
