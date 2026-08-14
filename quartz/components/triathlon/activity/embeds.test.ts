import assert from 'node:assert/strict'
import test from 'node:test'
import { dayExtrasFromDataset } from './embeds'

test('restores activity exclusions and trace settings from the embed dataset', () => {
  const dataset: DOMStringMap = {
    triathlonFilter: '19471122670&19476629599',
    triathlonSettings: 'matched-rides:false&power-balance:true',
    triathlonEmbedded: '1',
  }

  assert.deepEqual(dayExtrasFromDataset(dataset), {
    location: undefined,
    event: undefined,
    sport: undefined,
    excludedActivityIds: ['19471122670', '19476629599'],
    settings: { 'matched-rides': false, 'power-balance': true },
    expanded: false,
    embedded: true,
    dateHref: undefined,
  })
})
