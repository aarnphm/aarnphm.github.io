import assert from 'node:assert/strict'
import test from 'node:test'
import type { StreamManifestGroup } from '../../util/stream-manifest'
import { buildStreamSearchData, matchStreamEntries } from './stream-search-index'

const manifest: StreamManifestGroup[] = [
  {
    groupId: '2026-07-26',
    timestamp: 1785100000000,
    isoDate: '2026-07-26T12:00:00.000Z',
    groupSize: 2,
    path: '/stream/on/2026/07/26',
    entries: [
      {
        id: 'stream-entry-14',
        title: 'sub-3 triathlete, bib 6243',
        description: 'SuperTri Toronto 2026',
        content: 'distance 39.5 km time 1h10 speed 33.8 km/h',
        metadata: { tags: ['training'] },
        isoDate: '2026-07-26T12:00:00.000Z',
        displayDate: '2026/07/26',
        wordCount: 536,
      },
      {
        id: 'stream-entry-15',
        title: 'recovery',
        description: 'easy day',
        content: 'slept eight hours',
        metadata: { tags: ['life'] },
        isoDate: '2026-07-26T11:00:00.000Z',
        displayDate: '2026/07/26',
        wordCount: 20,
      },
    ],
  },
]

test('stream search matches rendered fragment text and preserves manifest order', async () => {
  const data = await buildStreamSearchData(manifest)

  assert.deepEqual(
    (await matchStreamEntries(data, '39.5 km')).map(result => result.entry.id),
    ['stream-entry-14'],
  )
  assert.deepEqual(
    (await matchStreamEntries(data, 'bib')).map(result => result.entry.id),
    ['stream-entry-14'],
  )
})
