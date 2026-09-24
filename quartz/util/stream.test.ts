import assert from 'node:assert/strict'
import test from 'node:test'
import { parseStreamManifest } from './stream-manifest'

test('parses newline-delimited stream manifest groups', () => {
  const groups = parseStreamManifest(
    [
      JSON.stringify({
        groupId: 'day-2026-06-09',
        timestamp: 1_749_427_200_000,
        isoDate: '2026-06-09T00:00:00.000Z',
        groupSize: 1,
        path: '/stream/on/2026/06/09',
        entries: [
          {
            id: 'entry-1',
            title: 'entry',
            description: null,
            content: 'entry body',
            metadata: { tags: ['note'] },
            isoDate: '2026-06-09T00:00:00.000Z',
            displayDate: '2026/06/09',
            wordCount: 2,
          },
        ],
      }),
      '',
    ].join('\n'),
  )

  assert.equal(groups.length, 1)
  assert.equal(groups[0].entries[0].id, 'entry-1')
  assert.equal(groups[0].entries[0].wordCount, 2)
})

test('rejects malformed stream manifest entries', () => {
  assert.throws(
    () =>
      parseStreamManifest(
        JSON.stringify({
          groupId: 'day-2026-06-09',
          timestamp: null,
          isoDate: null,
          groupSize: 1,
          path: null,
          entries: [{ id: 'entry-1' }],
        }),
      ),
    /invalid stream manifest group at line 1/,
  )
})
