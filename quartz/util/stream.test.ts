import assert from 'node:assert/strict'
import test from 'node:test'
import { selectStreamFeedGroups, type StreamEntryGroup } from './stream'
import { parseStreamManifest } from './stream-manifest'

const groups: StreamEntryGroup[] = [
  { id: 'day-2026-08-10', entries: [] },
  { id: 'day-2026-08-09', entries: [] },
]

test('root stream materializes only its newest date group', () => {
  const selected = selectStreamFeedGroups(groups, true)
  assert.deepEqual(
    selected.feedGroups.map(group => group.id),
    ['day-2026-08-10'],
  )
  assert.equal(selected.hasLazyGroups, true)
})

test('daily stream routes retain their complete feed', () => {
  const selected = selectStreamFeedGroups(groups, false)
  assert.deepEqual(
    selected.feedGroups.map(group => group.id),
    ['day-2026-08-10', 'day-2026-08-09'],
  )
  assert.equal(selected.hasLazyGroups, false)
})

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
            text: 'entry body',
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
