import assert from 'node:assert/strict'
import test from 'node:test'
import { renderToString } from 'preact-render-to-string'
import { buildHighParkLoops } from '../../../util/high-park-loops'
import { HighParkPanel } from './high-park'

test('empty page names both sports and has no fabricated activity links', () => {
  const html = renderToString(<HighParkPanel loops={buildHighParkLoops([])} />)
  assert.match(html, /cycling/)
  assert.match(html, /running/)
  assert.match(html, /no complete cycling loops matched/)
  assert.match(html, /no full loop on foot yet/)
  assert.doesNotMatch(html, /#tri-activity-/)
})

test('native record and activity row use the same canonical day anchor', () => {
  const loops = buildHighParkLoops([
    {
      id: '123',
      date: '2026-11-07',
      start: '2026-11-07T13:00:00Z',
      sport: 'bike',
      name: 'morning & park',
      route: [],
      analysisRanges: [
        {
          kind: 'segment',
          id: 'a',
          label: 'High Park - Main Loop - Centre Road',
          startElapsedS: 10,
          endElapsedS: 210,
          durationS: 200,
          distanceKm: 1.8261,
        },
      ],
    },
  ])
  const html = renderToString(<HighParkPanel loops={loops} />)
  // Fastest, most laps and first loop in the tally; the ride row expands instead of linking, and
  // its drop-down (with the activity link) renders on the client.
  assert.equal(
    (html.match(/href="\/triathlon\/on\/2026\/11\/07#tri-activity-123"/g) ?? []).length,
    3,
  )
  assert.match(html, /<summary>.*<span class="tri-hp-ride">morning &amp; park<\/span>/s)
  assert.match(html, /<button class="tri-hp-unit"[^>]*>mph<\/button>/)
  assert.match(html, /morning &amp; park/)
  assert.match(html, /3:20/)
  assert.match(html, /<ol>.*1×.*<\/ol>/s)
})
