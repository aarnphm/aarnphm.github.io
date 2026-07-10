import assert from 'node:assert'
import test, { describe } from 'node:test'
import { triathlonDataHtml } from './triathlon-data'

const FEED = [
  JSON.stringify({ kind: 'meta', v: 2, today: '2026-07-09', athlete: { ftp: 230 } }),
  JSON.stringify({ kind: 'day', date: '2026-07-08', km: 12.4, load: 55 }),
  JSON.stringify({
    kind: 'activity',
    date: '2026-07-08',
    sport: 'run',
    name: 'Morning <Run> & "tempo"',
  }),
  JSON.stringify({ kind: 'week', weekStart: '2026-07-06', hours: 8.5 }),
].join('\n')

describe('triathlonDataHtml', () => {
  test('renders a full html document with kind counts and raw link', () => {
    const html = triathlonDataHtml(FEED)
    assert.match(html, /^<!doctype html>/)
    assert.match(html, /<title>triathlon\/data<\/title>/)
    assert.match(html, /1 meta · 1 day · 1 activity · 1 week/)
    assert.match(html, /<a href="\/triathlon\/data\.jsonl">raw jsonl<\/a>/)
  })

  test('labels records with kind and date fields', () => {
    const html = triathlonDataHtml(FEED)
    assert.match(html, /<span class="k">day<\/span> 2026-07-08/)
    assert.match(html, /<span class="k">week<\/span> 2026-07-06/)
    assert.match(html, /<span class="k">meta<\/span> 2026-07-09/)
  })

  test('escapes html in labels and pretty-printed bodies', () => {
    const html = triathlonDataHtml(FEED)
    assert.match(html, /Morning &lt;Run&gt; &amp; &quot;tempo&quot;/)
    assert.doesNotMatch(html, /<Run>/)
    assert.match(html, /&quot;ftp&quot;: 230/)
  })

  test('opens only the first record', () => {
    const html = triathlonDataHtml(FEED)
    assert.strictEqual(html.match(/<details open>/g)?.length, 1)
    assert.strictEqual(html.match(/<details>/g)?.length, 3)
  })

  test('falls back to raw text for unparseable lines and skips blanks', () => {
    const html = triathlonDataHtml('not json\n\n{"kind":"day","date":"2026-07-01"}\n')
    assert.match(html, /<span class="k">unparsed<\/span>/)
    assert.match(html, /<pre>not json<\/pre>/)
    assert.match(html, /1 unparsed · 1 day/)
    assert.strictEqual(html.match(/<details/g)?.length, 2)
  })
})
