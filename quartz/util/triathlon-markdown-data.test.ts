import assert from 'node:assert/strict'
import test from 'node:test'
import { renderGfmTable, renderTitledSections } from './triathlon-markdown-data'

test('renders a dynamic GFM table with alignment and markdown-safe cells', () => {
  const markdown = renderGfmTable(
    [
      {
        name: 'ride | easy',
        note: 'line one\nline two',
        path: 'C:\\data\\ride',
        empty: '',
        missing: null,
      },
    ],
    { alignments: { name: 'left', note: 'center', missing: 'right' } },
  )

  assert.equal(
    markdown,
    '| name | note | path | empty | missing |\n' +
      '| :--- | :---: | --- | --- | ---: |\n' +
      '| "ride \\| easy" | "line one<br>line two" | "C:\\\\data\\\\ride" | "" | null |',
  )
})

test('distinguishes scalar types and neutralizes inline markdown and raw HTML', () => {
  const markdown = renderGfmTable([
    {
      stringNull: 'null',
      nullValue: null,
      quoted: '""',
      literalBreak: 'a<br>b',
      newline: 'a\nb',
      markdown: '*bold* [x](url)',
      html: '<script>run()</script>',
      entity: '&copy;',
    },
  ])

  assert.match(markdown, /\| "null" \| null \| "\\"\\"" \|/)
  assert.match(markdown, /\| "a\\<br\\>b" \| "a<br>b" \|/)
  assert.match(markdown, /"\\\*bold\\\* \\\[x\\\]\\\(url\\\)"/)
  assert.match(markdown, /"\\<script\\>run\\\(\\\)\\<\/script\\>"/)
  assert.match(markdown, /"&amp;copy;"/)
  assert.doesNotMatch(markdown, /<script>/)
})

test('renders flat objects with exact field labels and explicit scalar values', () => {
  assert.equal(
    renderTitledSections(
      { distance: 10, completed: false, note: '', missing: null },
      { title: 'activity', headingDepth: 2 },
    ),
    '## activity\n\n| field | value |\n| --- | --- |\n| distance | 10 |\n| completed | false |\n| note | "" |\n| missing | null |',
  )
})

test('renders flat record arrays as one ordered table with one-based indexes', () => {
  assert.equal(
    renderTitledSections(
      [
        { sport: 'run', distance: 5 },
        { sport: 'bike', distance: 40, note: '' },
      ],
      { title: 'activities' },
    ),
    '## activities\n\n| arrayIndex | sport | distance | note |\n' +
      '| ---: | --- | --- | --- |\n' +
      '| 1 | "run" | 5 | undefined |\n' +
      '| 2 | "bike" | 40 | "" |',
  )
})

test('keeps synthetic array positions distinct from source index fields', () => {
  assert.equal(
    renderTitledSections(
      [
        { arrayIndex: 99, _arrayIndex: 98, index: 97, value: 'first' },
        { arrayIndex: 89, _arrayIndex: 88, index: 87, value: 'second' },
      ],
      { title: 'rows' },
    ),
    '## rows\n\n| \\_\\_arrayIndex | arrayIndex | \\_arrayIndex | index | value |\n' +
      '| ---: | --- | --- | --- | --- |\n' +
      '| 1 | 99 | 98 | 97 | "first" |\n' +
      '| 2 | 89 | 88 | 87 | "second" |',
  )
})

test('renders nested records and arrays as indexed full-path sections', () => {
  const markdown = renderTitledSections(
    {
      analytics: { thresholds: { ftp: 300 } },
      activities: [
        { id: 1, detail: { route: ['start', 'finish'] } },
        { id: 2, detail: { route: [] } },
      ],
    },
    { title: 'payload', headingDepth: 2 },
  )

  assert.equal(
    markdown,
    '## payload\n\n### payload.analytics\n\n#### payload.analytics.thresholds\n\n| field | value |\n| --- | --- |\n| ftp | 300 |\n\n### payload.activities\n\n#### payload.activities\\[0\\]\n\n| field | value |\n| --- | --- |\n| id | 1 |\n\n##### payload.activities\\[0\\].detail\n\n###### payload.activities\\[0\\].detail.route\n\n| arrayIndex | value |\n| ---: | --- |\n| 1 | "start" |\n| 2 | "finish" |\n\n#### payload.activities\\[1\\]\n\n| field | value |\n| --- | --- |\n| id | 2 |\n\n##### payload.activities\\[1\\].detail\n\n###### payload.activities\\[1\\].detail.route\n\n| field | value |\n| --- | --- |\n| empty | [] |',
  )
})

test('neutralizes markup and newlines in section headings', () => {
  const markdown = renderTitledSections({ '<script>\n# forged': { value: 1 } }, { title: 'root' })

  assert.match(markdown, /^### root\.\\<script\\> \\# forged$/m)
  assert.doesNotMatch(markdown, /<script>/)
  assert.doesNotMatch(markdown, /^# forged$/m)
})

test('preserves empty objects, arrays, null roots, and scalar roots', () => {
  assert.equal(
    renderTitledSections({ emptyObject: {}, emptyArray: [] }, { title: 'data' }),
    '## data\n\n### data.emptyObject\n\n| field | value |\n| --- | --- |\n| empty | {} |\n\n### data.emptyArray\n\n| field | value |\n| --- | --- |\n| empty | [] |',
  )
  assert.equal(
    renderTitledSections(null, { title: 'null-value' }),
    '## null-value\n\n| value |\n| --- |\n| null |',
  )
  assert.equal(
    renderTitledSections('', { title: 'empty-value' }),
    '## empty-value\n\n| value |\n| --- |\n| "" |',
  )
})

test('caps heading markers while retaining full nested paths', () => {
  const markdown = renderTitledSections(
    { one: { two: { three: { four: { five: 'leaf' } } } } },
    { title: 'root', headingDepth: 5 },
  )

  assert.match(markdown, /^##### root/m)
  assert.match(markdown, /^###### root\.one/m)
  assert.match(markdown, /^###### root\.one\.two\.three\.four/m)
  assert.match(markdown, /\| five \| "leaf" \|/)
})
