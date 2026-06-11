import assert from 'node:assert/strict'
import test from 'node:test'
import {
  parseSepTarget,
  readSepPreview,
  sepEntryUrl,
  sepPreviewApiUrl,
  sepTargetFromSearchParams,
} from './sep'
import { readSepPreviewHtml } from './sep-preview'

test('parses SEP entry URLs', () => {
  assert.deepEqual(parseSepTarget('https://plato.stanford.edu/entries/qualia/'), {
    entry: 'qualia',
  })
  assert.deepEqual(parseSepTarget('https://plato.stanford.edu/entries/qualia/#Repres'), {
    entry: 'qualia',
  })
  assert.deepEqual(parseSepTarget('https://plato.stanford.edu/archives/win2023/entries/qualia/'), {
    entry: 'qualia',
    archive: 'win2023',
  })
})

test('rejects non-entry SEP URLs', () => {
  assert.equal(parseSepTarget('https://plato.stanford.edu/index.html'), undefined)
  assert.equal(parseSepTarget('https://plato.stanford.edu/entries/'), undefined)
  assert.equal(
    parseSepTarget('https://plato.stanford.edu/archives/bogus/entries/qualia/'),
    undefined,
  )
  assert.equal(parseSepTarget('https://plato.stanford.edu.evil.test/entries/qualia/'), undefined)
})

test('builds SEP entry and preview URLs', () => {
  assert.equal(sepEntryUrl({ entry: 'qualia' }), 'https://plato.stanford.edu/entries/qualia/')
  assert.equal(
    sepEntryUrl({ entry: 'qualia', archive: 'win2023' }),
    'https://plato.stanford.edu/archives/win2023/entries/qualia/',
  )

  const apiUrl = sepPreviewApiUrl(
    { entry: 'qualia', archive: 'win2023' },
    'https://aarnphm.xyz/thoughts/dualism',
  )
  assert.equal(apiUrl.pathname, '/api/sep')
  assert.equal(apiUrl.searchParams.get('entry'), 'qualia')
  assert.equal(apiUrl.searchParams.get('archive'), 'win2023')
})

test('reads SEP targets from search params', () => {
  assert.deepEqual(sepTargetFromSearchParams(new URLSearchParams({ entry: 'qualia' })), {
    entry: 'qualia',
  })
  assert.deepEqual(
    sepTargetFromSearchParams(new URLSearchParams({ entry: 'qualia', archive: 'win2023' })),
    { entry: 'qualia', archive: 'win2023' },
  )
  assert.equal(sepTargetFromSearchParams(new URLSearchParams({ entry: '../nope' })), null)
  assert.equal(
    sepTargetFromSearchParams(new URLSearchParams({ entry: 'qualia', archive: 'nope' })),
    null,
  )
})

test('reads SEP preview payloads', () => {
  assert.deepEqual(
    readSepPreview({
      entry: 'qualia',
      title: 'Qualia',
      pageUrl: 'https://plato.stanford.edu/entries/qualia/',
      extract: 'Feelings and experiences vary widely.',
      authors: ['Michael Tye', '', 1],
      pubInfo: 'First published Wed Aug 20, 1997; substantive revision Fri Sep 19, 2025',
      toc: [
        { text: 'Uses of the Term', href: '#UseTerQua', depth: 1 },
        { text: 'No href', depth: 2 },
        { text: 'Deep thing', href: '#Deep', depth: 99 },
      ],
    }),
    {
      entry: 'qualia',
      title: 'Qualia',
      pageUrl: 'https://plato.stanford.edu/entries/qualia/',
      extract: 'Feelings and experiences vary widely.',
      authors: ['Michael Tye'],
      pubInfo: 'First published Wed Aug 20, 1997; substantive revision Fri Sep 19, 2025',
      toc: [
        { text: 'Uses of the Term', href: '#UseTerQua', depth: 1 },
        { text: 'Deep thing', href: '#Deep', depth: 6 },
      ],
    },
  )

  assert.equal(
    readSepPreview({ entry: '../qualia', title: 'x', pageUrl: 'y', extract: 'z' }),
    undefined,
  )
})

test('reads SEP entry HTML previews', () => {
  const preview = readSepPreviewHtml(
    [
      '<!DOCTYPE html>',
      '<html><head>',
      '<meta property="citation_title" content="Qualia" />',
      '<meta property="citation_author" content="Tye, Michael" />',
      '<meta name="DCTERMS.issued" content="1997-08-20" />',
      '</head><body>',
      '<div id="aueditable">',
      '<h1>Qualia</h1>',
      '<div id="pubinfo"><em>First published Wed Aug 20, 1997; substantive revision Fri Sep 19, 2025</em></div>',
      '<div id="preamble">',
      '<p>Feelings and experiences vary widely. There is something it is <em>like</em> to undergo each state.</p>',
      '<p>Philosophers often use the term &lsquo;qualia&rsquo; to refer to the phenomenal aspects of our mental lives.</p>',
      '</div>',
      '<div id="toc">',
      '<ul>',
      '<li><a href="#UseTerQua">1. Uses of the Term</a>',
      '<ul><li><a href="#Sub">1.1 A subsection</a></li></ul>',
      '</li>',
      '<li><a href="#FunQua">2. Functionalism and Qualia</a></li>',
      '<li><a href="https://plato.stanford.edu/info.html">Not an anchor</a></li>',
      '</ul>',
      '</div>',
      '</div>',
      '</body></html>',
    ].join(''),
    { entry: 'qualia' },
  )

  assert.deepEqual(preview, {
    entry: 'qualia',
    title: 'Qualia',
    pageUrl: 'https://plato.stanford.edu/entries/qualia/',
    extract:
      'Feelings and experiences vary widely. There is something it is like to undergo each state. Philosophers often use the term ‘qualia’ to refer to the phenomenal aspects of our mental lives.',
    authors: ['Michael Tye'],
    pubInfo: 'First published Wed Aug 20, 1997; substantive revision Fri Sep 19, 2025',
    toc: [
      { text: '1. Uses of the Term', href: '#UseTerQua', depth: 1 },
      { text: '1.1 A subsection', href: '#Sub', depth: 2 },
      { text: '2. Functionalism and Qualia', href: '#FunQua', depth: 1 },
    ],
  })
})
