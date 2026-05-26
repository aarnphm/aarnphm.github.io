import assert from 'node:assert/strict'
import test from 'node:test'
import {
  parseWikipediaTarget,
  readWikipediaPreviewResponse,
  wikipediaActionApiUrl,
  wikipediaArticleUrl,
} from './wikipedia'

test('parses Wikipedia article URLs', () => {
  assert.deepEqual(parseWikipediaTarget('https://en.wikipedia.org/wiki/Alan_Turing#Career'), {
    lang: 'en',
    title: 'Alan_Turing',
  })

  assert.deepEqual(parseWikipediaTarget('https://zh-yue.wikipedia.org/wiki/%E9%A6%99%E6%B8%AF'), {
    lang: 'zh-yue',
    title: '香港',
  })
})

test('rejects non-article Wikipedia URLs', () => {
  assert.equal(parseWikipediaTarget('https://www.wikipedia.org/'), undefined)
  assert.equal(
    parseWikipediaTarget('https://en.wikipedia.org/w/index.php?title=Jupiter'),
    undefined,
  )
  assert.equal(parseWikipediaTarget('https://example.com/wiki/Jupiter'), undefined)
})

test('builds the Action API preview URL', () => {
  const url = wikipediaActionApiUrl({ lang: 'en', title: 'Jupiter' })

  assert.equal(url.origin, 'https://en.wikipedia.org')
  assert.equal(url.pathname, '/w/api.php')
  assert.equal(url.searchParams.get('origin'), '*')
  assert.equal(url.searchParams.get('prop'), 'extracts|pageimages|description')
  assert.equal(url.searchParams.get('exsentences'), '3')
  assert.equal(url.searchParams.get('pithumbsize'), '320')
  assert.equal(url.searchParams.get('titles'), 'Jupiter')
})

test('reads Wikipedia preview responses', () => {
  const preview = readWikipediaPreviewResponse(
    {
      query: {
        pages: [
          {
            title: 'Jupiter',
            extract: 'Jupiter is the fifth planet from the Sun.',
            description: 'Fifth planet from the Sun',
            thumbnail: {
              source: 'https://upload.wikimedia.org/jupiter.png',
              width: 320,
              height: 290,
            },
          },
        ],
      },
    },
    { lang: 'en', title: 'Jupiter' },
  )

  assert.deepEqual(preview, {
    title: 'Jupiter',
    extract: 'Jupiter is the fifth planet from the Sun.',
    description: 'Fifth planet from the Sun',
    thumbnail: { source: 'https://upload.wikimedia.org/jupiter.png', width: 320, height: 290 },
    pageUrl: 'https://en.wikipedia.org/wiki/Jupiter',
  })
})

test('builds article URLs from normalized titles', () => {
  assert.equal(
    wikipediaArticleUrl({ lang: 'en', title: 'Surface_(topology)' }),
    'https://en.wikipedia.org/wiki/Surface_(topology)',
  )
  assert.equal(
    wikipediaArticleUrl({ lang: 'en', title: 'Wikipedia:Manual_of_Style/Subpages' }),
    'https://en.wikipedia.org/wiki/Wikipedia%3AManual_of_Style/Subpages',
  )
})
