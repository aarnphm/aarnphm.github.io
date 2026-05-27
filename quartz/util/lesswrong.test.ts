import assert from 'node:assert/strict'
import test from 'node:test'
import {
  greaterWrongPostUrl,
  lessWrongPostUrl,
  lessWrongPreviewApiUrl,
  lessWrongTargetFromSearchParams,
  parseLessWrongTarget,
  readLessWrongPreview,
} from './lesswrong'
import { readGreaterWrongPreviewHtml } from './lesswrong-preview'

test('parses LessWrong post URLs', () => {
  assert.deepEqual(
    parseLessWrongTarget(
      'https://www.lesswrong.com/posts/zumnfc7jctgocfoe9/death-note-anonymity-and-information-theory',
    ),
    { postId: 'zumnfc7jctgocfoe9', slug: 'death-note-anonymity-and-information-theory' },
  )

  assert.deepEqual(
    parseLessWrongTarget(
      'https://www.lesswrong.com/s/h8DebDmuode4TMcRj/p/LJiGhpq8w4Badr5KJ/graphql-tutorial-for-lesswrong-and-effective-altruism-forum',
    ),
    {
      postId: 'LJiGhpq8w4Badr5KJ',
      slug: 'graphql-tutorial-for-lesswrong-and-effective-altruism-forum',
    },
  )
})

test('rejects non-post LessWrong URLs', () => {
  assert.equal(parseLessWrongTarget('https://www.lesswrong.com/users/gwern'), undefined)
  assert.equal(parseLessWrongTarget('https://www.lesswrong.com/w/bayes-rule'), undefined)
  assert.equal(
    parseLessWrongTarget('https://www.lesswrong.com.evil.test/posts/abc/example'),
    undefined,
  )
})

test('builds LessWrong preview URLs', () => {
  const target = {
    postId: 'zumnfc7jctgocfoe9',
    slug: 'death-note-anonymity-and-information-theory',
  }
  assert.equal(
    lessWrongPostUrl(target),
    'https://www.lesswrong.com/posts/zumnfc7jctgocfoe9/death-note-anonymity-and-information-theory',
  )
  assert.equal(
    greaterWrongPostUrl(target),
    'https://www.greaterwrong.com/posts/zumnfc7jctgocfoe9/death-note-anonymity-and-information-theory',
  )

  const apiUrl = lessWrongPreviewApiUrl(target, 'https://aarnphm.xyz/thoughts/rationality')
  assert.equal(apiUrl.pathname, '/api/lesswrong')
  assert.equal(apiUrl.searchParams.get('postId'), 'zumnfc7jctgocfoe9')
  assert.equal(apiUrl.searchParams.get('slug'), 'death-note-anonymity-and-information-theory')
})

test('reads LessWrong targets from search params', () => {
  assert.deepEqual(
    lessWrongTargetFromSearchParams(
      new URLSearchParams({
        postId: 'zumnfc7jctgocfoe9',
        slug: 'death-note-anonymity-and-information-theory',
      }),
    ),
    { postId: 'zumnfc7jctgocfoe9', slug: 'death-note-anonymity-and-information-theory' },
  )

  assert.equal(lessWrongTargetFromSearchParams(new URLSearchParams({ postId: '../nope' })), null)
})

test('reads LessWrong preview payloads', () => {
  assert.deepEqual(
    readLessWrongPreview({
      postId: 'zumnfc7jctgocfoe9',
      title: 'Death Note, Anonymity, and Information Theory',
      pageUrl:
        'https://www.lesswrong.com/posts/zumnfc7jctgocfoe9/death-note-anonymity-and-information-theory',
      extract: 'I recently wrote up an idea.',
      author: 'gwern',
      postedAt: '2011-05-08T15:44:00.000Z',
      score: 60,
      commentCount: 49,
      readTimeMinutes: 1,
      tags: ['Privacy / Confidentiality / Secrecy', '', 1],
      toc: [
        { text: 'Background', href: '#Background', depth: 1 },
        { text: 'No href', depth: 2 },
        { text: '', href: '#Empty', depth: 1 },
        { text: 'Deep thing', href: '#Deep', depth: 99 },
      ],
    }),
    {
      postId: 'zumnfc7jctgocfoe9',
      title: 'Death Note, Anonymity, and Information Theory',
      pageUrl:
        'https://www.lesswrong.com/posts/zumnfc7jctgocfoe9/death-note-anonymity-and-information-theory',
      extract: 'I recently wrote up an idea.',
      author: 'gwern',
      postedAt: '2011-05-08T15:44:00.000Z',
      score: 60,
      commentCount: 49,
      readTimeMinutes: 1,
      tags: ['Privacy / Confidentiality / Secrecy'],
      toc: [
        { text: 'Background', href: '#Background', depth: 1 },
        { text: 'Deep thing', href: '#Deep', depth: 6 },
      ],
    },
  )
})

test('reads GreaterWrong post HTML previews', () => {
  const preview = readGreaterWrongPreviewHtml(
    [
      '<!DOCTYPE html>',
      '<html><head>',
      '<meta property="og:title" content="Fallback title">',
      '<meta property="og:description" content="Fallback description">',
      '</head><body>',
      '<main class="post">',
      '<h1 class="post-title">Death Note, Anonymity, and Information Theory</h1>',
      '<div class="post-meta top-post-meta">',
      '<a class="author" href="/users/gwern">gwern</a>',
      '<span class="date" data-js-date="1304869440000">8 May 2011 15:44 UTC</span>',
      '<span class="karma-value" title="46 votes">60<span> points</span></span>',
      '<a class="comment-count" href="#comments">49<span> comments</span></a>',
      '<span class="read-time" title="100 words">1<span> min read</span></span>',
      '<div id="tags">',
      '<a href="/tag/privacy-confidentiality-secrecy">Privacy / Confidentiality / Secrecy</a>',
      '<a href="/tag/logic-and-mathematics">Logic &amp; Mathematics </a>',
      '</div>',
      '</div>',
      '<div class="body-text post-body">',
      '<nav class="contents"><div class="contents-head">Contents</div><ul class="contents-list">',
      '<li class="toc-item-1"><a href="#Background">Background</a></li>',
      '<li class="toc-item-2"><a href="#Prompting_LLMs">Prompting LLMs</a></li>',
      '</ul></nav>',
      '<p>I don&apos;t know if this is a little too afar field.</p>',
      '<h1 id="Background"><span id="section-1">Background</span></h1>',
      '<p>I recently wrote up an idea that has been bouncing around my head.</p>',
      '</div>',
      '</main>',
      '</body></html>',
    ].join(''),
    { postId: 'zumnfc7jctgocfoe9', slug: 'death-note-anonymity-and-information-theory' },
  )

  assert.deepEqual(preview, {
    postId: 'zumnfc7jctgocfoe9',
    title: 'Death Note, Anonymity, and Information Theory',
    pageUrl:
      'https://www.lesswrong.com/posts/zumnfc7jctgocfoe9/death-note-anonymity-and-information-theory',
    extract:
      "I don't know if this is a little too afar field. I recently wrote up an idea that has been bouncing around my head.",
    author: 'gwern',
    postedAt: '2011-05-08T15:44:00.000Z',
    score: 60,
    commentCount: 49,
    readTimeMinutes: 1,
    tags: ['Privacy / Confidentiality / Secrecy', 'Logic & Mathematics'],
    toc: [
      { text: 'Background', href: '#Background', depth: 1 },
      { text: 'Prompting LLMs', href: '#Prompting_LLMs', depth: 2 },
    ],
  })
})
