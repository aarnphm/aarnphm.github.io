import assert from 'node:assert/strict'
import { once } from 'node:events'
import { createServer } from 'node:http'
import test from 'node:test'
import { extractTwitterPost, parseTwitterPost, renderTwitterPost } from './twitter-content'

const source = 'https://x.com/garden_fixture/status/101'

test('Defuddle extracts post text, self replies, images, and quoted posts from rendered X HTML', async () => {
  const html = `<!doctype html><html><head></head><body><main>
    <div data-testid="cellInnerDiv"><article data-testid="tweet">
      <div data-testid="User-Name"><a href="/garden_fixture">Garden Fixture</a><a href="/garden_fixture">@garden_fixture</a></div>
      <a href="${source}"><time datetime="2026-09-16T00:00:00Z">Sep 16</time></a>
      <div data-testid="tweetText">The main post keeps x &lt; y and its paragraphs.\nA second line.</div>
      <div data-testid="tweetPhoto"><img src="https://pbs.twimg.com/media/main.png" alt="Main photo"></div>
      <div aria-labelledby="id__quote">
        <div data-testid="User-Name"><a href="/quoted_fixture">Quoted Fixture</a><a href="/quoted_fixture">@quoted_fixture</a></div>
        <div data-testid="tweetText">A quoted observation belongs to its own author.</div>
        <div data-testid="tweetPhoto"><img src="https://pbs.twimg.com/media/quote.png" alt="Quoted photo"></div>
      </div>
    </article></div>
    <div data-testid="cellInnerDiv"><article data-testid="tweet">
      <div data-testid="User-Name"><a href="/garden_fixture">Garden Fixture</a><a href="/garden_fixture">@garden_fixture</a></div>
      <div data-testid="tweetText">The author's thread continues here.</div>
    </article></div>
    <div data-testid="cellInnerDiv"><article data-testid="tweet">
      <div data-testid="User-Name"><a href="/stranger">Stranger</a><a href="/stranger">@stranger</a></div>
      <div data-testid="tweetText">Unrelated replies should stay out of the saved post.</div>
    </article></div>
  </main></body></html>`
  const article = await parseTwitterPost(html, source, { useAsync: false })
  const rendered = renderTwitterPost(article, source)
  assert.match(rendered, /The main post keeps x &#x3C; y/)
  assert.match(rendered, /The author's thread continues here/)
  assert.match(rendered, /<blockquote>[\s\S]*A quoted observation[\s\S]*<\/blockquote>/)
  assert.equal(rendered.match(/media\/quote.png/g)?.length, 1)
  assert.match(rendered, /Main photo/)
  assert.doesNotMatch(rendered, /Unrelated replies|twitter-tweet|widgets\.js|<iframe/)
})

test('native rendering retains semantic content and removes executable or publisher styling', async () => {
  const article = await parseTwitterPost('<html><body></body></html>', source, { useAsync: false })
  article.author = '@fixture <script>'
  article.title = 'Post by @fixture on X'
  article.published = '2026-09-16'
  article.content = `<p style="position:fixed" onclick="alert(1)">Safe <strong>text</strong>
    <a href="javascript:alert(1)">unsafe link</a><a href="/fixture">profile</a></p>
    <script>alert(1)</script><iframe src="https://evil.example"></iframe>
    <img src="https://pbs.twimg.com/media/image.png" onerror="alert(1)" alt="A photo">
    <video src="https://video.twimg.com/clip.mp4" poster="https://pbs.twimg.com/poster.jpg" autoplay controls></video>
    <blockquote class="twitter-tweet"><cite>Quoted author</cite><p>Quoted content.</p></blockquote>`
  const rendered = renderTwitterPost(article, source)
  assert.match(rendered, /@fixture &#x3C;script>/)
  assert.match(rendered, /<time datetime="2026-09-16">/)
  assert.match(rendered, /href="https:\/\/x.com\/fixture"/)
  assert.match(rendered, /<strong>text<\/strong>/)
  assert.match(rendered, /<video[^>]+controls[^>]+preload="none"/)
  assert.match(rendered, /<blockquote><cite>Quoted author<\/cite>/)
  assert.doesNotMatch(
    rendered,
    /<script|<iframe|onclick|onerror|javascript:|autoplay|style=|twitter-tweet/,
  )
})

test('Defuddle network extraction preserves quotes, photos, video, articles, and oEmbed fallback', async t => {
  const requested: string[] = []
  const server = createServer((request, response) => {
    const target = new URL(request.url ?? '/', 'http://localhost').searchParams.get('url') ?? ''
    requested.push(target)
    response.setHeader('Content-Type', 'application/json')
    if (target === 'https://api.fxtwitter.com/garden_fixture/status/101') {
      response.end(
        JSON.stringify({
          tweet: {
            author: { screen_name: 'garden_fixture' },
            text: 'A complete post with its own media.',
            created_at: '2026-09-16T00:00:00Z',
            quote: { url: 'https://x.com/quoted_fixture/status/102' },
            media: {
              photos: [{ url: 'https://pbs.twimg.com/media/photo.png' }],
              videos: [
                {
                  url: 'https://video.twimg.com/clip.mp4',
                  thumbnail_url: 'https://pbs.twimg.com/poster.jpg',
                },
              ],
            },
          },
        }),
      )
    } else if (target === 'https://api.fxtwitter.com/quoted_fixture/status/102') {
      response.end(
        JSON.stringify({
          tweet: {
            author: { screen_name: 'quoted_fixture' },
            text: 'The quoted post is extracted by Defuddle too.',
            quote: { url: source },
          },
        }),
      )
    } else if (target === 'https://api.fxtwitter.com/garden_fixture/status/104') {
      response.end(
        JSON.stringify({
          tweet: {
            author: { screen_name: 'garden_fixture' },
            article: {
              title: 'A long article on X',
              preview_text: 'Article preview',
              content: {
                blocks: [
                  {
                    type: 'header-two',
                    text: 'Article heading',
                    inlineStyleRanges: [],
                    entityRanges: [],
                  },
                  {
                    type: 'unstyled',
                    text: 'The full article body.',
                    inlineStyleRanges: [],
                    entityRanges: [],
                  },
                ],
                entityMap: [],
              },
            },
          },
        }),
      )
    } else if (target === 'https://api.fxtwitter.com/garden_fixture/status/105') {
      response.end(JSON.stringify({ text: 'x'.repeat(2 * 1024 * 1024) }))
    } else if (target.startsWith('https://publish.twitter.com/oembed?') && target.includes('103')) {
      response.end(
        JSON.stringify({
          author_url: 'https://x.com/garden_fixture',
          html: '<blockquote class="twitter-tweet"><p>The provider fallback still uses native styles.</p></blockquote><script src="https://platform.twitter.com/widgets.js"></script>',
        }),
      )
    } else {
      response.statusCode = 404
      response.end('{}')
    }
  })
  server.listen(0, '127.0.0.1')
  await once(server, 'listening')
  t.after(
    () =>
      new Promise<void>((resolve, reject) => {
        server.close(error => (error ? reject(error) : resolve()))
        server.closeAllConnections()
      }),
  )
  const address = server.address()
  assert.ok(address && typeof address !== 'string')
  const origin = `http://127.0.0.1:${address.port}`
  const fixtureFetch: typeof fetch = (resource, init) => {
    const request = new Request(resource, init)
    return fetch(`${origin}/extractor?url=${encodeURIComponent(request.url)}`, {
      signal: request.signal,
      redirect: request.redirect,
    })
  }
  const post = await extractTwitterPost(source, fixtureFetch)
  assert.match(post, /A complete post with its own media/)
  assert.match(
    post,
    /<blockquote>[\s\S]*quoted_fixture[\s\S]*The quoted post is extracted by Defuddle too/,
  )
  assert.match(post, /<img[^>]+photo.png/)
  assert.match(post, /<video[^>]+clip.mp4/)
  assert.equal(requested.filter(url => url.endsWith('/garden_fixture/status/101')).length, 1)
  const fallback = await extractTwitterPost('https://x.com/garden_fixture/status/103', fixtureFetch)
  assert.match(fallback, /The provider fallback still uses native styles/)
  const article = await extractTwitterPost('https://x.com/garden_fixture/status/104', fixtureFetch)
  assert.match(article, /A long article on X/)
  assert.match(article, /<h2>Article heading<\/h2>/)
  assert.match(article, /The full article body/)
  for (const id of ['105', '106']) {
    const unavailable = await extractTwitterPost(
      `https://x.com/garden_fixture/status/${id}`,
      fixtureFetch,
    )
    assert.match(unavailable, /Post unavailable/)
    assert.match(unavailable, new RegExp(`href="https://x.com/garden_fixture/status/${id}"`))
  }
  for (const html of [post, fallback, article]) {
    assert.match(html, /class="twitter-post"/)
    assert.doesNotMatch(html, /widgets\.js|twitter-tweet|<script|<iframe/)
  }
})

test('X extraction follows bounded provider redirects and rejects unsupported destinations', async t => {
  const requested: URL[] = []
  let location: string | undefined
  let redirectStatus = 301
  let loop = false
  let oversized = false
  const server = createServer((request, response) => {
    const target = new URL(new URL(request.url ?? '/', 'http://localhost').searchParams.get('url')!)
    requested.push(target)
    if (target.hostname === 'api.fxtwitter.com') {
      response.writeHead(404).end('{}')
      return
    }
    if (target.hostname === 'publish.twitter.com' || loop) {
      response.writeHead(redirectStatus, location ? { Location: location } : {})
      // Leave the body open so following the redirect requires cancelling it.
      response.write('redirecting')
      return
    }
    response.setHeader('Content-Type', 'application/json')
    response.end(
      JSON.stringify({
        author_url: 'https://x.com/garden_fixture',
        html: `<blockquote><p>${oversized ? 'x'.repeat(2 * 1024 * 1024) : 'The redirected post is available.'}</p></blockquote>`,
      }),
    )
  })
  server.listen(0, '127.0.0.1')
  await once(server, 'listening')
  t.after(
    () =>
      new Promise<void>((resolve, reject) => {
        server.close(error => (error ? reject(error) : resolve()))
        server.closeAllConnections()
      }),
  )
  const address = server.address()
  assert.ok(address && typeof address !== 'string')
  const origin = `http://127.0.0.1:${address.port}`
  const fixtureFetch: typeof fetch = (resource, init) => {
    const request = new Request(resource, init)
    return fetch(`${origin}/extractor?url=${encodeURIComponent(request.url)}`, {
      signal: request.signal,
      redirect: request.redirect,
    })
  }

  for (const status of [301, 302, 303, 307, 308]) {
    await t.test(`follows ${status} from publish.twitter.com to publish.x.com`, async () => {
      requested.length = 0
      redirectStatus = status
      location = `https://publish.x.com/oembed?url=${encodeURIComponent(source)}&omit_script=true`
      const post = await extractTwitterPost(source, fixtureFetch)
      assert.match(post, /The redirected post is available/)
      assert.deepEqual(
        requested.map(url => url.hostname),
        ['api.fxtwitter.com', 'publish.twitter.com', 'publish.x.com'],
      )
      assert.equal(requested[2].searchParams.get('url'), source)
      assert.equal(requested[2].searchParams.get('omit_script'), 'true')
    })
  }

  for (const destination of [
    'https://untrusted.example/oembed',
    'http://publish.x.com/oembed',
    'https://publish.x.com:8443/oembed',
    'https://user:password@publish.x.com/oembed',
    'http://[invalid',
    undefined,
  ]) {
    await t.test(`rejects redirect destination ${destination ?? '(missing)'}`, async () => {
      requested.length = 0
      location = destination
      const post = await extractTwitterPost(source, fixtureFetch)
      assert.match(post, /Post unavailable/)
      assert.ok(requested.every(url => url.hostname !== 'publish.x.com'))
      assert.ok(
        requested.every(url => ['api.fxtwitter.com', 'publish.twitter.com'].includes(url.hostname)),
      )
    })
  }

  await t.test('counts relative redirect loops against the shared request limit', async () => {
    requested.length = 0
    location = '/oembed?loop=true'
    loop = true
    const post = await extractTwitterPost(source, fixtureFetch)
    assert.match(post, /Post unavailable/)
    assert.equal(requested.length, 8)
    assert.equal(requested[2].href, 'https://publish.twitter.com/oembed?loop=true')
    loop = false
  })

  await t.test('keeps the response size limit after a redirect', async () => {
    location = 'https://publish.x.com/oembed'
    oversized = true
    const post = await extractTwitterPost(source, fixtureFetch)
    assert.match(post, /Post unavailable/)
  })
})
