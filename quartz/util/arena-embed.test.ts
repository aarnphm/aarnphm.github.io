import assert from 'node:assert'
import test, { describe } from 'node:test'
import {
  arenaArxivPdfUrl,
  arenaEmbedCapabilityPath,
  arenaEmbedCapturePath,
  arenaEmbedHtmlPath,
  arenaPdfFilenameFromUrl,
  arenaPdfViewerSource,
  defaultArenaExternalEmbedMode,
  isArenaPdfUrl,
  readArenaExternalEmbedMode,
} from './arena-embed'

describe('arena embeds', () => {
  test('resolves arXiv paper representations to PDFs without changing versions', () => {
    for (const [source, pdf] of [
      ['https://arxiv.org/abs/2206.00759', '2206.00759'],
      ['https://arxiv.org/abs/2206.00759v3?context=cs#abstract', '2206.00759v3'],
      ['http://www.arxiv.org/pdf/0706.0001v2.pdf', '0706.0001v2'],
      ['https://export.arxiv.org/html/2609.00001v1', '2609.00001v1'],
      ['https://arxiv.org/abs/hep-th/9901001', 'hep-th/9901001'],
      ['https://arxiv.org/pdf/math.GT/0309136v2.pdf', 'math.GT/0309136v2'],
    ]) {
      assert.strictEqual(arenaArxivPdfUrl(source), `https://arxiv.org/pdf/${pdf}`)
    }
    for (const source of [
      'https://arxiv.org/list/cs.LG/recent',
      'https://arxiv.org/abs/not-a-paper',
      'https://arxiv.org/abs/2213.00759',
      'https://arxiv.org/abs/2206.00759v0',
      'https://arxiv.org/abs/2206.00759/extra',
      'https://arxiv.org.example.com/abs/2206.00759',
      'https://user:secret@arxiv.org/abs/2206.00759',
      'https://arxiv.org:8081/abs/2206.00759',
      'file://arxiv.org/abs/2206.00759',
      '/abs/2206.00759',
    ]) {
      assert.strictEqual(arenaArxivPdfUrl(source), null, source)
    }
  })

  test('reads explicit metadata modes', () => {
    assert.strictEqual(readArenaExternalEmbedMode('auto'), 'auto')
    assert.strictEqual(readArenaExternalEmbedMode('iframe'), 'iframe')
    assert.strictEqual(readArenaExternalEmbedMode('frame'), 'iframe')
    assert.strictEqual(readArenaExternalEmbedMode('fetch'), 'fetch')
    assert.strictEqual(readArenaExternalEmbedMode('proxy'), 'fetch')
    assert.strictEqual(readArenaExternalEmbedMode('html'), 'fetch')
    assert.strictEqual(readArenaExternalEmbedMode('capture'), 'capture')
    assert.strictEqual(readArenaExternalEmbedMode('screenshot'), 'capture')
    assert.strictEqual(readArenaExternalEmbedMode('snapshot'), 'capture')
    assert.strictEqual(readArenaExternalEmbedMode('none'), 'none')
    assert.strictEqual(readArenaExternalEmbedMode('off'), 'none')
    assert.strictEqual(readArenaExternalEmbedMode('false'), 'none')
    assert.strictEqual(readArenaExternalEmbedMode('nonsense'), undefined)
  })

  test('preserves manual and GitHub disable defaults', () => {
    assert.strictEqual(defaultArenaExternalEmbedMode('https://example.com', true), 'none')
    assert.strictEqual(
      defaultArenaExternalEmbedMode('https://github.com/aarnphm/garden', false),
      'none',
    )
    assert.strictEqual(
      defaultArenaExternalEmbedMode('https://docs.github.com/en/actions', false),
      'none',
    )
    assert.strictEqual(
      defaultArenaExternalEmbedMode('https://github.com.evil.test/aarnphm/garden', false),
      'auto',
    )
  })

  test('builds same-origin API paths', () => {
    const rawUrl = 'https://stripe.dev/?q=a b'
    assert.strictEqual(
      arenaEmbedHtmlPath(rawUrl),
      '/api/arena-embed/html?url=https%3A%2F%2Fstripe.dev%2F%3Fq%3Da%20b',
    )
    assert.strictEqual(
      arenaEmbedCapabilityPath(rawUrl),
      '/api/arena-embed/capability?url=https%3A%2F%2Fstripe.dev%2F%3Fq%3Da%20b',
    )
    assert.strictEqual(
      arenaEmbedCapturePath(rawUrl),
      '/api/arena-embed/capture?url=https%3A%2F%2Fstripe.dev%2F%3Fq%3Da%20b',
    )
    assert.strictEqual(
      arenaEmbedCapturePath(rawUrl, { width: 1800, height: 920, dpr: 2 }),
      '/api/arena-embed/capture?url=https%3A%2F%2Fstripe.dev%2F%3Fq%3Da%20b&w=1800&h=920&dpr=2',
    )
  })

  test('detects PDF URLs and extracts stable filenames', () => {
    const rawUrl =
      'https://static1.squarespace.com/static/5e17b4d3834ea27accf7ef85/t/6837d373c8563c07dea5e115/1748489076368/Anderson%2C+Phenomenology+and+the+Ethics+of+Love+article+Symposium.pdf'
    assert.strictEqual(isArenaPdfUrl(rawUrl), true)
    assert.strictEqual(isArenaPdfUrl('https://stripe.dev/'), false)
    assert.strictEqual(
      arenaPdfFilenameFromUrl(rawUrl),
      'Anderson,+Phenomenology+and+the+Ethics+of+Love+article+Symposium.pdf',
    )
    assert.strictEqual(arenaPdfFilenameFromUrl('https://example.com/'), 'document.pdf')
  })

  test('routes remote PDF viewer loads through the PDF proxy', () => {
    assert.strictEqual(
      arenaPdfViewerSource('https://example.com/a paper.pdf'),
      '/api/pdf-proxy?url=https%3A%2F%2Fexample.com%2Fa%20paper.pdf',
    )
    assert.strictEqual(arenaPdfViewerSource('/local/paper.pdf'), '/local/paper.pdf')
  })
})
