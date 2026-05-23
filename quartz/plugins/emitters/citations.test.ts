import assert from 'node:assert'
import { mkdtemp, readFile, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'
import { buildCachePayload, hydrateCache } from '../stores/citations'
import { ensureBibEntries } from './citations'

test('repairs active arxiv ids missing from the paper cache', async t => {
  const originalFetch = globalThis.fetch
  const originalCache = buildCachePayload()

  t.after(() => {
    globalThis.fetch = originalFetch
    hydrateCache(originalCache)
  })

  hydrateCache({ papers: {}, documents: {} })
  const root = await mkdtemp(path.join(os.tmpdir(), 'quartz-citations-'))
  const bibliography = path.join(root, 'References.bib')
  await writeFile(bibliography, '')

  const requests: string[] = []
  const fetchStub: typeof fetch = async input => {
    const url = String(input)
    requests.push(url)
    if (url.includes('https://export.arxiv.org/api/query')) {
      return new Response(
        [
          '<?xml version="1.0" encoding="UTF-8"?>',
          '<feed xmlns:arxiv="http://arxiv.org/schemas/atom">',
          '<entry>',
          '<id>http://arxiv.org/abs/2605.12290v1</id>',
          '<title>Cache Repair for Missing Paper Rows</title>',
          '<published>2026-05-19T00:00:00Z</published>',
          '<author><name>Ada Lovelace</name></author>',
          '<arxiv:primary_category term="cs.LG" />',
          '</entry>',
          '</feed>',
        ].join(''),
        { status: 200 },
      )
    }

    if (url === 'https://arxiv.org/bibtex/2605.12290') {
      return new Response(
        [
          '@misc{herring2026targetedneuronmodulationcontrastive,',
          '      title={Targeted Neuron Modulation via Contrastive Pair Search}, ',
          '      author={Sam Herring and Jake Naviasky and Karan Malhotra},',
          '      year={2026},',
          '      eprint={2605.12290},',
          '      archivePrefix={arXiv},',
          '      primaryClass={cs.LG},',
          '      url={https://arxiv.org/abs/2605.12290}, ',
          '}',
        ].join('\n'),
        { status: 200 },
      )
    }

    assert.fail(`unexpected fetch ${url}`)
  }
  globalThis.fetch = fetchStub

  await ensureBibEntries(['2605.12290'], bibliography)

  const refs = await readFile(bibliography, 'utf8')
  const cache = buildCachePayload()

  assert.deepStrictEqual(requests, [
    'https://export.arxiv.org/api/query?id_list=2605.12290',
    'https://arxiv.org/bibtex/2605.12290',
  ])
  assert.match(refs, /@misc\{herring2026targetedneuronmodulationcontrastive,/)
  assert.match(refs, /title=\{Targeted Neuron Modulation via Contrastive Pair Search\}/)
  assert.match(refs, /author=\{Sam Herring and Jake Naviasky and Karan Malhotra\}/)
  assert.strictEqual(cache.papers['2605.12290'].inBibFile, true)
  assert.strictEqual(
    cache.papers['2605.12290'].bibkey,
    'herring2026targetedneuronmodulationcontrastive',
  )
})

test('syncs existing bibliography keys back into the citation cache', async t => {
  const originalFetch = globalThis.fetch
  const originalCache = buildCachePayload()

  t.after(() => {
    globalThis.fetch = originalFetch
    hydrateCache(originalCache)
  })

  hydrateCache({
    papers: {
      '2605.12290': {
        title: 'Targeted Neuron Modulation via Contrastive Pair Search',
        bibkey: 'arxiv-260512290',
        lastVerified: 0,
        inBibFile: true,
        bibtex: '@article{arxiv-260512290,\n  eprint = {2605.12290}\n}',
      },
    },
    documents: {},
  })

  const root = await mkdtemp(path.join(os.tmpdir(), 'quartz-citations-'))
  const bibliography = path.join(root, 'References.bib')
  await writeFile(
    bibliography,
    [
      '@misc{herring2026targetedneuronmodulationcontrastive,',
      '      title={Targeted Neuron Modulation via Contrastive Pair Search}, ',
      '      author={Sam Herring and Jake Naviasky and Karan Malhotra},',
      '      year={2026},',
      '      eprint={2605.12290},',
      '      archivePrefix={arXiv},',
      '      primaryClass={cs.LG},',
      '      url={https://arxiv.org/abs/2605.12290}, ',
      '}',
    ].join('\n'),
  )

  globalThis.fetch = async input => assert.fail(`unexpected fetch ${String(input)}`)

  await ensureBibEntries(['2605.12290'], bibliography)

  const cache = buildCachePayload()
  assert.strictEqual(
    cache.papers['2605.12290'].bibkey,
    'herring2026targetedneuronmodulationcontrastive',
  )
  assert.strictEqual(cache.papers['2605.12290'].inBibFile, true)
})
