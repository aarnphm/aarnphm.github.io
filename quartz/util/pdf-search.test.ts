import assert from 'node:assert/strict'
import test from 'node:test'
import { pdfSearchResults, rankPdfResults, type PdfPassage } from './pdf-search'

const passages: PdfPassage[] = [
  {
    slug: 'fastcdc.pdf',
    page: 2,
    pageLabel: '101',
    text: 'Chunk boundaries follow content.',
    title: 'FastCDC',
    chunk: 0,
  },
  {
    slug: 'fastcdc.pdf',
    page: 3,
    pageLabel: '102',
    text: 'A faster rolling hash.',
    title: 'FastCDC',
    chunk: 0,
  },
  {
    slug: 'moe.pdf',
    page: 1,
    pageLabel: '',
    text: 'Experts route tokens.',
    title: 'DeepSeek-V3 Technical Report',
    chunk: 0,
  },
]
const metadata = {
  'fastcdc.pdf': {
    title: 'FastCDC',
    aliases: ['Fast and Efficient Content-Defined Chunking'],
    acronyms: ['CDC'],
  },
  'moe.pdf': { title: 'DeepSeek-V3 Technical Report', aliases: [], acronyms: ['MoE', 'MLA'] },
}

test('PDF retrieval returns one paper with the best physical page and its passage', () => {
  const results = pdfSearchResults(
    [
      { id: 0, score: 0.6 },
      { id: 1, score: 0.9 },
      { id: 2, score: 0.7 },
    ],
    passages,
  )
  assert.deepEqual(
    results.map(result => [result.slug, result.passage.page]),
    [
      ['fastcdc.pdf', 3],
      ['moe.pdf', 1],
    ],
  )
  assert.equal(results[0].passage.text, 'A faster rolling hash.')
  assert.equal(results[0].passage.pageLabel, '102')
})

test('exact titles and acronyms stay ahead of unrelated semantic matches', () => {
  const semantic = pdfSearchResults(
    [
      { id: 0, score: 0.99 },
      { id: 2, score: 0.5 },
    ],
    passages,
  )
  const allowed = new Set(Object.keys(metadata))
  for (const query of ['DeepSeek-V3 Technical Report', 'MoE', 'MLA']) {
    assert.equal(rankPdfResults(query, [], semantic, metadata, allowed)[0].slug, 'moe.pdf')
  }
  assert.equal(rankPdfResults('CDC', [], semantic, metadata, allowed)[0].slug, 'fastcdc.pdf')
})

test('conceptual queries use semantic evidence while excluded papers cannot surface', () => {
  const semantic = pdfSearchResults(
    [
      { id: 0, score: 0.4 },
      { id: 2, score: 0.9 },
    ],
    passages,
  )
  assert.equal(
    rankPdfResults(
      'sparse token routing',
      [],
      semantic,
      metadata,
      new Set(Object.keys(metadata)),
    )[0].slug,
    'moe.pdf',
  )
  assert.deepEqual(
    rankPdfResults('MoE', ['moe.pdf'], semantic, metadata, new Set(['fastcdc.pdf'])).map(
      result => result.slug,
    ),
    ['fastcdc.pdf'],
  )
})

test('lexical matches survive an absent PDF index and retain their order', () => {
  const results = rankPdfResults(
    'paper',
    ['unindexed.pdf', 'another.pdf'],
    [],
    {},
    new Set(['unindexed.pdf', 'another.pdf']),
  )
  assert.deepEqual(
    results.map(result => result.slug),
    ['unindexed.pdf', 'another.pdf'],
  )
  assert.equal(results[0].passage, undefined)
})
