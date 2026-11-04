import assert from 'node:assert/strict'
import test from 'node:test'
import type { LatentDoc, LatentKind } from './types'
import { completeGoto, gotoIndex, matchDocs, parseGoto } from './goto'

let next = 0
const doc = (id: string, t: string, s: LatentKind = 'note'): LatentDoc => ({
  d: next++,
  id,
  s,
  t,
  date: null,
  prec: null,
  r0: next * 10,
  n: 3,
})

const docs = [
  doc('thoughts/Lebesgue-measure', 'Lebesgue measure'),
  doc("thoughts/Arrow's-impossibility-theorem", "Arrow's impossibility theorem"),
  doc('thoughts/pdfs/TRUTH-AND-PROBABILITY-(1926).pdf', 'TRUTH AND PROBABILITY (1926)', 'pdf'),
  doc('thoughts/pdfs/linear-transformers-deltanet-talk.pdf', 'Attention Is All You Need', 'pdf'),
  doc('thoughts/attention', 'attention'),
  doc('library/1984', '1984', 'base'),
  doc('library/posters/1984.jpg', '1984.jpg', 'image'),
  doc('thoughts/papers/attention-sinks.pdf', 'Efficient Streaming Language Models', 'pdf'),
]
const index = gotoIndex(docs)
const ids = (needle: string) => matchDocs(index, needle).map(d => d.id)

test('parseGoto reads only a goto: prefix', () => {
  assert.equal(parseGoto('goto:'), '')
  assert.equal(parseGoto('  GoTo:  lebesgue '), 'lebesgue')
  assert.equal(parseGoto('#goto'), null)
  assert.equal(parseGoto('go to lebesgue'), null)
})

test('an exact title ranks first, then title prefixes', () => {
  assert.deepEqual(ids('attention').slice(0, 2), [
    'thoughts/attention',
    'thoughts/pdfs/linear-transformers-deltanet-talk.pdf',
  ])
})

test('slug basenames match with dashes read as spaces', () => {
  assert.equal(ids('deltanet')[0], 'thoughts/pdfs/linear-transformers-deltanet-talk.pdf')
  assert.equal(ids('attention sinks')[0], 'thoughts/papers/attention-sinks.pdf')
})

test('every typed word prefixes some word', () => {
  assert.equal(ids('arr imposs')[0], "thoughts/Arrow's-impossibility-theorem")
  assert.equal(ids('truth prob')[0], 'thoughts/pdfs/TRUTH-AND-PROBABILITY-(1926).pdf')
})

test('ties prefer notes and pdfs over bases and images', () => {
  assert.deepEqual(ids('1984'), ['library/1984', 'library/posters/1984.jpg'])
})

test('a slash goes through the path matcher', () => {
  assert.equal(ids('/thoughts/pdfs/truth')[0], 'thoughts/pdfs/TRUTH-AND-PROBABILITY-(1926).pdf')
  assert.deepEqual(ids('library/posters/'), ['library/posters/1984.jpg'])
})

test('loose subsequences need four letters, close together', () => {
  assert.equal(ids('lbsg')[0], 'thoughts/Lebesgue-measure')
  assert.deepEqual(ids('lbs'), [])
  assert.deepEqual(ids('lgmr'), [])
})

test('Tab fills the full id', () => {
  assert.equal(completeGoto(docs[0]), 'goto:thoughts/Lebesgue-measure')
})
