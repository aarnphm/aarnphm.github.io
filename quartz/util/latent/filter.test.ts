import assert from 'node:assert/strict'
import test from 'node:test'
import type { LatentDoc, LatentKind } from './types'
import { filterCompletions, keepsAll, parseFilter, rowMask } from './filter'

test('parseFilter: no token leaves the box alone', () => {
  const f = parseFilter('attention heads')
  assert.equal(f.kinds, null)
  assert.equal(f.present, false)
  assert.equal(f.rest, 'attention heads')
})

test('parseFilter: types join with commas or plus, in KIND_ORDER', () => {
  assert.deepEqual(parseFilter('filter:image,pdf').kinds, ['pdf', 'image'])
  assert.deepEqual(parseFilter('filter:base+notes').kinds, ['note', 'base'])
  assert.deepEqual(parseFilter('filter:pdf filter:img').kinds, ['pdf', 'image'])
})

test('parseFilter: the rest of the box is the query, wherever the token sits', () => {
  assert.equal(parseFilter('filter:pdf attention is all').rest, 'attention is all')
  assert.equal(parseFilter('attention filter:pdf heads').rest, 'attention heads')
  assert.equal(parseFilter('filter:pdf goto:lebesgue').rest, 'goto:lebesgue')
  // a word that only contains the prefix is not a token
  assert.equal(parseFilter('prefilter:pdf').present, false)
})

test('parseFilter: -type drops a kind, alone or after includes', () => {
  assert.deepEqual(parseFilter('filter:-image').kinds, ['note', 'pdf', 'base'])
  assert.deepEqual(parseFilter('filter:note,pdf,-pdf').kinds, ['note'])
  assert.deepEqual(parseFilter('filter:!base,-note').kinds, ['pdf', 'image'])
})

test('parseFilter: unknown names are reported, an empty token is present but names nothing', () => {
  const f = parseFilter('filter:pdf,video x')
  assert.deepEqual(f.kinds, ['pdf'])
  assert.deepEqual(f.unknown, ['video'])
  const empty = parseFilter('filter: x')
  assert.equal(empty.present, true)
  assert.equal(empty.kinds, null)
  assert.equal(empty.rest, 'x')
  assert.equal(parseFilter('FILTER:PDF').kinds?.[0], 'pdf')
})

test('keepsAll: a filter naming every present kind masks nothing', () => {
  const present: LatentKind[] = ['note', 'pdf']
  assert.equal(keepsAll(null, present), true)
  assert.equal(keepsAll(['note', 'pdf', 'image'], present), true)
  assert.equal(keepsAll(['pdf'], present), false)
})

test('rowMask: one byte per row by document kind', () => {
  const doc = (s: LatentKind) => ({ s }) as LatentDoc
  const rows = [doc('note'), doc('note'), doc('pdf'), doc('image')]
  assert.deepEqual([...rowMask(rows, ['pdf', 'image'], 4)], [0, 0, 1, 1])
})

test('filterCompletions: kinds left to name, replacing only the part typed', () => {
  const counts = { note: 10, pdf: 4, image: 2, base: 0 }
  const all = filterCompletions('filter:', counts).map(c => c.kind)
  assert.deepEqual(all, ['note', 'pdf', 'image'])
  const next = filterCompletions('x filter:pdf,im', counts)
  assert.deepEqual(
    next.map(c => [c.kind, c.from, c.insert]),
    [['image', 'x filter:pdf,'.length, 'image']],
  )
  assert.deepEqual(filterCompletions('filter:-n', counts)[0].insert, '-note')
  assert.deepEqual(filterCompletions('filter:pdf', counts), [])
  assert.deepEqual(filterCompletions('filter:pdf ', counts), [])
})
