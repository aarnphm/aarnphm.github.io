import assert from 'node:assert/strict'
import { test } from 'node:test'
import type { SharedView } from '../share'
import { bytesToBase64 } from '../../base64url'
import { interpretReference } from '../../references/target'
import { decodeLatentView, latentEmbedSnippet, parseLatentFragment } from './fragment'

const view: SharedView = {
  v: 1,
  base: '/embeddings/latent/build',
  focus: 3716,
  tab: 'features',
  features: {
    selected: { dict: 'text', f: 939 },
    steer: [
      ['text:939', -2.2],
      ['base:124', -1.5],
    ],
    explain: true,
    query: '',
    pin: false,
    cluster: false,
  },
  bench: null,
}
const payload = bytesToBase64(new TextEncoder().encode(JSON.stringify(view)))

test('focus and view segments parse', () => {
  const parsed = parseLatentFragment(['focus=3716', `view=${payload}`])
  assert.equal(parsed.focus, 3716)
  assert.equal(parsed.view, payload)
  assert.deepEqual(parsed.errors, [])
  assert.deepEqual(decodeLatentView(payload), view)
})

test('focus alone, bad focus, bad payload and unknown keys', () => {
  assert.deepEqual(parseLatentFragment(['focus=12']), { focus: 12, errors: [], unsupported: [] })
  assert.match(parseLatentFragment(['focus=x']).errors[0], /row number/)
  assert.match(parseLatentFragment(['view=bm90IGpzb24']).errors[0], /does not decode/)
  assert.deepEqual(parseLatentFragment(['focus=1', 'zoom=2']).unsupported, ['zoom'])
})

test('a latent page reads its fragments as a latent selector', () => {
  const entry = { path: 'embeddings/index.md', kind: 'note' as const, latent: true }
  const ok = interpretReference(entry, { fragments: ['focus=3716', `view=${payload}`] })
  assert.equal(ok.selector.kind, 'latent')
  assert.deepEqual(ok.diagnostics, [])
  const bad = interpretReference(entry, { fragments: ['focus=3716', 'colour=red'] })
  assert.equal(bad.diagnostics[0].code, 'unsupported-selector')
  // without the flag the same fragments are a heading
  const note = interpretReference(
    { path: 'thoughts/x.md', kind: 'note', headings: [] },
    { fragments: ['focus=3716'] },
  )
  assert.equal(note.selector.kind, 'heading')
})

test('the copied embed round-trips', () => {
  const snippet = latentEmbedSnippet('embeddings', view, payload)
  assert.equal(snippet, `![[embeddings#focus=3716#view=${payload}]]`)
  const fragments = snippet.slice(3, -2).split('#').slice(1)
  assert.deepEqual(decodeLatentView(parseLatentFragment(fragments).view!), view)
})
