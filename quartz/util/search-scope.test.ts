import assert from 'node:assert/strict'
import test from 'node:test'
import { parseSearchQuery, searchScopeIncludes } from './search-scope'

const note = {
  slug: 'thoughts/cats',
  fileName: 'thoughts/cats',
  filePath: 'content/thoughts/cats.md',
}

test('scope syntax preserves tag syntax and search terms, and the last valid scope wins', () => {
  assert.deepEqual(parseSearchQuery('  #art filter: images blue cat  '), {
    scope: 'images',
    term: '#art blue cat',
    explicit: true,
  })
  assert.deepEqual(parseSearchQuery('filter:all cat filter: PDF'), {
    scope: 'pdf',
    term: 'cat',
    explicit: true,
  })
  assert.deepEqual(parseSearchQuery('cat filter:unknown'), {
    scope: 'notes',
    term: 'cat filter:unknown',
    explicit: false,
  })
  assert.deepEqual(parseSearchQuery('filter: images'), {
    scope: 'images',
    term: '',
    explicit: true,
  })
})

test('notes include local authored documents and exclude attachments and synthetic Arena rows', () => {
  for (const extension of ['md', 'canvas', 'base', 'bases', 'ipynb']) {
    assert.equal(
      searchScopeIncludes('notes', { ...note, filePath: `content/cats.${extension}` }),
      true,
    )
  }
  for (const extension of ['jpg', 'pdf', 'mp3', 'zip']) {
    assert.equal(
      searchScopeIncludes('notes', { ...note, filePath: `content/cats.${extension}` }),
      false,
    )
  }
  assert.equal(
    searchScopeIncludes('notes', { ...note, slug: 'arena/photos', filePath: 'content/are.na.md' }),
    false,
  )
  assert.equal(
    searchScopeIncludes('notes', { ...note, slug: 'arena', filePath: 'content/are.na.md' }),
    true,
  )
})

test('image and PDF scopes use the source filename and restrict results to local files', () => {
  assert.equal(searchScopeIncludes('images', { ...note, filePath: 'content/cats.AVIF' }), true)
  assert.equal(searchScopeIncludes('images', { ...note, filePath: 'content/cats.svg' }), true)
  assert.equal(
    searchScopeIncludes('images', { ...note, filePath: 'https://example.com/cat.png' }),
    false,
  )
  assert.equal(searchScopeIncludes('pdf', { ...note, filePath: 'content/cats.PDF' }), true)
  assert.equal(searchScopeIncludes('pdf', note), false)
  assert.equal(searchScopeIncludes('all', { ...note, slug: 'arena/photos' }), true)
})

test('code filters accept language names and file extensions with last-valid semantics', () => {
  for (const alias of ['py', 'python', 'PYTHON3']) {
    assert.deepEqual(parseSearchQuery(`train filter: ${alias}`), {
      scope: 'code:python',
      term: 'train',
      explicit: true,
    })
  }
  for (const alias of ['ts', 'tsx', 'typescript']) {
    assert.equal(parseSearchQuery(`filter:${alias}`).scope, 'code:typescript')
  }
  assert.deepEqual(parseSearchQuery('filter:py cats filter:code filter:unsupported'), {
    scope: 'code',
    term: 'cats filter:unsupported',
    explicit: true,
  })
  assert.equal(parseSearchQuery('filter:code filter:images').scope, 'images')
  assert.equal(parseSearchQuery('filter: c++').scope, 'code:cpp')
  assert.equal(parseSearchQuery('filter:hs').scope, 'code:haskell')
})

test('code scope classifies source paths without admitting notes, media or unknown files', () => {
  for (const filename of ['script.py', 'view.TSX', 'kernel.cu', 'proof.lean', 'config.toml']) {
    assert.equal(searchScopeIncludes('code', { ...note, filePath: `content/${filename}` }), true)
  }
  for (const filename of ['notes.md', 'notebook.ipynb', 'image.png', 'paper.pdf', 'mystery.xyz']) {
    assert.equal(searchScopeIncludes('code', { ...note, filePath: `content/${filename}` }), false)
  }
  assert.equal(searchScopeIncludes('code:python', { ...note, filePath: 'content/Script.PY' }), true)
  assert.equal(
    searchScopeIncludes('code:typescript', { ...note, filePath: 'content/view.tsx' }),
    true,
  )
  assert.equal(
    searchScopeIncludes('code:javascript', { ...note, filePath: 'content/view.jsx' }),
    true,
  )
  assert.equal(searchScopeIncludes('code:rust', { ...note, filePath: 'content/script.py' }), false)
  assert.equal(
    searchScopeIncludes('code', { ...note, filePath: 'https://example.com/script.py' }),
    false,
  )
})

test('image scope excludes course paths in any source field while all preserves them', () => {
  const image = {
    slug: 'images/cat.png',
    fileName: 'images/cat.png',
    filePath: 'content/images/cat.png',
  }
  for (const field of ['slug', 'fileName', 'filePath']) {
    const courseImage = { ...image, [field]: 'content/thoughts/university/cat.png' }
    assert.equal(searchScopeIncludes('images', courseImage), false)
    assert.equal(searchScopeIncludes('all', courseImage), true)
  }
  assert.equal(searchScopeIncludes('images', image), true)
})
