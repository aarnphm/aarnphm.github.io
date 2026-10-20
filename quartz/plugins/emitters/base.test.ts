import assert from 'node:assert/strict'
import test from 'node:test'
import type { FilePath, FullSlug, SimpleSlug } from '../../util/path'
import type { ProcessedContent } from '../vfile'
import { compileBaseConfig } from '../../util/base/compile'
import { planBaseViewPartialEmit } from '../../util/base/partial-emit'
import { defaultProcessedContent } from '../vfile'

const baseSource = `
filters:
  and:
    - file.inFolder("library")
views:
  - type: table
    name: all
    order:
      - file.name
      - file.backlinks
`

function baseFile(): ProcessedContent {
  const compiled = compileBaseConfig(baseSource, 'library.base')
  return defaultProcessedContent({
    slug: 'library' as FullSlug,
    filePath: 'library.base' as FilePath,
    relativePath: 'library.base' as FilePath,
    bases: true,
    basesConfig: compiled.config,
    basesDiagnostics: compiled.diagnostics,
    basesExpressions: compiled.expressions,
    frontmatter: { title: 'library', pageLayout: 'default', tags: ['bases'] },
  })
}

function note(slug: string, relativePath: string, links: string[] = []): ProcessedContent {
  const title = slug.split('/').pop() ?? slug
  return defaultProcessedContent({
    slug: slug as FullSlug,
    filePath: relativePath as FilePath,
    relativePath: relativePath as FilePath,
    frontmatter: { title, pageLayout: 'default' },
    links: links.map(link => link as SimpleSlug),
  })
}

const rebuiltSlugs = (plan: ReturnType<typeof planBaseViewPartialEmit>): string[] =>
  [...plan.slugsToRebuild].sort()

test('base view partial emit ignores unrelated markdown changes', () => {
  const base = baseFile()
  const index = note('index', 'index.md')
  const book = note('library/book', 'library/book.md')
  const content = [base, index, book]
  const previous = planBaseViewPartialEmit(content, []).nextState

  const plan = planBaseViewPartialEmit(
    content,
    [{ type: 'change', path: 'index.md' as FilePath, file: index[1] }],
    previous,
  )

  assert.deepEqual(rebuiltSlugs(plan), [])
})

test('base view partial emit rebuilds when a matched file changes', () => {
  const base = baseFile()
  const index = note('index', 'index.md')
  const book = note('library/book', 'library/book.md')
  const content = [base, index, book]
  const previous = planBaseViewPartialEmit(content, []).nextState

  const plan = planBaseViewPartialEmit(
    content,
    [{ type: 'change', path: 'library/book.md' as FilePath, file: book[1] }],
    previous,
  )

  assert.deepEqual(rebuiltSlugs(plan), ['library'])
})

test('base view partial emit ignores matched file body changes with stable metadata', () => {
  const base = baseFile()
  const index = note('index', 'index.md')
  const previousBook = note('library/book', 'library/book.md')
  const previousContent = [base, index, previousBook]
  const previous = planBaseViewPartialEmit(previousContent, []).nextState
  const nextBook = note('library/book', 'library/book.md')

  const plan = planBaseViewPartialEmit(
    [base, index, nextBook],
    [
      {
        type: 'change',
        path: 'library/book.md' as FilePath,
        file: nextBook[1],
        previousFile: previousBook[1],
      },
    ],
    previous,
  )

  assert.deepEqual(rebuiltSlugs(plan), [])
})

test('base view partial emit rebuilds when matched file metadata changes', () => {
  const base = baseFile()
  const index = note('index', 'index.md')
  const previousBook = note('library/book', 'library/book.md')
  const previousContent = [base, index, previousBook]
  const previous = planBaseViewPartialEmit(previousContent, []).nextState
  const nextBook = note('library/book', 'library/book.md')
  nextBook[1].data.frontmatter = { title: 'book, revised', pageLayout: 'default' }

  const plan = planBaseViewPartialEmit(
    [base, index, nextBook],
    [
      {
        type: 'change',
        path: 'library/book.md' as FilePath,
        file: nextBook[1],
        previousFile: previousBook[1],
      },
    ],
    previous,
  )

  assert.deepEqual(rebuiltSlugs(plan), ['library'])
})

test('base view partial emit rebuilds when a file enters the base result set', () => {
  const base = baseFile()
  const index = note('index', 'index.md')
  const previousContent = [base, index]
  const previous = planBaseViewPartialEmit(previousContent, []).nextState
  const book = note('library/book', 'library/book.md')

  const plan = planBaseViewPartialEmit(
    [base, index, book],
    [{ type: 'add', path: 'library/book.md' as FilePath, file: book[1] }],
    previous,
  )

  assert.deepEqual(rebuiltSlugs(plan), ['library'])
})

test('base view partial emit rebuilds when backlinks to a matched file change', () => {
  const base = baseFile()
  const book = note('library/book', 'library/book.md')
  const oldIndex = note('index', 'index.md')
  const previousContent = [base, oldIndex, book]
  const previous = planBaseViewPartialEmit(previousContent, []).nextState
  const newIndex = note('index', 'index.md', ['library/book'])

  const plan = planBaseViewPartialEmit(
    [base, newIndex, book],
    [{ type: 'change', path: 'index.md' as FilePath, file: newIndex[1] }],
    previous,
  )

  assert.deepEqual(rebuiltSlugs(plan), ['library'])
})

test('base view partial emit rebuilds when a linked non-member changes', () => {
  const base = baseFile()
  const index = note('index', 'index.md')
  const book = note('library/book', 'library/book.md', ['/'])
  const content = [base, index, book]
  const previous = planBaseViewPartialEmit(content, []).nextState

  const plan = planBaseViewPartialEmit(
    content,
    [{ type: 'change', path: 'index.md' as FilePath, file: index[1] }],
    previous,
  )

  assert.deepEqual(rebuiltSlugs(plan), ['library'])
})

test('createBaseQueryData compacts empty arrays, default filePath/pageLayout, and pdfTitles while round-tripping via readBaseQueryData', async () => {
  const { createBaseQueryData, readBaseQueryData } = await import('../../util/base/browser-data')
  const book = note('library/book', 'library/book.md')[1].data
  book.frontmatter = {
    title: 'Book Title',
    pageLayout: 'default',
    author: 'Ada Lovelace',
    pdfTitles: { 'folder/paper.pdf': 'Internal PDF Title' },
  }
  const payload = createBaseQueryData([book])
  assert.equal(payload.files.length, 1)
  const rawRecord = payload.files[0] as unknown as Record<string, unknown>
  assert.equal(rawRecord.filePath, undefined)
  assert.equal(rawRecord.datePaths, undefined)
  assert.equal(rawRecord.links, undefined)
  assert.equal(rawRecord.embeds, undefined)
  assert.equal((rawRecord.frontmatter as Record<string, unknown>).pageLayout, undefined)
  assert.equal((rawRecord.frontmatter as Record<string, unknown>).pdfTitles, undefined)

  const hydrated = readBaseQueryData(JSON.parse(JSON.stringify(payload)))
  assert.equal(hydrated.length, 1)
  assert.equal(hydrated[0].slug, 'library/book')
  assert.equal(hydrated[0].filePath, 'library/book.md')
  assert.equal(hydrated[0].frontmatter?.title, 'Book Title')
  assert.equal(hydrated[0].frontmatter?.pageLayout, 'default')
  assert.equal(hydrated[0].frontmatter?.author, 'Ada Lovelace')
  assert.deepEqual(hydrated[0].links, [])
  assert.deepEqual(hydrated[0].embeds, [])
})

test('createBaseQueryCatalog excludes nested dictionary bloat and this.file.* from sort properties while keeping top-level and configured properties', async () => {
  const { createBaseQueryCatalog } = await import('../../components/scripts/base-query-editor')
  const compiled = compileBaseConfig(
    `
formulas:
  age_years: 2026 - number(year)
properties:
  note.custom.nested:
    displayName: Custom Nested
views:
  - type: table
    name: all
`,
    'library.base',
  )
  const book = note('library/book', 'library/book.md')[1].data
  book.frontmatter = {
    title: 'Book Title',
    pageLayout: 'default',
    author: 'Ada Lovelace',
    year: 1843,
    pdfTitles: { 'folder/scan.pdf': 'Scan Title' },
    socials: { twitter: 'https://x.com/ada' },
  }
  const catalog = createBaseQueryCatalog([book], compiled.config)
  assert.ok(catalog.properties.includes('note.author'))
  assert.ok(catalog.properties.includes('note.year'))
  assert.ok(catalog.properties.includes('formula.age_years'))
  assert.ok(catalog.properties.includes('note.custom.nested'))
  assert.ok(catalog.properties.includes('file.name'))
  assert.ok(!catalog.properties.some(prop => prop.startsWith('note.pdfTitles')))
  assert.ok(!catalog.properties.some(prop => prop.startsWith('note.socials.')))
  assert.ok(!catalog.properties.some(prop => prop.startsWith('this.')))
})

test('renderBaseViewsForFile marks card images for thumbnail generation and omits empty table cell wrappers', async () => {
  const { renderBaseViewsForFile } = await import('../../util/base/render')
  const { toHtml } = await import('hast-util-to-html')
  const compiled = compileBaseConfig(
    `
filters:
  and:
    - file.inFolder("library")
views:
  - type: cards
    name: gallery
    image: note.posters
    order:
      - title
      - author
  - type: table
    name: editors
    order:
      - title
      - author
      - missing_col
      - read_flag
`,
    'library.base',
  )
  const baseData = defaultProcessedContent({
    slug: 'library' as FullSlug,
    filePath: 'library.base' as FilePath,
    relativePath: 'library.base' as FilePath,
    bases: true,
    basesConfig: compiled.config,
    basesDiagnostics: compiled.diagnostics,
    basesExpressions: compiled.expressions,
  })[1].data
  const book = note('library/book', 'library/book.md')[1].data
  book.frontmatter = {
    title: 'Book Title',
    pageLayout: 'default',
    author: 'Ada Lovelace',
    posters: '[[library/posters/book.jpg]]',
    read_flag: true,
  }
  const rendered = renderBaseViewsForFile(baseData, [book], baseData)
  const galleryHtml = toHtml(rendered.views[0].tree)
  assert.match(galleryHtml, /data-bookcase-cover="library\/posters\/book\.jpg"/)

  const tableHtml = toHtml(rendered.views[1].tree)
  assert.match(tableHtml, /<td><\/td>/)
  assert.doesNotMatch(tableHtml, /<div class="base-table-cell-content"><\/div>/)
})

test('pele.inline.ts stays a compact lazy loader when bundled into a synchronous IIFE resource', async () => {
  const { bundleInlineScript } = await import('../../util/inline-script-bundler')
  const { joinScripts } = await import('./component-resources/script-assets')
  const inline = await bundleInlineScript('quartz/components/scripts/pele.inline.ts')
  const bundled = await joinScripts([inline])
  assert.ok(
    Buffer.byteLength(bundled, 'utf8') < 4096,
    `expected pele.inline.ts IIFE bundle < 4096 B, got ${Buffer.byteLength(bundled, 'utf8')} B`,
  )
  assert.doesNotMatch(bundled, /KaTeX parse error/)
})

test('writePageScripts bundles afterDOMLoaded scripts into a single static/scripts/script module in source order', async () => {
  const { mkdtemp, readFile } = await import('node:fs/promises')
  const os = await import('node:os')
  const path = await import('node:path')
  const { writePageScripts } = await import('./component-resources/script-assets')
  const output = await mkdtemp(path.join(os.tmpdir(), 'quartz-page-scripts-'))
  const ctx = {
    buildId: 'test-page-scripts',
    argv: {
      directory: 'content',
      output,
      verbose: false,
      serve: true,
      watch: false,
      port: 8080,
      wsPort: 3001,
      force: false,
    },
    cfg: {} as never,
    allSlugs: [],
    allFiles: [],
    incremental: false,
  }
  const files = await writePageScripts(ctx, {
    css: [],
    componentCss: [],
    beforeDOMLoaded: ['window.__before = 1;'],
    afterDOMLoaded: [
      'window.__order = (window.__order ?? []).concat("first");',
      'window.__order = (window.__order ?? []).concat("first");',
      'window.__order = (window.__order ?? []).concat("second");',
      'window.__order = (window.__order ?? []).concat("spa");',
    ],
  })
  const scriptModules = files.filter(file => file.includes('static/scripts/script-'))
  assert.equal(
    scriptModules.length,
    1,
    `expected 1 combined afterDOMLoaded entry module, got ${scriptModules.length}`,
  )
  const postscript = await readFile(path.join(output, 'postscript.js'), 'utf8')
  assert.doesNotMatch(postscript, /for \(const src of chunks\) await import\(src\)/)
  const combined = await readFile(scriptModules[0], 'utf8')
  const firstIdx = combined.indexOf('"first"')
  const secondIdx = combined.indexOf('"second"')
  const spaIdx = combined.indexOf('"spa"')
  assert.ok(firstIdx >= 0 && secondIdx > firstIdx && spaIdx > secondIdx)
  assert.equal(combined.indexOf('"first"', firstIdx + 1), -1)
})

test('createHardcoverPreview sizes mount immediately on open, decodes cover and spine concurrently, and defers full-res texture swap until reveal finishes', async () => {
  const events: string[] = []
  let resolveCoverDecode!: () => void
  let resolveOriginalDecode!: () => void

  class FakeElement {
    tagName: string
    className = ''
    dataset: Record<string, string> = {}
    style: Record<string, string> & {
      setProperty: (k: string, v: string) => void
      getPropertyValue: (k: string) => string
    }
    children: FakeElement[] = []
    attributes = new Map<string, string>()
    listeners = new Map<string, ((e?: unknown) => void)[]>()
    textContent = ''
    hidden = false
    alt = ''
    decoding = ''
    draggable = false
    complete = true
    naturalWidth = 320
    naturalHeight = 480
    clientWidth = 0
    clientHeight = 0
    private _src = ''

    constructor(tagName: string) {
      this.tagName = tagName
      const styleStore: Record<string, string> = {}
      this.style = Object.assign(styleStore, {
        setProperty: (k: string, v: string) => {
          styleStore[k] = v
        },
        getPropertyValue: (k: string) => styleStore[k] ?? '',
      })
    }
    get src() {
      return this._src
    }
    set src(value: string) {
      this._src = value
      this.attributes.set('src', value)
    }
    setAttribute(k: string, v: string) {
      this.attributes.set(k, v)
    }
    getAttribute(k: string) {
      return this.attributes.get(k) ?? null
    }
    removeAttribute(k: string) {
      this.attributes.delete(k)
      if (k === 'src') this._src = ''
    }
    toggleAttribute(k: string, force?: boolean) {
      const next = force ?? !this.attributes.has(k)
      if (next) this.attributes.set(k, '')
      else this.attributes.delete(k)
      if (k === 'data-art-ready') {
        if (next) this.dataset.artReady = ''
        else delete this.dataset.artReady
      }
      return next
    }
    append(...nodes: FakeElement[]) {
      this.children.push(...nodes)
    }
    replaceChildren(...nodes: FakeElement[]) {
      this.children = [...nodes]
    }
    addEventListener(type: string, cb: (e?: unknown) => void) {
      const list = this.listeners.get(type) ?? []
      list.push(cb)
      this.listeners.set(type, list)
    }
    animate() {
      let cancelled = false
      const anim = {
        onfinish: null as (() => void) | null,
        cancel: () => {
          cancelled = true
          events.push('animation:cancel')
        },
        finish: () => {
          if (!cancelled) anim.onfinish?.()
        },
      }
      ;(globalThis as unknown as { __lastAnims: (typeof anim)[] }).__lastAnims.push(anim)
      return anim
    }
    decode() {
      if (this._src.includes('thumb.webp')) {
        events.push('decode:thumb:start')
        return new Promise<void>(resolve => {
          resolveCoverDecode = () => {
            events.push('decode:thumb:done')
            resolve()
          }
        })
      }
      if (this._src.includes('spine-low.webp')) {
        events.push('decode:spine-low:start')
        return Promise.resolve()
      }
      if (this._src.includes('full.jpg')) {
        events.push('decode:full:start')
        return new Promise<void>(resolve => {
          resolveOriginalDecode = () => {
            events.push('decode:full:done')
            resolve()
          }
        })
      }
      return Promise.resolve()
    }
  }

  const prevWindow = globalThis.window
  const prevDocument = globalThis.document
  const prevImage = globalThis.Image
  const prevRO = globalThis.ResizeObserver
  const prevRAF = globalThis.requestAnimationFrame
  const prevCAF = globalThis.cancelAnimationFrame

  try {
    ;(globalThis as unknown as { __lastAnims: unknown[] }).__lastAnims = []
    let roCallback: (() => void) | undefined
    globalThis.ResizeObserver = class {
      constructor(cb: () => void) {
        roCallback = cb
      }
      observe() {
        roCallback?.()
      }
      disconnect() {}
    } as unknown as typeof ResizeObserver
    globalThis.Image = class extends FakeElement {
      constructor() {
        super('img')
      }
    } as unknown as typeof Image
    globalThis.requestAnimationFrame = (cb: FrameRequestCallback) => {
      cb(0)
      return 1
    }
    globalThis.cancelAnimationFrame = () => {}
    globalThis.window = {
      matchMedia: () => ({ matches: false, addEventListener: () => {} }),
      setTimeout: globalThis.setTimeout.bind(globalThis),
      clearTimeout: globalThis.clearTimeout.bind(globalThis),
    } as unknown as Window & typeof globalThis
    globalThis.document = {
      baseURI: 'http://localhost:8080/antilibrary',
      createElement: (tag: string) => new FakeElement(tag),
      createElementNS: (_ns: string, tag: string) => new FakeElement(tag),
    } as unknown as Document

    const { createHardcoverPreview } =
      await import('../../components/scripts/base-bookcase-preview')
    const root = new FakeElement('div')
    const controller = new AbortController()
    const preview = createHardcoverPreview(root as unknown as HTMLElement, controller.signal)
    const mount = root.children[0]
    // While dialog is closed (0x0), mount must not be forced to 1px width
    assert.notEqual(mount.style.width, '1px')

    // Open dialog (590x660) and call show()
    root.clientWidth = 590
    root.clientHeight = 660
    const showPromise = preview.show({
      cover: 'http://localhost:8080/library/posters/full.jpg',
      thumbnail: 'http://localhost:8080/static/bookcase/thumb.webp',
      spine: '#332211',
      ink: '#fffcf0',
      title: 'Test Book',
      author: 'Test Author',
      spineArt: 'http://localhost:8080/static/bookcase-spines/spine-low.webp',
    })

    // Mount must be sized immediately before thumb decode resolves, and spine decode must start in parallel
    assert.ok(
      parseFloat(mount.style.width) > 100,
      `expected mount width > 100px before decode finishes, got ${mount.style.width}`,
    )
    assert.deepEqual(events.slice(0, 2), ['decode:thumb:start', 'decode:spine-low:start'])

    resolveCoverDecode()
    const ready = await showPromise
    assert.equal(ready, true)

    preview.reveal()
    assert.equal(root.dataset.bookcaseFlying, '')

    // Wait for any deferred full-cover decode to start, then resolve it while reveal() is still flying
    await new Promise(resolve => setTimeout(resolve, 320))
    const frontFace = mount.children[0].children.find(
      c => c.className === 'bookcase-hardcover-front',
    )!
    const frontImg = frontFace.children[0]
    if (resolveOriginalDecode) resolveOriginalDecode()
    await Promise.resolve()
    await Promise.resolve()

    // While reveal() is flying, image.src must stay on thumb.webp and reveal must not be cancelled
    assert.equal(frontImg.src, 'http://localhost:8080/static/bookcase/thumb.webp')
    assert.ok(!events.includes('animation:cancel'))

    // Finishing reveal() flushes the full-res cover cleanly
    const anims = (globalThis as unknown as { __lastAnims: { finish: () => void }[] }).__lastAnims
    for (const anim of anims) anim.finish()
    assert.equal(root.dataset.bookcaseFlying, undefined)
    assert.equal(frontImg.src, 'http://localhost:8080/library/posters/full.jpg')
    controller.abort()
  } finally {
    globalThis.window = prevWindow
    globalThis.document = prevDocument
    globalThis.Image = prevImage
    globalThis.ResizeObserver = prevRO
    globalThis.requestAnimationFrame = prevRAF
    globalThis.cancelAnimationFrame = prevCAF
  }
})

test('compileBaseConfig minimizes YAML filter trees via Minato-Morreale ISOP recurrence and flat N-ary jumps', () => {
  const source = `
filters:
  and:
    - note.category.containsAny("books", "articles", "papers")
    - file.inFolder("library")
    - file.ext == "md"
    - file.basename != "index"
    - not:
        - file.inFolder("library/posters")
    - or:
        - file.inFolder("library")
        - file.hasTag("redundant")
views:
  - type: bookcase
    name: shelf
`
  const compiled = compileBaseConfig(source, 'antilibrary.base')
  assert.equal(compiled.diagnostics.length, 0)
  const filterIr = compiled.expressions.filters
  assert.ok(filterIr, 'expected compiled root filter IR')

  // The redundant `or: [file.inFolder("library"), file.hasTag("redundant")]` clause is absorbed by `file.inFolder("library")`
  assert.equal(
    filterIr.instructions.filter(ins => ins.op === 'call_method' && ins.name === 'hasTag').length,
    0,
  )

  // All 4 short-circuit branch instructions for the 5 remaining conjuncts share a single false jump target
  const falseJumps = filterIr.instructions.filter(
    (ins): ins is Extract<(typeof filterIr.instructions)[number], { target: number }> =>
      ins.op === 'jump_if_false' || ins.op === 'jump_if_true',
  )
  assert.equal(falseJumps.length, 4)
  assert.equal(new Set(falseJumps.map(j => j.target)).size, 1)
  assert.equal(
    filterIr.instructions.length,
    27,
    `expected 27 instructions after ISOP + flat N-ary jump optimization, got ${filterIr.instructions.length}`,
  )
})
