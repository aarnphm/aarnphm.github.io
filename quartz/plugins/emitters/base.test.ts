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

test('buildVault emits data-base-slug on entries and initializeVault decouples modal browsing, debounces/defers full-res poster decode, and bounds zoom/lift work', async () => {
  const { buildVault } = await import('../../util/base/vault/render')
  const { toHtml } = await import('hast-util-to-html')

  const entries = [0, 1, 2, 3, 4].map(i => ({
    path: `movies/film-${i}.md`,
    slug: `movies/film-${i}`,
    href: `./movies/film-${i}`,
    title: `Film ${i}`,
    directors: [`Director ${i}`],
    year: 1950 + i * 10,
    runtime: 110 + i * 5,
    status: 'finished',
    watched: '2026-01-01',
    rating: 8.5,
    cover: `http://localhost:8080/static/bookcase/film-${i}-w320.webp`,
    coverSlug: `movies/posters/film-${i}.jpg`,
    studio: 'Studio',
    description: `Description ${i}`,
    group: `${1950 + i * 10}s`,
    rows: [],
  }))
  const tree = buildVault(entries, { name: 'vault' })
  const html = toHtml(tree)
  assert.match(html, /data-base-slug="movies\/film-0"/)

  let resolveFullDecode: (() => void) | undefined
  const createdImages: { src: string; removedSrc: boolean }[] = []
  const posterAnims: { onfinish: (() => void) | null; cancel: () => void; finish: () => void }[] =
    []

  class FakeVaultNode {
    tagName: string
    className = ''
    dataset: Record<string, string> = {}
    style: Record<string, string> & {
      setProperty: (k: string, v: string) => void
      getPropertyValue: (k: string) => string
      removeProperty: (k: string) => void
    }
    children: FakeVaultNode[] = []
    parentElement: FakeVaultNode | null = null
    attributes = new Map<string, string>()
    listeners = new Map<string, ((e: Record<string, unknown>) => void)[]>()
    textContent = ''
    href = ''
    disabled = false
    open = false
    isConnected = true
    clientWidth = 800
    clientHeight = 600
    scrollLeft = 0
    scrollTop = 0
    scrollCalls = 0
    widthAttrMutations = 0
    viewBox = { baseVal: { x: 0, y: 0, width: 1600, height: 1200 } }
    width = { baseVal: { value: 1600 } }
    private _src = ''

    constructor(tagName: string, className = '') {
      this.tagName = tagName
      this.className = className
      const store: Record<string, string> = {}
      this.style = Object.assign(store, {
        setProperty: (k: string, v: string) => {
          store[k] = v
        },
        getPropertyValue: (k: string) => store[k] ?? '',
        removeProperty: (k: string) => {
          delete store[k]
        },
      })
    }
    get childNodes() {
      return this.children
    }
    get src() {
      return this._src
    }
    set src(v: string) {
      this._src = v
      this.attributes.set('src', v)
    }
    setAttribute(k: string, v: string) {
      if (k === 'width' || k === 'height') this.widthAttrMutations++
      this.attributes.set(k, v)
    }
    getAttribute(k: string) {
      return this.attributes.get(k) ?? null
    }
    removeAttribute(k: string) {
      this.attributes.delete(k)
      if (k === 'src') this._src = ''
    }
    append(...nodes: FakeVaultNode[]) {
      for (const n of nodes) {
        n.parentElement = this
        this.children.push(n)
      }
    }
    replaceChildren(...nodes: FakeVaultNode[]) {
      for (const c of this.children) c.parentElement = null
      this.children = []
      this.append(...nodes)
    }
    remove() {
      if (!this.parentElement) return
      const idx = this.parentElement.children.indexOf(this)
      if (idx >= 0) this.parentElement.children.splice(idx, 1)
      this.parentElement = null
    }
    cloneNode(): FakeVaultNode {
      const copy = new FakeVaultNode(this.tagName, this.className)
      copy.dataset = { ...this.dataset }
      for (const [k, v] of this.attributes) copy.attributes.set(k, v)
      for (const c of this.children) copy.append(c.cloneNode())
      return copy
    }
    matchesSelector(sel: string): boolean {
      if (sel.startsWith(':scope > ')) return this.matchesSelector(sel.slice(9))
      if (sel === 'svg.vault-scene')
        return this.tagName === 'svg' && this.className.includes('vault-scene')
      if (sel === 'img') return this.tagName === 'img'
      if (sel === 'details') return this.tagName === 'details'
      if (sel.startsWith('[data-camera="')) {
        const val = sel.slice(14, -2)
        return this.dataset.camera === val
      }
      if (sel === '[data-camera]') return this.dataset.camera !== undefined
      if (sel.startsWith('[data-card="')) {
        const val = sel.slice(12, -2)
        return this.dataset.card === val
      }
      if (sel === '[data-card]') return this.dataset.card !== undefined
      if (sel === '[data-vault-readout]') return this.dataset.vaultReadout !== undefined
      if (sel === '[data-vault-section]') return this.dataset.vaultSection !== undefined
      if (sel.startsWith('.')) {
        const cls = sel.slice(1)
        return this.className.split(/\s+/).includes(cls)
      }
      return false
    }
    querySelector(sel: string): FakeVaultNode | null {
      return this.querySelectorAll(sel)[0] ?? null
    }
    querySelectorAll(sel: string): FakeVaultNode[] {
      const out: FakeVaultNode[] = []
      const directOnly = sel.startsWith(':scope > ')
      const targetSel = directOnly ? sel.slice(9) : sel
      const visit = (node: FakeVaultNode, depth: number) => {
        for (const child of node.children) {
          if (child.matchesSelector(targetSel)) out.push(child)
          if (!directOnly && depth < 10) visit(child, depth + 1)
        }
      }
      visit(this, 0)
      return out
    }
    closest(sel: string): FakeVaultNode | null {
      if (this.matchesSelector(sel)) return this
      return this.parentElement?.closest(sel) ?? null
    }
    addEventListener(type: string, cb: (e: Record<string, unknown>) => void) {
      const list = this.listeners.get(type) ?? []
      list.push(cb)
      this.listeners.set(type, list)
    }
    removeEventListener() {}
    dispatch(type: string, extra: Record<string, unknown> = {}) {
      const event: Record<string, unknown> = {
        type,
        target: extra.target ?? this,
        preventDefault: () => {},
        ...extra,
      }
      for (const cb of this.listeners.get(type) ?? []) cb(event)
    }
    scrollTo(opts: { left?: number; top?: number }) {
      this.scrollCalls++
      if (typeof opts.left === 'number') this.scrollLeft = opts.left
      if (typeof opts.top === 'number') this.scrollTop = opts.top
    }
    getBoundingClientRect() {
      return { left: 0, top: 0, right: 400, bottom: 300, width: 400, height: 300 }
    }
    showModal() {
      this.open = true
    }
    close() {
      if (!this.open) return
      this.open = false
      this.dispatch('close')
    }
    focus() {}
    setPointerCapture() {}
    hasPointerCapture() {
      return false
    }
    releasePointerCapture() {}
    animate() {
      const anim = {
        onfinish: null as (() => void) | null,
        cancel: () => {
          anim.onfinish = null
        },
        finish: () => {
          const cb = anim.onfinish
          anim.onfinish = null
          cb?.()
        },
      }
      if (this.tagName === 'img') posterAnims.push(anim)
      return anim
    }
  }

  const prevWindow = globalThis.window
  const prevDocument = globalThis.document
  const prevImage = globalThis.Image
  const prevRO = globalThis.ResizeObserver
  const prevRAF = globalThis.requestAnimationFrame
  const prevCAF = globalThis.cancelAnimationFrame
  const prevElement = globalThis.Element
  const prevHTMLElement = globalThis.HTMLElement

  try {
    globalThis.Element = FakeVaultNode as unknown as typeof Element
    globalThis.HTMLElement = FakeVaultNode as unknown as typeof HTMLElement
    globalThis.ResizeObserver = class {
      observe() {}
      disconnect() {}
    } as unknown as typeof ResizeObserver
    globalThis.Image = class {
      decoding = 'async'
      onload: (() => void) | null = null
      onerror: (() => void) | null = null
      private _record = { src: '', removedSrc: false }
      constructor() {
        createdImages.push(this._record)
      }
      get src() {
        return this._record.src
      }
      set src(v: string) {
        this._record.src = v
      }
      removeAttribute(k: string) {
        if (k === 'src') {
          this._record.removedSrc = true
          this._record.src = ''
        }
      }
      decode() {
        if (this._record.src.includes('-full.jpg')) {
          return new Promise<void>(resolve => {
            resolveFullDecode = () => {
              this.onload?.()
              resolve()
            }
          })
        }
        return Promise.resolve()
      }
    } as unknown as typeof Image
    globalThis.requestAnimationFrame = (cb: FrameRequestCallback) => {
      cb(16)
      return 1
    }
    globalThis.cancelAnimationFrame = () => {}
    const controller = new AbortController()
    globalThis.window = {
      quartzNavLifecycle: { listening: true, controller },
      matchMedia: () => ({ matches: false, addEventListener: () => {} }),
      setTimeout: globalThis.setTimeout.bind(globalThis),
      clearTimeout: globalThis.clearTimeout.bind(globalThis),
      addCleanup: () => {},
    } as unknown as Window & typeof globalThis
    globalThis.document = {
      activeElement: null,
      addEventListener: () => {},
      removeEventListener: () => {},
      querySelectorAll: () => [],
      createElement: (tag: string) => new FakeVaultNode(tag),
      createElementNS: (_ns: string, tag: string) => new FakeVaultNode(tag),
    } as unknown as Document

    const mod = (await import('../../components/scripts/base-vault.inline')) as {
      initializeVault?: (root: HTMLElement) => void
    }
    assert.equal(typeof mod.initializeVault, 'function')

    const root = new FakeVaultNode('div', 'vault')
    const stage = new FakeVaultNode('div', 'vault-stage')
    const viewport = new FakeVaultNode('div', 'vault-viewport')
    const svg = new FakeVaultNode('svg', 'vault-scene')
    svg.dataset.start = '100 100'
    const lift = new FakeVaultNode('g', 'vault-lift')
    for (let i = 0; i < 5; i++) {
      const can = new FakeVaultNode('g', 'cn-can')
      can.dataset.index = String(i)
      can.dataset.cx = String(100 + i * 50)
      can.dataset.cy = String(200)
      can.style.setProperty('--px', '-24')
      can.style.setProperty('--py', '14')
      const lid = new FakeVaultNode('circle', 'cn-lid')
      lid.setAttribute('transform', 'translate(0,0)')
      lid.setAttribute('r', '80')
      can.append(lid)
      svg.append(can)
    }
    svg.append(lift)
    viewport.append(svg)

    const zoomIn = new FakeVaultNode('button')
    zoomIn.dataset.camera = 'in'
    const zoomOut = new FakeVaultNode('button')
    zoomOut.dataset.camera = 'out'
    const readout = new FakeVaultNode('span', 'vault-corner')
    readout.dataset.vaultReadout = ''
    const section = new FakeVaultNode('span', 'vault-corner')
    section.dataset.vaultSection = ''
    stage.append(viewport, zoomIn, zoomOut, readout, section)

    const indexPane = new FakeVaultNode('div', 'vault-index')
    const groupDetails = new FakeVaultNode('details', 'vault-index-group')
    const groupLabel = new FakeVaultNode('span', 'vault-index-label')
    groupLabel.textContent = '1950s'
    groupDetails.append(groupLabel)
    for (let i = 0; i < 5; i++) {
      const row = new FakeVaultNode('li', 'vault-entry')
      row.dataset.index = String(i)
      row.dataset.href = `./movies/film-${i}`
      row.dataset.cover = `http://localhost:8080/static/bookcase/film-${i}-w320.webp`
      row.dataset.fullCover = `http://localhost:8080/movies/posters/film-${i}-full.jpg`
      const btn = new FakeVaultNode('button', 'vault-entry-select')
      const title = new FakeVaultNode('span', 'vault-entry-title')
      title.textContent = `Film ${i}`
      btn.append(title)
      const detail = new FakeVaultNode('div', 'vault-entry-detail')
      row.append(btn, detail)
      groupDetails.append(row)
    }
    indexPane.append(groupDetails)

    const dialog = new FakeVaultNode('dialog', 'vault-card')
    const poster = new FakeVaultNode('figure', 'vault-card-poster')
    const posterImg = new FakeVaultNode('img')
    poster.append(posterImg)
    const cardTitle = new FakeVaultNode('a', 'vault-card-title')
    const cardDetail = new FakeVaultNode('div', 'vault-card-detail')
    const cardOpen = new FakeVaultNode('a', 'vault-card-open')
    const cardPos = new FakeVaultNode('span', 'vault-card-position')
    const prevBtn = new FakeVaultNode('button')
    prevBtn.dataset.card = 'previous'
    const nextBtn = new FakeVaultNode('button')
    nextBtn.dataset.card = 'next'
    const closeBtn = new FakeVaultNode('button')
    closeBtn.dataset.card = 'close'
    dialog.append(poster, cardTitle, cardDetail, cardOpen, cardPos, prevBtn, nextBtn, closeBtn)

    root.append(stage, indexPane, dialog)
    mod.initializeVault!(root as unknown as HTMLElement)

    // 1. Compositor zoom: clicking zoom-in 3 times must NOT mutate svg width/height synchronously on every click,
    // and MUST update scrollLeft/scrollTop immediately so the viewport center stays anchored.
    svg.widthAttrMutations = 0
    const scrollLeftBeforeZoom = viewport.scrollLeft
    root.dispatch('click', { target: zoomIn })
    root.dispatch('click', { target: zoomIn })
    root.dispatch('click', { target: zoomIn })
    assert.equal(
      svg.widthAttrMutations,
      0,
      'interactive zoom steps must scale via compositor transform rather than synchronous SVG width/height relayout',
    )
    // The interim zoom rides on the scaler wrapper; the drawing's own style stays untouched.
    const scaler = viewport.querySelector('.vault-scaler')
    assert.ok(scaler, 'the drawing is wrapped in a scaler')
    assert.match(scaler.style.transform ?? '', /scale\(/)
    assert.equal(svg.style.transform ?? '', '')
    assert.notEqual(
      viewport.scrollLeft,
      scrollLeftBeforeZoom,
      'interactive zoom must update viewport scroll position to keep the anchor point stationary',
    )

    // 1b. Directional drag-to-pan vs can-pull disambiguation:
    // Pressing on an unpulled can (cans[0], px=-24, py=14) and dragging in the push-in direction (+48, -28)
    // must pan the viewport rather than locking into a dead pull gesture clamped at t=0.
    const allCans = svg.querySelectorAll(':scope > .cn-can')
    const scrollBeforeReverseDrag = viewport.scrollLeft
    viewport.dispatch('pointerdown', {
      button: 0,
      isPrimary: true,
      pointerId: 1,
      clientX: 200,
      clientY: 200,
      target: allCans[0],
    })
    viewport.dispatch('pointermove', {
      pointerId: 1,
      buttons: 1,
      clientX: 248,
      clientY: 172,
      target: allCans[0],
    })
    viewport.dispatch('pointerup', { pointerId: 1, clientX: 248, clientY: 172, target: allCans[0] })
    assert.equal(
      viewport.scrollLeft,
      scrollBeforeReverseDrag - 48,
      'dragging against the pull vector on an unpulled can must pan the viewport instead of freezing',
    )
    assert.equal(readout.textContent, '– / 5', 'panning across an unpulled can must not select it')

    // Pressing on cans[1] and dragging along its pull-out vector (-24, +14) pulls the can out and selects it.
    viewport.dispatch('pointerdown', {
      button: 0,
      isPrimary: true,
      pointerId: 2,
      clientX: 200,
      clientY: 200,
      target: allCans[1],
    })
    viewport.dispatch('pointermove', {
      pointerId: 2,
      buttons: 1,
      clientX: 176,
      clientY: 214,
      target: allCans[1],
    })
    viewport.dispatch('pointerup', { pointerId: 2, clientX: 176, clientY: 214, target: allCans[1] })
    assert.equal(
      readout.textContent,
      '2 / 5',
      'dragging along the pull-out vector on an unpulled can must pull it out and select it',
    )

    // Dragging in the pull-out direction (-48, +28) on the ALREADY pulled-out cans[1] must pan the viewport
    // rather than locking into a dead pull gesture clamped at t=1.
    const scrollBeforeOutDrag = viewport.scrollLeft
    viewport.dispatch('pointerdown', {
      button: 0,
      isPrimary: true,
      pointerId: 3,
      clientX: 200,
      clientY: 200,
      target: allCans[1],
    })
    viewport.dispatch('pointermove', {
      pointerId: 3,
      buttons: 1,
      clientX: 152,
      clientY: 228,
      target: allCans[1],
    })
    viewport.dispatch('pointerup', { pointerId: 3, clientX: 152, clientY: 228, target: allCans[1] })
    assert.equal(
      viewport.scrollLeft,
      scrollBeforeOutDrag + 48,
      'dragging in the pull-out direction on an already pulled can must pan the viewport instead of freezing at t=1',
    )

    // Dragging in the push-in direction (+24, -14) on the pulled-out cans[1] slides it back in and releases selection.
    viewport.dispatch('pointerdown', {
      button: 0,
      isPrimary: true,
      pointerId: 4,
      clientX: 200,
      clientY: 200,
      target: allCans[1],
    })
    viewport.dispatch('pointermove', {
      pointerId: 4,
      buttons: 1,
      clientX: 224,
      clientY: 186,
      target: allCans[1],
    })
    viewport.dispatch('pointerup', { pointerId: 4, clientX: 224, clientY: 186, target: allCans[1] })
    assert.equal(
      readout.textContent,
      '– / 5',
      'dragging back along the push-in vector on a pulled can must return it to its slot and clear selection',
    )

    // 2. Rapid can stepping: pull starts immediately and bounds settling clones in .vault-lift to <= 3 (1 active + <= 2 settling)
    for (let i = 0; i < 5; i++) {
      viewport.dispatch('keydown', { key: 'ArrowRight' })
      assert.ok(
        lift.children.length <= 3,
        `expected at most 3 clones in .vault-lift during rapid stepping, got ${lift.children.length}`,
      )
    }

    // 3. Modal browsing: opening card and stepping Next 3 times must NOT mutate .vault-lift or scroll viewport while open
    const cans = svg.querySelectorAll(':scope > .cn-can')
    createdImages.length = 0
    posterAnims.length = 0
    viewport.dispatch('click', { target: cans[0] })
    assert.equal(dialog.open, true)
    assert.equal(posterAnims.length, 1, 'entrance animation runs on modal open')

    const liftChildrenAtOpen = [...lift.children]
    const scrollCallsAtOpen = viewport.scrollCalls
    root.dispatch('click', { target: nextBtn })
    root.dispatch('click', { target: nextBtn })
    root.dispatch('click', { target: nextBtn })

    assert.equal(
      posterAnims.length,
      1,
      'stepping Next inside open modal must not re-run the entrance transform animation',
    )
    assert.deepEqual(
      lift.children,
      liftChildrenAtOpen,
      'stepping inside open modal must not clone or mutate background rack cans',
    )
    assert.equal(
      viewport.scrollCalls,
      scrollCallsAtOpen,
      'stepping inside open modal must not scroll the hidden background viewport',
    )

    // Full-res poster decode is debounced (300ms): before 300ms, zero full-res decodes have started
    const fullResBeforeDebounce = createdImages.filter(img => img.src.includes('-full.jpg'))
    assert.equal(
      fullResBeforeDebounce.length,
      0,
      'full-res poster decode must be debounced during rapid modal stepping',
    )

    // Wait 320ms for debounced full-res decode of film-3 to start
    await new Promise(resolve => setTimeout(resolve, 320))
    const fullResAfterDebounce = createdImages.filter(img => img.src.includes('-full.jpg'))
    assert.equal(
      fullResAfterDebounce.length,
      1,
      'only the settled card index (film-3) should start a full-res poster decode',
    )

    // Resolve full-res decode while entrance animation is still in-flight: posterImg.src stays on 320px thumb until animation finishes
    resolveFullDecode?.()
    await Promise.resolve()
    await Promise.resolve()
    assert.equal(
      posterImg.src,
      'http://localhost:8080/static/bookcase/film-3-w320.webp',
      'full-res poster swap must wait until entrance animation finishes',
    )
    posterAnims[0].finish()
    assert.equal(
      posterImg.src,
      'http://localhost:8080/movies/posters/film-3-full.jpg',
      'finishing entrance animation flushes the decoded full-res poster',
    )

    // Closing modal commits selection (index 3) to the rack and clears posterImg src
    root.dispatch('click', { target: closeBtn })
    assert.equal(dialog.open, false)
    assert.equal(posterImg.getAttribute('src'), null, 'closing modal must clear posterImg src')
    assert.equal(readout.textContent, '4 / 5', 'closing modal commits final cardIndex to the rack')
    controller.abort()
  } finally {
    globalThis.window = prevWindow
    globalThis.document = prevDocument
    globalThis.Image = prevImage
    globalThis.ResizeObserver = prevRO
    globalThis.requestAnimationFrame = prevRAF
    globalThis.cancelAnimationFrame = prevCAF
    globalThis.Element = prevElement
    globalThis.HTMLElement = prevHTMLElement
  }
})
