"""The antilibrary bookcase: camera, preview and search, one scenario each.

Synthetic wheel events never scroll natively, so a wheel pan dispatches the event (the page's
wheel handler runs) and then writes the scroll offset the browser would have written (the page's
scroll handler and camera paint run). Hover lifts need `buttons: 0` and wait out the page's 500 ms
wheel settle, so they follow a quiet gap after the pans."""

from .prelude import HELPERS

READY = r"""document.querySelector('.bookcase[data-bookcase-ready] .bookcase-viewport .bookcase-world') && document.querySelectorAll('.bc-spine-art').length > 10 && document.querySelectorAll('.bookcase-entry-link').length > 0"""

# Shared setup: the stage, its centre, the books fully inside the stage, resident spine art.
SETUP = r"""
    const root = document.querySelector('.bookcase')
    const viewport = root.querySelector('.bookcase-viewport')
    const scene = root.querySelector('svg.bookcase-scene')
    const world = root.querySelector('.bookcase-world')
    const r = viewport.getBoundingClientRect()
    const cx = r.left + r.width / 2, cy = r.top + r.height / 2
    const inView = () => { const vr = viewport.getBoundingClientRect(); return [...root.querySelectorAll('svg.bookcase-scene .bc-book')].filter(b => { const q = b.getBoundingClientRect(); return q.width > 4 && q.height > 24 && q.left > vr.left && q.right < vr.right && q.top > vr.top && q.bottom < vr.bottom }) }
    const resident = () => new Set([...scene.querySelectorAll('.bc-spine-art, .bc-cover')].map(i => `${i.closest('.bc-book')?.dataset.bookIndex}:${i.classList[0]}`))
    const spineLoads = () => performance.getEntriesByType('resource').filter(e => e.name.includes('/static/bookcase-spines/') || e.name.includes('/static/bookcase/')).length
    const shown = () => [...root.querySelectorAll('.bookcase-entry')].filter(e => e.style.display !== 'none' && !e.hasAttribute('data-topic-hidden') && !e.closest('.bookcase-index-group[hidden], .bookcase-index-group[data-topic-hidden]')).length
    const placed = () => scene.querySelectorAll('.bc-book').length
    const selectedTopic = () => root.querySelector('.bookcase-topic[aria-selected="true"]')?.dataset.topic ?? null
    const pan = async (dx, dy, n) => { for (let i = 0; i < n; i++) { sync(() => { wheelXY(viewport, cx, cy, dx, dy); viewport.scrollLeft += dx; viewport.scrollTop += dy }); await raf() } }
    const press = async (el, x, y) => { pointer(el, 'pointerdown', x, y); await raf(); pointer(el, 'pointerup', x, y) }
    const centre = el => { const q = el.getBoundingClientRect(); return [q.left + q.width / 2, q.top + q.height / 2] }
"""

bookcase = {
  'path': '/antilibrary',
  'ready': READY,
  'settle': 1500,
  'script': HELPERS
  + SETUP
  + r"""
    const scroll0 = [viewport.scrollLeft, viewport.scrollTop]
    await step('wheel-pan-y', () => pan(0, 90, 20))
    await step('wheel-pan-x', () => pan(90, 0, 20))
    check('wheel-pan', viewport.scrollLeft !== scroll0[0] && viewport.scrollTop !== scroll0[1])
    await sleep(600)
    const drag0 = viewport.scrollLeft
    await step('drag-pan', async () => {
      pointer(viewport, 'pointerdown', cx, cy)
      for (let i = 1; i <= 24; i++) { sync(() => pointer(viewport, 'pointermove', cx + i * 12, cy - i * 4)); await raf() }
      pointer(viewport, 'pointerup', cx + 288, cy - 96, { buttons: 0 })
    })
    check('drag-pan', Math.abs(viewport.scrollLeft - drag0) > 100)
    const width0 = world.style.width
    await step('zoom-in-wheel', async () => { for (let i = 0; i < 12; i++) { sync(() => wheel(viewport, cx, cy, -40, true)); await sleep(30) } await sleep(250) })
    check('zoom-baked', world.style.width !== width0)
    await step('zoom-out-wheel', async () => { for (let i = 0; i < 12; i++) { sync(() => wheel(viewport, cx, cy, 40, true)); await sleep(30) } await sleep(250) })
    await step('zoom-keys', async () => { for (const k of ['+', '+', '-', '-']) { sync(() => key(viewport, k)); await sleep(160) } })
    const fitBtn = root.querySelector('[data-camera="fit"]'), inBtn = root.querySelector('[data-camera="in"]'), outBtn = root.querySelector('[data-camera="out"]')
    await step('fit', async () => { sync(() => fitBtn.click()); await sleep(200) })
    // Spine-art residency: zoom in on the controls, sweep the whole width, and the resident set turns over.
    await step('zoom-buttons', async () => { for (let i = 0; i < 4; i++) { sync(() => inBtn.click()); await sleep(160) } })
    const before = resident(), loads0 = spineLoads()
    const sweep = Math.max(0, viewport.scrollWidth - viewport.clientWidth - viewport.scrollLeft)
    await step('residency-sweep', async () => { await pan(Math.max(40, sweep / 30), 0, 30); await sleep(600) })
    const after = resident()
    const turned = [...after].filter(k => !before.has(k)).length + [...before].filter(k => !after.has(k)).length
    note('residency', { before: before.size, after: after.size, turned, loads: spineLoads() - loads0 })
    check('residency', turned > 8 && after.size > 0)
    await step('zoom-out-buttons', async () => { for (let i = 0; i < 4; i++) { sync(() => outBtn.click()); await sleep(160) } sync(() => fitBtn.click()); await sleep(200) })
    await sleep(600)
    // Hover lifts a copy out of the shelf and reveals its sidebar entry.
    const books = inView().slice(0, 10)
    const readout = root.querySelector('[data-bookcase-readout]')
    await step('hover-books', async () => { for (const b of books) { const [x, y] = centre(b); sync(() => pointer(b, 'pointermove', x, y, { buttons: 0 })); await sleep(110) } await sleep(300) })
    check('hover-lift', books.length >= 6 && !!root.querySelector('.bc-lift') && !readout?.textContent.startsWith('–'))
    await step('leave', async () => { sync(() => root.dispatchEvent(new PointerEvent('pointerleave', { pointerType: 'mouse' }))); await sleep(250) })
    const topics = [...root.querySelectorAll('.bookcase-topic')]
    const clicked = []
    await step('topics', async () => { for (const t of topics.slice(1, 5)) { sync(() => t.click()); clicked.push(selectedTopic()); await sleep(150) } })
    check('topics', clicked.join() === topics.slice(1, 5).map(t => t.dataset.topic).join())
    // Category initials from the keyboard: the first letter with several matching topics, pressed
    // once per match, cycles them; then the "all" initial.
    const counts = {}
    for (const t of topics.slice(1)) counts[t.dataset.shortcut] = (counts[t.dataset.shortcut] || 0) + 1
    const letter = Object.keys(counts).sort((a, b) => counts[b] - counts[a])[0]
    const cycled = []
    await step('topic-initials', async () => {
      for (let i = 0; i < Math.min(3, counts[letter]); i++) { const target = root.contains(document.activeElement) ? document.activeElement : viewport; sync(() => key(target, letter)); cycled.push(selectedTopic()); await sleep(150) }
      const target = root.contains(document.activeElement) ? document.activeElement : viewport
      sync(() => key(target, topics[0].dataset.shortcut)); await sleep(150)
    })
    note('initials', { letter, cycled })
    check('topic-initials', cycled.length >= 2 && new Set(cycled).size === cycled.length && cycled.every(t => topics.find(b => b.dataset.topic === t)?.dataset.shortcut === letter) && selectedTopic() === topics[0].dataset.topic)
    const index = root.querySelector('.bookcase-index')
    await step('index-scroll', () => scrollBy(index, 160, 20))
    await step('index-scroll-up', () => scrollBy(index, -160, 20))
    const links = [...root.querySelectorAll('.bookcase-entry-link')].filter(a => { const q = a.getBoundingClientRect(); return q.height > 0 && q.top > 0 && q.bottom < innerHeight }).slice(0, 12)
    await step('hover-entries', () => hoverEach(links))
    check('groups', root.querySelectorAll('.bookcase-index-group').length > 3)
    return __bp.result()
    """,
}

bookcase_preview = {
  'path': '/antilibrary',
  'ready': READY,
  'settle': 1500,
  'script': HELPERS
  + SETUP
  + r"""
    const dialog = root.querySelector('.bookcase-preview')
    const position = () => dialog.querySelector('.bookcase-preview-position')?.textContent ?? ''
    const book = inView()[0]
    // A still press on a book opens the preview (an index link would navigate instead).
    await step('open-preview', async () => { if (book) { const [x, y] = centre(book); await press(book, x, y) } await sleep(600) })
    check('preview-open', dialog.open)
    const p0 = position()
    const next = dialog.querySelector('[data-preview="next"]'), previous = dialog.querySelector('[data-preview="previous"]')
    await step('preview-next', async () => { for (let i = 0; i < 4; i++) { sync(() => next.click()); await sleep(180) } })
    const p1 = position()
    await step('preview-previous', async () => { for (let i = 0; i < 2; i++) { sync(() => previous.click()); await sleep(180) } })
    await step('preview-keys', async () => { for (const k of ['ArrowRight', 'ArrowRight', 'ArrowRight', 'ArrowLeft']) { sync(() => key(dialog, k)); await sleep(180) } })
    const dr = dialog.getBoundingClientRect()
    await step('preview-wheel', async () => { for (let i = 0; i < 3; i++) { sync(() => wheelXY(dialog, dr.left + 20, dr.top + 20, 0, 120)); await sleep(260) } })
    const p2 = position()
    note('preview', { p0, p1, p2 })
    check('preview-browse', parseInt(p1) === parseInt(p0) + 4 && parseInt(p2) === parseInt(p0) + 7)
    await step('close-preview', async () => { sync(() => dialog.querySelector('[data-preview="close"]').click()); await sleep(500) })
    check('preview-closed', !dialog.open && !!root.querySelector('.bc-lift'))
    await step('open-preview-key', async () => { sync(() => key(viewport, 'Enter')); await sleep(600) })
    check('preview-key-open', dialog.open)
    await step('close-preview-escape', async () => { escape(); await sleep(500) })
    check('preview-escape-closed', !dialog.open)
    return __bp.result()
    """,
}

bookcase_search = {
  'path': '/antilibrary',
  'ready': READY,
  'settle': 1500,
  'script': HELPERS
  + SETUP
  + r"""
    const input = document.querySelector('#base-search-input')
    const restore = lexical()
    const total = placed()
    // Every keystroke repacks the shelf from the matching sidebar entries.
    await step('search-type', () => typeInto(input, 'the', 110))
    await sleep(200)
    const narrowed = [shown(), placed()]
    await step('search-refine', () => typeInto(input, ' art', 110))
    await sleep(200)
    const refined = [shown(), placed()]
    await step('search-clear', async () => { clearInput(input); await sleep(300) })
    const cleared = placed()
    await step('search-miss', async () => { await typeInto(input, 'zzqx', 90); await sleep(200) })
    const empty = root.hasAttribute('data-empty')
    await step('search-clear-miss', async () => { clearInput(input); await sleep(300) })
    restore()
    note('search', { total, narrowed, refined, cleared, empty })
    check('search-repack', narrowed[0] > 0 && narrowed[0] < total && narrowed[1] === narrowed[0] && refined[1] === refined[0] && refined[0] <= narrowed[0])
    check('search-restore', cleared === total && placed() === total)
    check('search-empty', empty)
    return __bp.result()
    """,
}
