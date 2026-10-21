"""Base views: vault, bookcase, cards, table, kanban, mapbox map."""

from .prelude import HELPERS

vault = {
    'path': '/cinematheque',
    'ready': r"""document.querySelector('[data-vault-ready] .vault-viewport .vault-scaler > svg.vault-scene') && document.querySelectorAll('.vault-entry-select').length > 0""",
    'settle': 1500,
    'script': HELPERS + r"""
    const viewport = document.querySelector('.vault-viewport')
    const scaler = viewport.querySelector('.vault-scaler')
    const svg = scaler.querySelector('svg')
    const r = viewport.getBoundingClientRect()
    const cx = r.left + r.width / 2, cy = r.top + r.height / 2
    await step('scroll-down', () => scrollBy(viewport, 240, 20))
    await step('scroll-up', () => scrollBy(viewport, -240, 20))
    const transform0 = scaler.style.transform
    await step('zoom-in', async () => { for (let i = 0; i < 12; i++) { sync(() => wheel(viewport, cx, cy, -60, true)); await sleep(40) } })
    await sleep(400)
    note('zoom', { transform0, transform1: scaler.style.transform, width: svg.getAttribute('width') })
    check('zoom-applied', scaler.style.transform !== transform0)
    await step('zoom-out', async () => { for (let i = 0; i < 12; i++) { sync(() => wheel(viewport, cx, cy, 60, true)); await sleep(40) } })
    await sleep(400)
    const rows = [...document.querySelectorAll('.vault-entry-select')].slice(0, 12)
    await step('hover-rows', () => hoverEach(rows))
    await step('open-card', async () => { rows[3].click(); await sleep(250) })
    check('card-open', !!document.querySelector('.vault-card[open], .vault-card.is-open, .vault-card:not([hidden])'))
    await step('close-card', async () => { escape(); await sleep(250) })
    check('scene-nodes', svg.querySelectorAll('*').length > 1000)
    return __bp.result()
    """,
}

bookcase = {
    'path': '/antilibrary',
    'ready': r"""document.querySelector('.bookcase-viewport .bookcase-world') && document.querySelectorAll('.bc-cover').length > 10 && document.querySelectorAll('.bookcase-entry-link').length > 0""",
    'settle': 1500,
    'script': HELPERS + r"""
    const viewport = document.querySelector('.bookcase-viewport')
    const r = viewport.getBoundingClientRect()
    const cx = r.left + r.width / 2, cy = r.top + r.height / 2
    await step('wheel-pan', async () => { for (let i = 0; i < 20; i++) { sync(() => wheel(viewport, cx, cy, 120, false)); await raf() } })
    await step('drag-pan', async () => {
      pointer(viewport, 'pointerdown', cx, cy)
      for (let i = 1; i <= 24; i++) { sync(() => pointer(viewport, 'pointermove', cx - i * 12, cy + i * 4)); await raf() }
      pointer(viewport, 'pointerup', cx - 288, cy + 96)
    })
    const topics = [...document.querySelectorAll('.bookcase-topic')]
    await step('topics', async () => { for (const t of topics.slice(0, 4)) { sync(() => t.click()); await sleep(120) } })
    const list = document.querySelector('.bookcase-index-list, .bookcase-index')
    await step('index-scroll', () => scrollBy(list, 160, 20))
    const links = [...document.querySelectorAll('.bookcase-entry-link')].slice(0, 12)
    await step('hover-entries', () => hoverEach(links))
    // A still press on a book opens the preview (the index link would navigate instead).
    const vr = viewport.getBoundingClientRect()
    const book = [...document.querySelectorAll('.bc-book')].find(b => { const q = b.getBoundingClientRect(); return q.width > 4 && q.left > vr.left && q.right < vr.right && q.top > vr.top && q.bottom < vr.bottom })
    await step('open-preview', async () => { if (book) { const q = book.getBoundingClientRect(); const x = q.left + q.width / 2, y = q.top + q.height / 2; pointer(book, 'pointerdown', x, y); await raf(); pointer(book, 'pointerup', x, y) } await sleep(400) })
    note('preview', { book: !!book, open: document.querySelector('.bookcase-preview')?.open ?? null })
    check('preview-open', !!document.querySelector('.bookcase-preview[open]'))
    await step('close-preview', async () => { const d = document.querySelector('.bookcase-preview[open]'); if (d?.close) d.close(); else escape(); await sleep(250) })
    check('groups', document.querySelectorAll('.bookcase-index-group').length > 3)
    return __bp.result()
    """,
}


def cards(path):
    return {
        'path': path,
        'ready': r"""document.querySelector('.base-view') && document.querySelectorAll('.base-view a[href]').length > 10""",
        'settle': 800,
        'script': HELPERS + r"""
    await step('scroll-down', () => pageScroll(300, 20))
    await step('scroll-up', () => pageScroll(-300, 20))
    const links = [...document.querySelectorAll('.base-view a[href]')].slice(0, 16)
    await step('hover-cards', () => hoverEach(links))
    check('cards', links.length === 16)
    return __bp.result()
    """,
    }


def table(path):
    return {
        'path': path,
        'ready': r"""document.querySelector('.base-table') && document.querySelectorAll('.base-table-cell-content').length > 10""",
        'settle': 800,
        'script': HELPERS + r"""
    await step('scroll-down', () => pageScroll(300, 20))
    const heads = [...document.querySelectorAll('.base-table th, .base-table thead button')].slice(0, 3)
    await step('sort-clicks', async () => { for (const h of heads) { sync(() => h.click()); await sleep(80); sync(() => h.click()); await sleep(80) } })
    const toggles = [...document.querySelectorAll('[data-base-query-toggle]')]
    await step('query-panel', async () => { for (const t of toggles.slice(0, 2)) { sync(() => t.click()); await sleep(200) } })
    const rows = [...document.querySelectorAll('.base-table tbody tr')].slice(0, 16)
    await step('hover-rows', () => hoverEach(rows))
    check('rows', rows.length > 8)
    return __bp.result()
    """,
    }


kanban = {
    'path': '/cinematheque/analysis',
    'ready': r"""document.querySelector('.base-view') && document.querySelectorAll('.base-view a[href]').length > 5""",
    'settle': 800,
    'script': HELPERS + r"""
    const board = document.querySelector('.base-kanban, .base-view')
    await step('scroll-x', async () => { for (let i = 0; i < 12; i++) { sync(() => { board.scrollLeft += 200 }); await raf() } })
    await step('scroll-y', () => pageScroll(300, 12))
    const links = [...document.querySelectorAll('.base-view a[href]')].slice(0, 16)
    await step('hover-cards', () => hoverEach(links))
    check('cards', links.length > 5)
    return __bp.result()
    """,
}


def mapbox(path, ready, settle=2500):
    # Pan and zoom on a mapbox canvas; the pointer path goes through the canvas's own listeners.
    return {
        'path': path,
        'ready': ready,
        'settle': settle,
        'script': HELPERS + r"""
    const el = document.querySelector('.mapboxgl-canvas, .base-map canvas, .tri-map canvas')
    const r = el.getBoundingClientRect()
    const cx = r.left + r.width / 2, cy = r.top + r.height / 2
    await step('zoom-in', async () => { for (let i = 0; i < 8; i++) { sync(() => wheel(el, cx, cy, -100, false)); await sleep(60) } })
    await step('drag', async () => { pointer(el, 'pointerdown', cx, cy); for (let i = 1; i <= 16; i++) { sync(() => pointer(el, 'pointermove', cx + i * 10, cy)); await sleep(16) } pointer(el, 'pointerup', cx + 160, cy) })
    await step('zoom-out', async () => { for (let i = 0; i < 8; i++) { sync(() => wheel(el, cx, cy, 100, false)); await sleep(60) } })
    await sleep(500)
    check('canvas', el.width > 0)
    return __bp.result()
    """,
    }


places_map = mapbox('/places-to-go', r"""document.querySelector('.base-map canvas.mapboxgl-canvas')""", 4000)
