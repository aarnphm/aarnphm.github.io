"""Base views: vault, cards, table, kanban, mapbox map (the bookcase lives in bookcase.py)."""

from .prelude import HELPERS

vault = {
  'path': '/cinematheque',
  'ready': r"""document.querySelector('[data-vault-ready] .vault-viewport .vault-scaler > svg.vault-scene') && document.querySelectorAll('.vault-entry-select').length > 0""",
  'settle': 1500,
  'script': HELPERS
  + r"""
    const root = document.querySelector('[data-vault-ready]')
    const viewport = root.querySelector('.vault-viewport')
    const scaler = viewport.querySelector('.vault-scaler')
    const svg = scaler.querySelector('svg')
    const dialog = root.querySelector('dialog.vault-card')
    const r = viewport.getBoundingClientRect()
    const cx = r.left + r.width / 2, cy = r.top + r.height / 2
    const current = () => root.querySelector('.vault-entry[data-current]')?.dataset.index ?? null
    const cardTitle = () => dialog.querySelector('.vault-card-position')?.textContent ?? ''
    await step('scroll-down', () => scrollBy(viewport, 240, 20))
    await step('scroll-up', () => scrollBy(viewport, -240, 20))
    // A sideways wheel follows the rack's slant with a smooth scroll of the page's own.
    const left0 = viewport.scrollLeft
    await step('pan-x-wheel', async () => { for (let i = 0; i < 6; i++) { sync(() => wheelXY(viewport, cx, cy, 120, 0)); await sleep(80) } await sleep(400) })
    check('pan-x', viewport.scrollLeft !== left0)
    const pan0 = [viewport.scrollLeft, viewport.scrollTop]
    await step('drag-pan', async () => {
      pointer(viewport, 'pointerdown', cx, cy)
      for (let i = 1; i <= 24; i++) { sync(() => pointer(viewport, 'pointermove', cx, cy - i * 8)); await raf() }
      pointer(viewport, 'pointerup', cx, cy - 192)
    })
    check('drag-pan', Math.abs(viewport.scrollTop - pan0[1]) > 50)
    const transform0 = scaler.style.transform
    await step('zoom-in', async () => { for (let i = 0; i < 12; i++) { sync(() => wheel(viewport, cx, cy, -60, true)); await sleep(40) } })
    await sleep(400)
    note('zoom', { transform0, transform1: scaler.style.transform, width: svg.getAttribute('width') })
    check('zoom-applied', scaler.style.transform !== transform0)
    await step('zoom-out', async () => { for (let i = 0; i < 12; i++) { sync(() => wheel(viewport, cx, cy, 60, true)); await sleep(40) } })
    await sleep(400)
    await step('zoom-keys', async () => { for (const k of ['+', '+', '-', '-']) { sync(() => key(viewport, k)); await sleep(180) } })
    await step('fit-key', async () => { sync(() => key(viewport, '0')); await sleep(300) })
    const cam = s => root.querySelector(`[data-camera="${s}"]`)
    await step('zoom-buttons', async () => { for (const s of ['in', 'in', 'out', 'out', 'fit']) { if (cam(s)) sync(() => cam(s).click()); await sleep(180) } })
    const rows = [...document.querySelectorAll('.vault-entry-select')].slice(0, 12)
    await step('hover-rows', () => hoverEach(rows))
    // A row click pulls its can out and glides the camera to it; arrows step along the rack.
    await step('select-row', async () => { rows[3].click(); await sleep(400) })
    const selected = current()
    check('select-row', selected !== null)
    await step('step-keys', async () => { for (let i = 0; i < 4; i++) { sync(() => key(viewport, 'ArrowRight')); await sleep(200) } })
    check('step-keys', current() !== null && current() !== selected)
    await step('open-card', async () => { sync(() => key(viewport, 'Enter')); await sleep(400) })
    check('card-open', dialog.open)
    const t0 = cardTitle()
    const card = s => dialog.querySelector(`[data-card="${s}"]`)
    await step('card-next', async () => { for (let i = 0; i < 3; i++) { if (card('next')) sync(() => card('next').click()); await sleep(180) } })
    await step('card-keys', async () => { for (const k of ['ArrowRight', 'ArrowRight', 'ArrowLeft']) { sync(() => key(dialog, k)); await sleep(180) } })
    const dr = dialog.getBoundingClientRect()
    await step('card-wheel', async () => { for (let i = 0; i < 3; i++) { sync(() => wheelXY(dialog, dr.left + 8, dr.top + 8, 0, 120)); await sleep(260) } })
    note('card', { t0, t1: cardTitle() })
    check('card-browse', cardTitle() !== t0)
    await step('close-card', async () => { if (card('close')) sync(() => card('close').click()); else escape(); await sleep(300) })
    check('card-closed', !dialog.open)
    const input = document.querySelector('#base-search-input')
    const restore = lexical()
    await step('search-type', () => typeInto(input, 'the', 110))
    const dimmed = svg.querySelectorAll('[data-dimmed]').length
    await step('search-clear', async () => { clearInput(input); await sleep(200) })
    restore()
    note('search', { dimmed })
    check('search-dims', dimmed > 0 && svg.querySelectorAll('[data-dimmed]').length === 0)
    check('scene-nodes', svg.querySelectorAll('*').length > 1000)
    return __bp.result()
    """,
}


# Base search on the shared bar: type a query (one filter pass per keystroke), then clear it.
SEARCH = r"""
    const input = document.querySelector('#base-search-input')
    const items = [...document.querySelectorAll('.main-col > .base-view :is(.base-card, .base-list > li, table.base-table tbody tr:not(.base-group-header))')]
    const visibleCount = () => items.filter(i => i.style.display !== 'none').length
    const restore = lexical()
    await step('search-type', () => typeInto(input, 'the', 110))
    const narrowed = visibleCount()
    await step('search-clear', async () => { clearInput(input); await sleep(200) })
    restore()
    note('search', { total: items.length, narrowed })
    check('search', narrowed > 0 && narrowed < items.length && visibleCount() === items.length)
"""


def cards(path):
  return {
    'path': path,
    'ready': r"""document.querySelector('.base-view') && document.querySelectorAll('.base-view a[href]').length > 10""",
    'settle': 800,
    'script': HELPERS
    + r"""
    await step('scroll-down', () => pageScroll(300, 20))
    await step('scroll-up', () => pageScroll(-300, 20))
    const links = [...document.querySelectorAll('.base-view a[href]')].slice(0, 16)
    await step('hover-cards', () => hoverEach(links))
    check('cards', links.length === 16)
    """
    + SEARCH
    + '    return __bp.result()',
  }


def table(path):
  return {
    'path': path,
    'ready': r"""document.querySelector('.base-table') && document.querySelectorAll('.base-table-cell-content').length > 10""",
    'settle': 800,
    'script': HELPERS
    + r"""
    await step('scroll-down', () => pageScroll(300, 20))
    const heads = [...document.querySelectorAll('.base-table th, .base-table thead button')].slice(0, 3)
    await step('sort-clicks', async () => { for (const h of heads) { sync(() => h.click()); await sleep(80); sync(() => h.click()); await sleep(80) } })
    const toggles = [...document.querySelectorAll('[data-base-query-toggle]')]
    await step('query-panel', async () => { for (const t of toggles.slice(0, 2)) { sync(() => t.click()); await sleep(200) } })
    const rows = [...document.querySelectorAll('.base-table tbody tr')].slice(0, 16)
    await step('hover-rows', () => hoverEach(rows))
    check('rows', rows.length > 8)
    """
    + SEARCH
    + '    return __bp.result()',
  }


kanban = {
  'path': '/cinematheque/analysis',
  'ready': r"""document.querySelector('.base-view') && document.querySelectorAll('.base-view a[href]').length > 5""",
  'settle': 800,
  'script': HELPERS
  + r"""
    const board = document.querySelector('.base-kanban, .base-view')
    await step('scroll-x', async () => { for (let i = 0; i < 12; i++) { sync(() => { board.scrollLeft += 200 }); await raf() } })
    await step('scroll-y', () => pageScroll(300, 12))
    const links = [...document.querySelectorAll('.base-view a[href]')].slice(0, 16)
    await step('hover-cards', () => hoverEach(links))
    check('cards', links.length > 5)
    """
  + SEARCH
  + '    return __bp.result()',
}


def mapbox(path, ready, settle=2500):
  # Pan and zoom on a mapbox canvas; the pointer path goes through the canvas's own listeners.
  return {
    'path': path,
    'ready': ready,
    'settle': settle,
    'script': HELPERS
    + r"""
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


places_map = mapbox(
  '/places-to-go',
  r"""document.querySelector('.base-map canvas.mapboxgl-canvas')""",
  4000,
)
