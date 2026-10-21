"""Triathlon pages: hub, day page, analytics, calendar, training, maps."""

from .prelude import HELPERS
from .bases import mapbox

tri_maps = {
  **mapbox(
    '/triathlon/maps',
    r"""document.querySelector('.tri-map canvas.mapboxgl-canvas') && document.querySelectorAll('.tri-map-mode').length > 0""",
    4000,
  )
}
tri_maps['script'] = (
  HELPERS
  + r"""
    const modes = [...document.querySelectorAll('.tri-map-mode')]
    const sports = [...document.querySelectorAll('.tri-map-sport')]
    await step('modes', async () => { for (const b of modes) { sync(() => b.click()); await sleep(400) } })
    await step('sports', async () => { for (const b of sports) { sync(() => b.click()); await sleep(400) } })
    const el = document.querySelector('.tri-map canvas.mapboxgl-canvas')
    const r = el.getBoundingClientRect()
    const cx = r.left + r.width / 2, cy = r.top + r.height / 2
    await step('zoom-in', async () => { for (let i = 0; i < 8; i++) { sync(() => wheel(el, cx, cy, -100, false)); await sleep(60) } })
    await step('drag', async () => { pointer(el, 'pointerdown', cx, cy); for (let i = 1; i <= 16; i++) { sync(() => pointer(el, 'pointermove', cx + i * 10, cy)); await sleep(16) } pointer(el, 'pointerup', cx + 160, cy) })
    const search = document.querySelector('.tri-map-search input, input.tri-map-search')
    await step('search', async () => { if (search) { search.focus(); for (const ch of 'toronto') { sync(() => { search.value += ch; search.dispatchEvent(new Event('input', { bubbles: true })) }); await sleep(60) } } })
    check('canvas', el.width > 0)
    check('modes', modes.length > 2)
    return __bp.result()
    """
)

tri_hub = {
  'path': '/triathlon',
  'ready': r"""document.querySelectorAll('.tri-bar').length > 20 && document.querySelectorAll('.tri-calendar-day').length > 100""",
  'settle': 2000,
  'script': HELPERS
  + r"""
    await step('scroll-down', () => pageScroll(400, 25))
    const bars = [...document.querySelectorAll('.tri-bar')].slice(0, 40)
    await step('hover-bars', () => hoverEach(bars))
    const dates = bars.map(b => b.dataset.dateIso).filter(Boolean).slice(-4)
    await step('focus-days', async () => { for (const date of dates) { sync(() => window.dispatchEvent(new CustomEvent('tri:focus-day', { detail: { date } }))); await sleep(300) } })
    const days = [...document.querySelectorAll('.tri-calendar-day:not([aria-hidden])')].slice(0, 60)
    await step('hover-days', () => hoverEach(days))
    const segs = [...document.querySelectorAll('.tri-seg')].slice(0, 40)
    await step('hover-segs', () => hoverEach(segs))
    await step('scroll-up', () => pageScroll(-400, 25))
    check('bars', bars.length === 40)
    check('focus', dates.length > 0)
    return __bp.result()
    """,
}

tri_day = {
  'path': '/triathlon/on/2026/10/20',
  'ready': r"""document.querySelectorAll('.tri-cax-stage').length > 3 && document.querySelectorAll('.tri-workout-analysis-tab').length > 2""",
  'settle': 2500,
  'script': HELPERS
  + r"""
    await step('scroll-down', () => pageScroll(400, 25))
    const stages = [...document.querySelectorAll('.tri-cax-stage')].slice(0, 6)
    await step('scrub-charts', async () => { for (const s of stages) { const r = s.getBoundingClientRect(); for (let i = 0; i <= 20; i++) { const x = r.left + (r.width * i) / 20; sync(() => s.dispatchEvent(new PointerEvent('pointermove', { clientX: x, clientY: r.top + r.height / 2, bubbles: true, pointerType: 'mouse' }))); await raf() } unhover(s) } })
    const tabs = [...document.querySelectorAll('.tri-workout-analysis-tab')].slice(0, 4)
    await step('analysis-tabs', async () => { for (const t of tabs) { sync(() => t.click()); await sleep(200) } })
    const env = [...document.querySelectorAll('.tri-environment-tab')].slice(0, 6)
    await step('environment-tabs', async () => { for (const t of env) { sync(() => t.click()); await sleep(200) } })
    const elev = [...document.querySelectorAll('svg.tri-elev')].slice(0, 4)
    await step('scrub-elev', async () => { for (const s of elev) { const r = s.getBoundingClientRect(); for (let i = 0; i <= 20; i++) { const x = r.left + (r.width * i) / 20; sync(() => s.dispatchEvent(new PointerEvent('pointermove', { clientX: x, clientY: r.top + r.height / 2, bubbles: true, pointerType: 'mouse' }))); await raf() } unhover(s) } })
    await step('scroll-up', () => pageScroll(-400, 25))
    check('stages', stages.length > 3)
    check('tabs', tabs.length > 2)
    return __bp.result()
    """,
}

tri_analytics = {
  'path': '/triathlon/analytics',
  'ready': r"""document.querySelectorAll('.tri-pmc-hit').length > 50 && document.querySelectorAll('.tri-trend-dot').length > 50""",
  'settle': 2500,
  'script': HELPERS
  + r"""
    await step('scroll-down', () => pageScroll(400, 25))
    const hits = [...document.querySelectorAll('.tri-pmc-hit')].slice(0, 60)
    await step('hover-pmc', () => hoverEach(hits))
    const dots = [...document.querySelectorAll('.tri-trend-dot')].slice(0, 60)
    await step('hover-trend', () => hoverEach(dots))
    const pts = [...document.querySelectorAll('.tri-wkt-pt, .tri-bodywt-pt')].slice(0, 40)
    await step('hover-points', () => hoverEach(pts))
    const tips = [...document.querySelectorAll('.tri-health-tooltip-trigger')].slice(0, 12)
    await step('hover-tooltips', () => hoverEach(tips))
    await step('scroll-up', () => pageScroll(-400, 25))
    check('hits', hits.length === 60)
    return __bp.result()
    """,
}

tri_calendar = {
  'path': '/triathlon/calendar',
  'ready': r"""document.querySelectorAll('.tri-calendar-day').length > 100 && document.querySelectorAll('.tri-calendar-event').length > 0""",
  'settle': 1500,
  'script': HELPERS
  + r"""
    await step('scroll-down', () => pageScroll(400, 20))
    const days = [...document.querySelectorAll('.tri-calendar-day:not([aria-hidden])')].slice(0, 80)
    await step('hover-days', () => hoverEach(days))
    const events = [...document.querySelectorAll('.tri-calendar-event')].slice(0, 8)
    await step('hover-events', () => hoverEach(events))
    const seasons = [...document.querySelectorAll('.tri-calendar-season-option')]
    await step('seasons', async () => { for (const s of seasons) { sync(() => s.click()); await sleep(250) } })
    const tabs = [...document.querySelectorAll('.tri-calendar-source-tab')]
    await step('source-tabs', async () => { for (const t of tabs) { sync(() => t.click()); await sleep(250) } })
    await step('scroll-up', () => pageScroll(-400, 20))
    check('days', days.length > 40)
    return __bp.result()
    """,
}

tri_training = {
  'path': '/triathlon/training',
  'ready': r"""document.querySelector('.tri-training-list') && document.querySelectorAll('.tri-key').length > 5""",
  'settle': 1500,
  'script': HELPERS
  + r"""
    await step('scroll-down', () => pageScroll(400, 20))
    const search = document.querySelector('.tri-training-search input, input.tri-training-search, .tri-ana-search input')
    await step('search-type', async () => { if (search) { search.focus(); for (const ch of 'threshold') { sync(() => { search.value += ch; search.dispatchEvent(new Event('input', { bubbles: true })) }); await sleep(60) } } })
    await step('search-clear', async () => { if (search) { sync(() => { search.value = ''; search.dispatchEvent(new Event('input', { bubbles: true })) }); await sleep(200) } })
    const keys = [...document.querySelectorAll('.tri-key')].slice(0, 10)
    await step('hover-keys', () => hoverEach(keys))
    const links = [...document.querySelectorAll('.tri-training-tree-link, .tri-training-list a')].slice(0, 10)
    await step('hover-links', () => hoverEach(links))
    await step('scroll-up', () => pageScroll(-400, 20))
    check('search', !!search)
    return __bp.result()
    """,
}
