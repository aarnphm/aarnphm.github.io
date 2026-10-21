"""In-page helpers every scenario script gets."""

# Installed once the page is ready: a buffered long-task observer and the step timer. `step` runs an
# action and waits two animation frames; frame_ms is its wall time. Inside an action, `sync` runs
# one dispatch and forces a layout, so the handler, style and layout cost of that event lands in the
# step's sync_ms (the waits between events do not).
PRELUDE = r"""
window.__bp = { longtasks: [], steps: [], checks: {}, notes: {}, _sync: 0 }
try { new PerformanceObserver(l => { for (const e of l.getEntries()) __bp.longtasks.push(+e.duration.toFixed(1)) }).observe({ type: 'longtask', buffered: true }) } catch {}
__bp.raf = () => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))
__bp.sleep = ms => new Promise(r => setTimeout(r, ms))
__bp.step = async (name, action) => {
  const t0 = performance.now(), s0 = __bp._sync
  await action()
  await __bp.raf()
  __bp.steps.push({ name, sync_ms: +(__bp._sync - s0).toFixed(2), frame_ms: +(performance.now() - t0).toFixed(2) })
}
__bp.sync = fn => { const t = performance.now(); fn(); void document.body.offsetWidth; __bp._sync += performance.now() - t }
__bp.check = (name, ok) => { __bp.checks[name] = !!ok }
__bp.note = (name, value) => { __bp.notes[name] = value }
__bp.wheel = (el, x, y, deltaY, ctrl) => el.dispatchEvent(new WheelEvent('wheel', { clientX: x, clientY: y, deltaY, deltaMode: 0, ctrlKey: ctrl, bubbles: true, cancelable: true }))
__bp.pointer = (el, type, x, y, extra = {}) => el.dispatchEvent(new PointerEvent(type, { clientX: x, clientY: y, pointerId: 1, pointerType: 'mouse', isPrimary: true, bubbles: true, cancelable: true, button: 0, buttons: type === 'pointerup' ? 0 : 1, ...extra }))
__bp.hover = (el) => { const r = el.getBoundingClientRect(); const x = r.left + r.width / 2, y = r.top + r.height / 2; for (const t of ['pointerover', 'pointerenter', 'pointermove']) el.dispatchEvent(new PointerEvent(t, { clientX: x, clientY: y, bubbles: t !== 'pointerenter', pointerType: 'mouse' })); for (const t of ['mouseover', 'mouseenter', 'mousemove']) el.dispatchEvent(new MouseEvent(t, { clientX: x, clientY: y, bubbles: t !== 'mouseenter' })) }
__bp.unhover = (el) => { for (const t of ['pointerout', 'pointerleave']) el.dispatchEvent(new PointerEvent(t, { bubbles: t !== 'pointerleave', pointerType: 'mouse' })); for (const t of ['mouseout', 'mouseleave']) el.dispatchEvent(new MouseEvent(t, { bubbles: t !== 'mouseleave' })) }
__bp.hoverEach = async (els) => { for (const el of els) { __bp.sync(() => __bp.hover(el)); await __bp.raf(); __bp.unhover(el) } }
__bp.scrollBy = async (el, dy, n) => { for (let i = 0; i < n; i++) { __bp.sync(() => { el.scrollTop += dy; el.dispatchEvent(new Event('scroll')) }); await __bp.raf() } }
__bp.pageScroll = async (dy, n) => __bp.scrollBy(document.scrollingElement, dy, n)
__bp.escape = () => { for (const t of [document, document.activeElement]) t?.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true })) }
__bp.result = () => ({ steps: __bp.steps.splice(0), checks: { ...__bp.checks }, notes: { ...__bp.notes, visibility: document.visibilityState } })
"""

HELPERS = 'const { step, raf, sleep, wheel, pointer, hover, unhover, hoverEach, scrollBy, pageScroll, escape, check, note, sync } = __bp'
