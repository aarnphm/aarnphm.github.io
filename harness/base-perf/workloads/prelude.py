"""In-page helpers every scenario script gets."""

# Installed once the page is ready: buffered long-task and long-animation-frame observers, a
# display-interval calibration (20 idle rAF gaps; 16.7 ms at 60 Hz, 8.3 at 120) and the step timer.
# `step` runs an action and waits two animation frames; frame_ms is its wall time. While it runs, a
# rAF loop of its own (one per step, so a finished loop never stamps into the next step) stamps
# every frame the compositor produced: fps is stamps over wall time, a gap over
# 1.5 intervals counts the frames it skipped as dropped, worst_ms is the longest gap, and the
# long-animation-frame entries that landed inside the step give loaf and its blocking time. Inside
# an action, `sync` runs one dispatch and forces a layout, so the handler, style and layout cost of
# that event lands in the step's sync_ms (the waits between events do not).
PRELUDE = r"""
window.__bp = { longtasks: [], loaf: [], steps: [], checks: {}, notes: {}, _sync: 0, interval: 16.7 }
try { new PerformanceObserver(l => { for (const e of l.getEntries()) __bp.longtasks.push(+e.duration.toFixed(1)) }).observe({ type: 'longtask', buffered: true }) } catch {}
try { new PerformanceObserver(l => { for (const e of l.getEntries()) __bp.loaf.push({ d: +e.duration.toFixed(1), b: +(e.blockingDuration || 0).toFixed(1) }) }).observe({ type: 'long-animation-frame', buffered: true }) } catch {}
__bp.raf = () => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))
__bp.sleep = ms => new Promise(r => setTimeout(r, ms))
__bp.sampler = () => { const st = [], s = { on: true, stamps: st }; const tick = t => { if (!s.on) return; if (t !== st[st.length - 1]) st.push(t); requestAnimationFrame(tick) }; requestAnimationFrame(tick); return s }
;(async () => { const st = []; for (let i = 0; i < 21; i++) st.push(await new Promise(requestAnimationFrame)); const g = st.slice(1).map((t, i) => t - st[i]).sort((a, b) => a - b); __bp.interval = +g[Math.floor(g.length / 2)].toFixed(2); __bp.notes.hz = Math.round(1000 / __bp.interval) })()
__bp.step = async (name, action) => {
  const t0 = performance.now(), s0 = __bp._sync, l0 = __bp.loaf.length, sample = __bp.sampler()
  await action()
  await __bp.raf()
  sample.on = false
  const st = sample.stamps, wall = performance.now() - t0, iv = __bp.interval
  let dropped = 0, worst = 0
  for (let i = 1; i < st.length; i++) { const g = st[i] - st[i - 1]; if (g > worst) worst = g; if (g > 1.5 * iv) dropped += Math.round(g / iv) - 1 }
  const loaf = __bp.loaf.slice(l0)
  __bp.steps.push({ name, sync_ms: +(__bp._sync - s0).toFixed(2), frame_ms: +wall.toFixed(2), frames: st.length, fps: +(st.length / wall * 1000).toFixed(1), dropped, worst_ms: +worst.toFixed(1), loaf: loaf.length, loaf_block_ms: +loaf.reduce((a, e) => a + e.b, 0).toFixed(1) })
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
__bp.wheelXY = (el, x, y, deltaX, deltaY, extra = {}) => el.dispatchEvent(new WheelEvent('wheel', { clientX: x, clientY: y, deltaX, deltaY, deltaMode: 0, bubbles: true, cancelable: true, ...extra }))
__bp.key = (el, key, extra = {}) => el.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true, ...extra }))
// Typing: one input event per character, the way a keyboard delivers them; clearing is one event.
__bp.typeInto = async (input, text, gap = 90) => { for (const ch of text) { __bp.sync(() => { input.value += ch; input.dispatchEvent(new InputEvent('input', { bubbles: true, data: ch, inputType: 'insertText' })) }); await __bp.sleep(gap) } }
__bp.clearInput = input => __bp.sync(() => { input.value = ''; input.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'deleteContentBackward' })) })
// Search scenarios measure the lexical filter; a stored semantic preference is switched off and restored.
__bp.lexical = () => { const bar = document.querySelector('#base-search-bar'); if (bar?.dataset.mode !== 'semantic') return () => {}; bar.querySelector('.mode-option[data-mode="lexical"]')?.click(); return () => bar.querySelector('.mode-option[data-mode="semantic"]')?.click() }
__bp.result = () => ({ steps: __bp.steps.splice(0), checks: { ...__bp.checks }, notes: { ...__bp.notes, visibility: document.visibilityState } })
"""

HELPERS = 'const { step, raf, sleep, wheel, wheelXY, key, typeInto, clearInput, lexical, pointer, hover, unhover, hoverEach, scrollBy, pageScroll, escape, check, note, sync } = __bp'
