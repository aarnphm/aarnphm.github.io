// The runtime arrives after the shared page chunk (together about 1.5 MB before compression), so
// on a slow link the toolbar paints many seconds before anything listens to it, and a tap in
// that window did nothing. The boot script runs before the body parses and marks the control
// that was tapped; the runtime replays the tap once it has mounted. gear-shell.scss draws the
// held state.
export const TRI_READY_ATTR = 'data-tri-ready'
const QUEUED_ATTR = 'data-tri-queued'
const CONTROLS = '.tri-note-foot button, .tri-cmdk-trigger'

export const TRI_TAP_QUEUE_SCRIPT = `if (!window.__triTapQueue) { window.__triTapQueue = true; document.addEventListener('click', function (event) { var target = event.target; if (!(target instanceof Element)) return; var root = target.closest('.triathlon'); if (!root || root.hasAttribute('${TRI_READY_ATTR}')) return; var control = target.closest('${CONTROLS}'); if (!control) return; root.querySelectorAll('[${QUEUED_ATTR}]').forEach(function (held) { held.removeAttribute('${QUEUED_ATTR}'); held.removeAttribute('aria-busy') }); control.setAttribute('${QUEUED_ATTR}', ''); control.setAttribute('aria-busy', 'true') }) }`

/** Mark the root as listening and run the tap it held, if any. Returns the undo for unmount. */
export function releaseQueuedTap(root: HTMLElement): () => void {
  root.setAttribute(TRI_READY_ATTR, '')
  const held = root.querySelector<HTMLElement>(`[${QUEUED_ATTR}]`)
  if (held) {
    held.removeAttribute(QUEUED_ATTR)
    held.removeAttribute('aria-busy')
    held.click()
  }
  return () => root.removeAttribute(TRI_READY_ATTR)
}
