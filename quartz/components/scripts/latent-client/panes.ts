import { type Runtime } from './runtime'

/**
 * The reader has no end: it mounts the next document as the last one nears. A pane the layout
 * stacks under it (the probe below 1100px, every pane on a phone) can then never be scrolled to.
 * The bar at the foot of a narrow screen opens such a pane as a sheet over the reader
 * (latent-panes.scss); the reader stays mounted under it, so its scroll position survives. A
 * navigation from inside a sheet (a link, a rail label, a point on the bench) returns to the
 * reader, which is where the document it asked for shows.
 */
export function bindPanes(rt: Runtime): void {
  const bar = rt.root.querySelector<HTMLElement>(':scope > .lat-panes')
  if (!bar) return
  const buttons = [...bar.querySelectorAll<HTMLButtonElement>('button[data-pane]')]
  const show = (pane: string) => {
    if (pane === 'reader') delete rt.root.dataset.latPane
    else rt.root.dataset.latPane = pane
    for (const button of buttons)
      button.setAttribute('aria-pressed', String(button.dataset.pane === pane))
  }
  bar.addEventListener(
    'click',
    event => {
      const button = (event.target as HTMLElement | null)?.closest<HTMLButtonElement>(
        'button[data-pane]',
      )
      const pane = button?.dataset.pane
      if (!pane) return
      // the open pane's own button closes it
      show(pane === rt.root.dataset.latPane ? 'reader' : pane)
    },
    { signal: rt.signal },
  )
  rt.root.addEventListener('latent:goto', () => show('reader'), { signal: rt.signal })
  rt.root.addEventListener(
    'keydown',
    event => {
      if (event.key !== 'Escape' || !rt.root.dataset.latPane || event.defaultPrevented) return
      // a field keeps its own Escape (clear the query, close a menu)
      if ((event.target as HTMLElement | null)?.closest('input, select, [role="listbox"]')) return
      show('reader')
    },
    { signal: rt.signal },
  )
  rt.signal.addEventListener('abort', () => show('reader'))
}
