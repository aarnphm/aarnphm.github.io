import type { Runtime } from './runtime'

// The axis cards under the 3-d view (markup.ts `axesCarousel`): one card a row in a track that
// scrolls sideways and snaps per card. The pager names every card; a click slides the track, and
// the card in view marks its name. A card the search matches marks its name too.

const reduced = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches

/** Slide the carousel so this axis card fills the row. */
export function showAxis(card: HTMLElement): void {
  const track = card.parentElement
  if (!track?.classList.contains('lat-axes-track')) return
  const n = Array.prototype.indexOf.call(track.children, card) as number
  track.scrollTo({ left: n * track.clientWidth, behavior: reduced() ? 'auto' : 'smooth' })
}

export function bindAxesCarousel(rt: Runtime): void {
  const track = rt.probe.querySelector<HTMLElement>('.lat-axes-track')
  const pager = rt.probe.querySelector<HTMLElement>('.lat-axes-pager')
  if (!track || !pager) return
  const names = [...pager.querySelectorAll<HTMLButtonElement>('button[data-axis]')]
  const cardOf = (id: string | undefined) =>
    track.querySelector<HTMLElement>(`.lat-axis[data-axis="${CSS.escape(id ?? '')}"]`)

  pager.addEventListener(
    'click',
    event => {
      const name = (event.target as Element).closest<HTMLButtonElement>('button[data-axis]')
      const card = cardOf(name?.dataset.axis)
      if (card) showAxis(card)
    },
    { signal: rt.signal },
  )

  let frame = 0
  const mark = () => {
    frame = 0
    const width = track.clientWidth
    if (!width) return
    const n = Math.round(track.scrollLeft / width)
    names.forEach((name, i) => {
      if (i === n) name.setAttribute('aria-current', 'true')
      else name.removeAttribute('aria-current')
    })
  }
  track.addEventListener(
    'scroll',
    () => {
      frame ||= requestAnimationFrame(mark)
    },
    { passive: true, signal: rt.signal },
  )

  // search.ts sets and clears data-match on the cards; the pager mirrors it
  const mirror = () => {
    for (const name of names)
      name.toggleAttribute('data-match', cardOf(name.dataset.axis)?.dataset.match !== undefined)
  }
  const observer = new MutationObserver(mirror)
  observer.observe(track, { attributes: true, attributeFilter: ['data-match'], subtree: true })
  rt.signal.addEventListener(
    'abort',
    () => {
      observer.disconnect()
      if (frame) cancelAnimationFrame(frame)
    },
    { once: true },
  )
}
