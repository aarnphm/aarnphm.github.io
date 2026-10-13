import { trailIds } from './layout'

// Each place the reading board links to has a trail and a blaze colour. Pointing at a link (or
// focusing it) lights that trail's blaze posts and signposts and walks its footprints out from the
// trailhead. Links take trails in the order their destinations first appear on the board, so the
// same link keeps the same blaze between visits.
export function mountTrailBlazes(scene: HTMLElement, signal: AbortSignal) {
  const document = scene.ownerDocument
  const routes = new Map<string, number>()
  const blazeOf = (link: HTMLAnchorElement) => {
    const landing = link.closest('.landing')
    if (!landing || link.origin !== location.origin) return null
    if (!routes.size)
      for (const other of landing.querySelectorAll<HTMLAnchorElement>('a[href]'))
        if (other.origin === location.origin && !routes.has(other.pathname))
          routes.set(other.pathname, routes.size % trailIds.length)
    return routes.get(link.pathname) ?? null
  }
  const steps = (route: string) => {
    const group = scene.querySelector<SVGGElement>(`.trail-steps [data-route="${route}"]`)
    // A redrawn scene brings unnumbered footprints.
    if (!group || group.dataset.numbered) return
    group.dataset.numbered = 'true'
    for (const [index, mark] of [...group.children].entries())
      (mark as SVGElement).style.setProperty('--step', String(index))
  }
  let lit: HTMLAnchorElement | null = null
  const light = (link: HTMLAnchorElement | null) => {
    if (link === lit) return
    if (lit) {
      delete lit.dataset.trailBlaze
      lit.style.removeProperty('--trail-blaze')
    }
    lit = link
    const blaze = link ? blazeOf(link) : null
    if (!link || blaze === null) {
      lit = null
      delete scene.dataset.trailLit
      delete scene.dataset.trailLitBlaze
      return
    }
    const route = trailIds[blaze]
    link.dataset.trailBlaze = String(blaze)
    // The link's underline takes the blaze colour the scene paints.
    const post = scene.querySelector(`.trail-wayside-blaze-${blaze}`)
    if (post) link.style.setProperty('--trail-blaze', getComputedStyle(post).fill)
    steps(route)
    scene.dataset.trailLit = route
    scene.dataset.trailLitBlaze = String(blaze)
  }
  const target = (event: Event) =>
    (event.target as Element | null)?.closest?.<HTMLAnchorElement>('.landing a[href]') ?? null
  document.addEventListener('pointerover', event => light(target(event)), { signal })
  document.addEventListener('focusin', event => light(target(event)), { signal })
  document.addEventListener(
    'pointerout',
    event => {
      if (lit && !lit.contains(event.relatedTarget as Node | null)) light(null)
    },
    { signal },
  )
  document.addEventListener('focusout', () => light(null), { signal })
  signal.addEventListener('abort', () => light(null), { once: true })
}
