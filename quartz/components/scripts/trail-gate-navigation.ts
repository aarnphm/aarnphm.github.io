import { GATE } from './404-gate'

export function mountTrailGateNavigation(host: HTMLElement, signal: AbortSignal) {
  let entering = false
  host.addEventListener(
    'click',
    event => {
      const link = event
        .composedPath()
        .find(
          target => target instanceof HTMLAnchorElement && target.matches('.trail-far-gate-link'),
        )
      if (
        !(link instanceof HTMLAnchorElement) ||
        event.button !== 0 ||
        event.metaKey ||
        event.ctrlKey ||
        event.shiftKey ||
        event.altKey
      )
        return
      event.preventDefault()
      if (entering) return
      entering = true
      const url = new URL(link.href)
      const source = host.shadowRoot?.querySelector<HTMLCanvasElement>('.trail-compositor')
      const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches
      if (source && !reduced) {
        const snapshot = document.createElement('canvas')
        snapshot.className = 'trail-gate-transition'
        snapshot.setAttribute('aria-hidden', 'true')
        const ratio = devicePixelRatio
        snapshot.width = Math.ceil(innerWidth * ratio)
        snapshot.height = Math.ceil(innerHeight * ratio)
        const paint = snapshot.getContext('2d')
        if (paint) {
          const bounds = source.getBoundingClientRect()
          const gate = link.getBoundingClientRect()
          paint.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--light')
          paint.fillRect(0, 0, snapshot.width, snapshot.height)
          paint.drawImage(
            source,
            bounds.left * ratio,
            bounds.top * ratio,
            bounds.width * ratio,
            bounds.height * ratio,
          )
          const x = gate.left + gate.width / 2
          const y = gate.top + gate.height / 2
          // The 404's 1600 by 1000 SVG uses centered slice with 48px of overscan.
          const scale = Math.max((innerWidth + 96) / 1600, (innerHeight + 96) / 1000)
          const destinationX = innerWidth / 2
          const destinationY = innerHeight / 2 + (GATE.y - 500) * scale
          const zoom = ((GATE.r + GATE.ring) * scale) / (gate.width / 2 - 4)
          snapshot.style.transformOrigin = `${x}px ${y}px`
          document.documentElement.append(snapshot)
          // This one painted frame survives the old iframe while the destination renders below it.
          const approach = snapshot.animate(
            [
              { transform: 'translate(0px, 0px) scale(1)' },
              {
                transform: `translate(${destinationX - x}px, ${destinationY - y}px) scale(${zoom})`,
              },
            ],
            { duration: 760, easing: 'cubic-bezier(0.55, 0, 0.85, 0.35)', fill: 'forwards' },
          )
          let fade: Animation | undefined
          let arrived: () => void
          const cleanup = () => {
            document.removeEventListener('nav', arrived)
            window.removeEventListener('resize', cleanup)
            approach.cancel()
            fade?.cancel()
            snapshot.remove()
            snapshot.width = snapshot.height = 1
          }
          const destination = new Promise<void>(resolve => {
            function afterPaint() {
              if (!snapshot.isConnected || location.pathname !== url.pathname) {
                cleanup()
                return resolve()
              }
              if (document.querySelector('.nf-scene.is-ready')) resolve()
              else requestAnimationFrame(afterPaint)
            }
            arrived = () => requestAnimationFrame(afterPaint)
            document.addEventListener('nav', arrived, { once: true })
          })
          window.addEventListener('resize', cleanup, { once: true })
          void Promise.all([approach.finished, destination])
            .then(async () => {
              if (!snapshot.isConnected) return
              fade = snapshot.animate([{ opacity: 1 }, { opacity: 0 }], {
                duration: 200,
                easing: 'ease-out',
                fill: 'forwards',
              })
              await fade.finished
              cleanup()
            })
            .catch(cleanup)
        }
      }
      if (window.spaNavigate) {
        void window.spaNavigate(url, false, { transition: false, signal }).catch(() => {
          if (!signal.aborted) location.assign(url)
        })
      } else location.assign(url)
    },
    { signal },
  )
}
