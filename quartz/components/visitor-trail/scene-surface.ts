declare const TRAIL_SCENE_STYLES: string

export function mountSceneSurface(host: HTMLElement, scene: HTMLElement, signal: AbortSignal) {
  const document = host.ownerDocument
  const view = document.defaultView
  if (!view) throw new Error('The trail surface needs its document')
  const shadow = host.shadowRoot ?? host.attachShadow({ mode: 'open' })
  shadow.replaceChildren()
  const style = document.createElement('style')
  // Use authored CSS so variable animation shorthands survive longhand overrides.
  style.textContent = TRAIL_SCENE_STYLES
  const surface = document.createElement('div')
  surface.className = 'homepage-trail-scene'
  surface.append(scene)
  // Scene mutations cannot reach the article and navigation through site-wide :has() rules.
  shadow.append(style, surface)
  return Promise.resolve().then(() => {
    if (signal.aborted) return
    // Owners append their controls during mounting. Keep those controls inert while preparing.
    for (const control of scene.children)
      if (control instanceof view.HTMLElement) control.inert = Boolean(host.hidden)
    scene.hidden = false
    host.dispatchEvent(new Event('trail:mounted'))
  })
}
