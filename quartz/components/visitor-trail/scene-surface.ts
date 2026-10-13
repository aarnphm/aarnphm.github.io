function ownsScene(rules: CSSRuleList, view: Window & typeof globalThis): boolean {
  return [...rules].some(rule =>
    rule instanceof view.CSSStyleRule
      ? ['.homepage-trail-scene', '.g-select', '.g-icon-button'].some(selector =>
          rule.selectorText.includes(selector),
        )
      : rule instanceof view.CSSGroupingRule && ownsScene(rule.cssRules, view),
  )
}

export function mountSceneSurface(host: HTMLElement, scene: HTMLElement, signal: AbortSignal) {
  const document = host.ownerDocument
  const view = document.defaultView
  if (!view) throw new Error('The trail surface needs its document')
  const shadow = host.shadowRoot ?? host.attachShadow({ mode: 'open' })
  shadow.replaceChildren()
  const ready: Promise<void>[] = []
  for (const sheet of document.styleSheets) {
    if (sheet.href && new URL(sheet.href).origin !== view.location.origin) continue
    if (!ownsScene(sheet.cssRules, view)) continue
    const owner = sheet.ownerNode
    if (owner instanceof view.HTMLStyleElement) {
      const style = document.createElement('style')
      style.textContent = owner.textContent
      shadow.append(style)
    } else if (owner instanceof view.HTMLLinkElement) {
      const link = document.createElement('link')
      link.rel = 'stylesheet'
      link.href = owner.href
      // CSSOM serialization loses variable animation shorthands after a longhand override.
      // Reusing the cached source preserves wind and stride motion.
      ready.push(
        new Promise((resolve, reject) => {
          link.addEventListener('load', () => resolve(), { once: true, signal })
          link.addEventListener(
            'error',
            () => reject(new Error('The trail stylesheet did not load')),
            { once: true, signal },
          )
          signal.addEventListener('abort', () => resolve(), { once: true })
        }),
      )
      shadow.append(link)
    }
  }
  const reset = document.createElement('style')
  reset.textContent =
    '.homepage-trail-scene { height: 100%; } [hidden] { display: none !important; } button, input, select { font: inherit; }'
  const surface = document.createElement('div')
  surface.className = 'homepage-trail-scene'
  scene.hidden = true
  surface.append(scene)
  // Scene mutations cannot reach the article and navigation through site-wide :has() rules.
  shadow.append(reset, surface)
  return Promise.all(ready).then(() => {
    if (signal.aborted) return
    scene.hidden = false
    host.dispatchEvent(new Event('trail:mounted'))
  })
}
