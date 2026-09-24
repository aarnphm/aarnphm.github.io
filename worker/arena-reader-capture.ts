// This function also runs through Puppeteer's serialized page.evaluate boundary.
export function serializeArenaReaderDocument(limit: number, source: Document = document): string {
  const mathJax: unknown = 'MathJax' in window ? window.MathJax : undefined
  const startup =
    mathJax && typeof mathJax === 'object' && 'startup' in mathJax ? mathJax.startup : null
  const mathDocument =
    startup && typeof startup === 'object' && 'document' in startup ? startup.document : null
  const items =
    mathDocument && typeof mathDocument === 'object' && 'math' in mathDocument
      ? mathDocument.math
      : null
  if (
    items &&
    typeof items === 'object' &&
    Symbol.iterator in items &&
    typeof items[Symbol.iterator] === 'function'
  ) {
    for (const item of items) {
      if (
        !item ||
        typeof item !== 'object' ||
        !('math' in item) ||
        !('typesetRoot' in item) ||
        !('inputJax' in item)
      )
        continue
      const inputJax = item.inputJax
      const root = item.typesetRoot
      if (
        typeof item.math === 'string' &&
        item.math.length <= 16_384 &&
        inputJax &&
        typeof inputJax === 'object' &&
        'name' in inputJax &&
        inputJax.name === 'TeX' &&
        root instanceof Element &&
        root.ownerDocument === source &&
        root.matches('mjx-container, .MathJax')
      )
        root.setAttribute('data-latex', item.math)
    }
  }
  for (const container of source.querySelectorAll('mjx-container, .MathJax')) {
    const math = container.querySelector('mjx-assistive-mml math, .MJX_Assistive_MathML math')
    if (!math) continue
    const semantic = math.cloneNode(true)
    if (semantic instanceof Element) {
      if (container.getAttribute('display') === 'true' || container.closest('.MathJax_Display'))
        semantic.setAttribute('display', 'block')
      const latex = container.getAttribute('data-latex')
      if (latex) semantic.setAttribute('data-latex', latex)
      container.replaceWith(semantic)
    }
  }
  for (const image of source.querySelectorAll('img')) {
    const url =
      image.currentSrc ||
      image.getAttribute('data-src') ||
      image.getAttribute('data-original') ||
      image.src
    if (url) image.setAttribute('src', url)
  }
  // Rendering is finished. Keep structured article metadata for Defuddle, without
  // serializing executable bundles, stylesheets, or duplicated equation glyphs.
  for (const element of source.querySelectorAll('script, style, link[rel="stylesheet"]')) {
    if (element.getAttribute('type') !== 'application/ld+json') element.remove()
  }
  return source.documentElement.outerHTML.slice(0, limit + 1)
}
