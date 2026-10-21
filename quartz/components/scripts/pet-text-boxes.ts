type TextBox = { left: number; right: number; top: number; bottom: number }

// Read once for a route search. Repeated hit tests at every footstep traverse the entire SVG
// forest before the trail compositor is ready and can hold the main thread for seconds.
export function visibleTextBoxes(document: Document, exclude: Element | null): TextBox[] {
  const view = document.defaultView
  if (!view) return []
  const viewport: TextBox = { left: 0, top: 0, right: view.innerWidth, bottom: view.innerHeight }
  const clips = new Map<Element, TextBox | null>()
  const clipOf = (element: Element): TextBox | null => {
    const known = clips.get(element)
    if (known !== undefined) return known
    const parent = element.parentElement
    const inherited = parent && parent !== document.body ? clipOf(parent) : viewport
    if (!inherited) {
      clips.set(element, null)
      return null
    }
    const style = view.getComputedStyle(element)
    if (style.display === 'none' || style.visibility === 'hidden') {
      clips.set(element, null)
      return null
    }
    const clip = { ...inherited }
    if (style.overflowX !== 'visible' || style.overflowY !== 'visible') {
      const rect = element.getBoundingClientRect()
      if (style.overflowX !== 'visible') {
        clip.left = Math.max(clip.left, rect.left)
        clip.right = Math.min(clip.right, rect.right)
      }
      if (style.overflowY !== 'visible') {
        clip.top = Math.max(clip.top, rect.top)
        clip.bottom = Math.min(clip.bottom, rect.bottom)
      }
    }
    const visible = clip.right > clip.left && clip.bottom > clip.top ? clip : null
    clips.set(element, visible)
    return visible
  }
  // Off-screen blocks are skipped whole: walking their text costs most of the search on a long
  // page (9 ms → 2 ms on a triathlon day). Only static blocks a viewport or more away are skipped;
  // their text stays inside them unless it overflows by that much or positions against an
  // ancestor outside them.
  const far = (element: Element): boolean => {
    if (element.childElementCount < 3) return false
    const rect = element.getBoundingClientRect()
    return (
      (rect.bottom < -viewport.bottom || rect.top > 2 * viewport.bottom) &&
      view.getComputedStyle(element).position === 'static'
    )
  }
  const nodes = document.createTreeWalker(
    document.body,
    NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT,
  )
  const pastSubtree = (): Node | null => {
    for (;;) {
      const next = nodes.nextSibling()
      if (next) return next
      if (!nodes.parentNode()) return null
    }
  }
  const range = document.createRange()
  const boxes: TextBox[] = []
  let node = nodes.nextNode()
  while (node) {
    if (node instanceof Element) {
      const skip =
        node === exclude ||
        node.tagName === 'SCRIPT' ||
        node.tagName === 'STYLE' ||
        node.classList.contains('trail-canvas') ||
        far(node)
      node = skip ? pastSubtree() : nodes.nextNode()
      continue
    }
    const parent = node.parentElement
    const clip = parent && /\S/.test((node as Text).data) ? clipOf(parent) : null
    if (clip) {
      range.selectNodeContents(node)
      // Keep overflowing and fixed descendants even when their parent's box is outside the view.
      for (const rect of range.getClientRects()) {
        const box = {
          left: Math.max(rect.left, clip.left),
          right: Math.min(rect.right, clip.right),
          top: Math.max(rect.top, clip.top),
          bottom: Math.min(rect.bottom, clip.bottom),
        }
        if (box.right > box.left && box.bottom > box.top) boxes.push(box)
      }
    }
    node = nodes.nextNode()
  }
  return boxes
}
