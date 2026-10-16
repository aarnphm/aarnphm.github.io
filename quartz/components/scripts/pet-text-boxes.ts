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
  const nodes = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT)
  const range = document.createRange()
  const boxes: TextBox[] = []
  for (let node = nodes.nextNode(); node; node = nodes.nextNode()) {
    const parent = node.parentElement
    if (
      !parent ||
      !node.textContent?.trim() ||
      exclude?.contains(node) ||
      parent.closest('script, style, .trail-canvas')
    )
      continue
    const clip = clipOf(parent)
    if (!clip) continue
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
  return boxes
}
