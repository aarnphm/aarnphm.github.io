import { MarkdownRenderer, type MarkdownPostProcessorContext } from 'obsidian'
import { MarkerRenderChild, type MarkerAnnotations } from './annotations'
import { findMarkers, maskMarkdown } from './parser'

function collectText(root: HTMLElement): { nodes: Text[]; text: string; masked: string } {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT)
  const nodes: Text[] = []
  let text = ''
  let masked = ''
  let node: Node | null
  while ((node = walker.nextNode())) {
    if (!(node instanceof Text)) continue
    if (node.parentElement?.closest('.garden-marked, svg.rough-annotation')) continue
    nodes.push(node)
    text += node.data
    const excluded = node.parentElement?.closest(
      'code, pre, mjx-container, .math, .frontmatter, .metadata-container, .internal-embed',
    )
    masked += excluded ? node.data.replace(/[^\n]/g, ' ') : node.data
  }
  return { nodes, text, masked }
}

function rangeAt(nodes: Text[], from: number, to: number): Range | undefined {
  const range = document.createRange()
  let offset = 0
  let started = false
  for (const node of nodes) {
    const end = offset + node.length
    if (!started && from < end) {
      range.setStart(node, from - offset)
      started = true
    }
    if (started && to <= end) {
      range.setEnd(node, to - offset)
      return range
    }
    offset = end
  }
}

export function processMarkers(
  el: HTMLElement,
  ctx: MarkdownPostProcessorContext,
  manager: MarkerAnnotations,
): void {
  if (el.closest('.garden-marked, .cm-editor') || !el.textContent?.includes('::')) return
  const section = ctx.getSectionInfo(el)
  const source = section?.text
    .split('\n')
    .slice(section.lineStart, section.lineEnd + 1)
    .join('\n')
  const originals =
    source === undefined ? undefined : findMarkers(source, maskMarkdown(source), true)
  if (originals?.every(match => match.escaped)) return
  const { nodes, text, masked } = collectText(el)
  const matches = findMarkers(text, masked, true)
  // Markdown removes escape backslashes. Preserve their meaning using the source section.
  const paired = originals?.length === matches.length
  for (let index = matches.length - 1; index >= 0; index--) {
    const match = matches[index]
    if (match.escaped || (paired && originals?.[index].escaped)) continue
    const range = rangeAt(nodes, match.from, match.to)
    const contentRange = rangeAt(nodes, match.contentFrom, match.contentTo)
    if (!range || !contentRange) continue
    const element = document.createElement('span')
    const child = new MarkerRenderChild(element, match.intensity, manager, () => {
      if (!element.parentNode) return
      element.replaceWith(
        '::',
        ...Array.from(child.target.childNodes),
        text.slice(match.contentTo, match.to),
      )
    })
    child.target.append(contentRange.cloneContents())
    range.deleteContents()
    range.insertNode(element)
    child.ready = true
    ctx.addChild(child)
  }
}

export async function renderMarkerContent(
  child: MarkerRenderChild,
  source: string,
  sourcePath: string,
  manager: MarkerAnnotations,
): Promise<void> {
  await MarkdownRenderer.render(manager.plugin.app, source, child.target, sourcePath, child)
  const paragraph = child.target.firstElementChild
  if (paragraph?.tagName === 'P' && child.target.children.length === 1) {
    paragraph.replaceWith(...Array.from(paragraph.childNodes))
  }
  child.ready = true
  manager.schedule()
}
