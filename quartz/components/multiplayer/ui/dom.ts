import type { MarkdownEditor, MarkdownEditorConfig } from '../../scripts/markdown-editor'

let markdownEditorModule: Promise<typeof import('../../scripts/markdown-editor')> | null = null

const notebookCellBoundarySelector = '.notebook-code-cell'

export async function createMarkdownEditor(config: MarkdownEditorConfig): Promise<MarkdownEditor> {
  markdownEditorModule ??= import('../../scripts/markdown-editor')
  const { MarkdownEditor } = await markdownEditorModule
  return new MarkdownEditor(config)
}

export function rangeIntersectsNotebookCellBoundary(range: Range): boolean {
  for (const cell of document.querySelectorAll(notebookCellBoundarySelector)) {
    if (range.intersectsNode(cell)) return true
  }
  return false
}

export const formatRelativeTime = (timestamp: number): string => {
  const now = Date.now()
  const diff = now - timestamp
  const seconds = Math.floor(diff / 1000)
  const minutes = Math.floor(seconds / 60)
  const hours = Math.floor(minutes / 60)
  const days = Math.floor(hours / 24)
  const weeks = Math.floor(days / 7)
  const months = Math.floor(days / 30)
  const years = Math.floor(days / 365)

  if (years > 0) return `${years} yr${years > 1 ? 's' : ''} ago`
  if (months > 0) return `${months} mo${months > 1 ? 's' : ''} ago`
  if (weeks > 0) return `${weeks} wk${weeks > 1 ? 's' : ''} ago`
  if (days > 0) return `${days} day${days > 1 ? 's' : ''} ago`
  if (hours > 0) return `${hours} hr${hours > 1 ? 's' : ''} ago`
  if (minutes > 0) return `${minutes} min${minutes > 1 ? 's' : ''} ago`
  return 'just now'
}

export const correctionOpId = (commentId: string, start: number, end: number): string => {
  return `correction:${commentId}:${start}:${end}`
}

export const getTextNodeRects = (range: Range): DOMRect[] => {
  const rects: DOMRect[] = []
  const rootNode =
    range.commonAncestorContainer.nodeType === Node.TEXT_NODE
      ? range.commonAncestorContainer.parentElement
      : range.commonAncestorContainer
  if (!rootNode) return rects
  const walker = document.createTreeWalker(rootNode, NodeFilter.SHOW_TEXT)

  let currentNode: Node | null = null
  while ((currentNode = walker.nextNode())) {
    if (!(currentNode instanceof Text)) continue
    const node = currentNode
    if (!range.intersectsNode(node)) continue

    const nodeRange = document.createRange()
    nodeRange.selectNodeContents(node)

    const startOffset = node === range.startContainer ? range.startOffset : 0
    const endOffset = node === range.endContainer ? range.endOffset : node.length

    if (startOffset >= endOffset) continue

    nodeRange.setStart(node, startOffset)
    nodeRange.setEnd(node, endOffset)

    const nodeRects = nodeRange.getClientRects()
    for (const rect of nodeRects) {
      if (rect.width > 0 && rect.height > 0) {
        rects.push(rect)
      }
    }
  }

  return rects
}

export const copyToClipboard = async (text: string) => {
  try {
    await navigator.clipboard.writeText(text)
    return true
  } catch (err) {
    console.error('Failed to copy:', err)
    return false
  }
}

export const keepModalInViewport = (modal: HTMLElement) => {
  const rect = modal.getBoundingClientRect()
  const maxLeft = document.documentElement.clientWidth - rect.width - 12
  const left = Math.max(12, Math.min(rect.left, maxLeft))
  modal.style.left = `${left + window.scrollX}px`
  modal.style.right = 'auto'
}
