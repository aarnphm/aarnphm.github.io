import type { StructuralAnchor } from "./model"

export async function hashText(text: string): Promise<string> {
  const encoder = new TextEncoder()
  const data = encoder.encode(text)
  const hashBuffer = await crypto.subtle.digest("SHA-256", data)
  const hashArray = Array.from(new Uint8Array(hashBuffer))
  return hashArray.map((b) => b.toString(16).padStart(2, "0")).join("")
}

export function getArticleText(): string {
  const article = document.querySelector("article.popover-hint")
  return article?.textContent || ""
}

export function findClosestHeading(node: Node): string | null {
  let current: Node | null = node
  while (current) {
    if (current instanceof HTMLElement) {
      const headingId = current.getAttribute("data-heading-id")
      if (headingId) return headingId
      if (current.classList.contains("collapsible-header")) {
        return current.id || null
      }
    }
    current = current.parentNode
  }
  return null
}

export function findBlockId(node: Node): string | null {
  let current: Node | null = node
  while (current) {
    if (current instanceof HTMLElement) {
      const id = current.id
      if (id && /^[a-zA-Z0-9_-]+$/.test(id) && !id.startsWith("collapsible-")) {
        return id
      }
    }
    current = current.parentNode
  }
  return null
}

export function findContainingParagraph(node: Node): Element | null {
  let current: Node | null = node
  while (current) {
    if (current instanceof HTMLElement) {
      const tag = current.tagName.toLowerCase()
      if (tag === "p" || tag === "li" || tag === "blockquote" || tag === "td" || tag === "th") {
        return current
      }
    }
    current = current.parentNode
  }
  return null
}

export function countParagraphsBefore(section: Element | null, node: Node): number {
  if (!section) return -1
  const paragraphs = section.querySelectorAll("p, li, blockquote, td, th")
  const containingParagraph = findContainingParagraph(node)
  if (!containingParagraph) return -1
  for (let i = 0; i < paragraphs.length; i++) {
    if (paragraphs[i] === containingParagraph || paragraphs[i].contains(containingParagraph)) {
      return i
    }
  }
  return -1
}

export function computeLocalOffset(
  paragraph: Element | null,
  node: Node,
  nodeOffset: number,
): number {
  if (!paragraph) return -1
  const walker = document.createTreeWalker(paragraph, NodeFilter.SHOW_TEXT)
  let offset = 0
  while (walker.nextNode()) {
    const textNode = walker.currentNode
    if (textNode === node) {
      return offset + nodeOffset
    }
    offset += textNode.textContent?.length || 0
  }
  return offset
}

export function extractContextWords(range: Range, count: number): [string, string] {
  const articleText = getArticleText()
  const article = document.querySelector("article.popover-hint")
  if (!article) return ["", ""]

  const offsets = getRangeOffsets(range, article)
  if (!offsets) return ["", ""]

  const beforeText = articleText.slice(0, offsets.startOffset)
  const afterText = articleText.slice(offsets.endOffset)

  const beforeWords = beforeText.trim().split(/\\s+/).slice(-count).join(" ")
  const afterWords = afterText.trim().split(/\\s+/).slice(0, count).join(" ")

  return [beforeWords, afterWords]
}

export function computeStructuralAnchor(range: Range, article: Element): StructuralAnchor {
  const headingId = findClosestHeading(range.startContainer)
  const blockId = findBlockId(range.startContainer)

  let section: Element | null = null
  if (headingId) {
    section =
      article.querySelector(`[data-heading-id="${headingId}"]`) ||
      article.querySelector(`#${headingId}`)
  }

  const containingParagraph = findContainingParagraph(range.startContainer)
  const paragraphIndex = countParagraphsBefore(section || article, range.startContainer)
  const localOffset = computeLocalOffset(
    containingParagraph,
    range.startContainer,
    range.startOffset,
  )
  const contextWords = extractContextWords(range, 3)

  return { headingId, blockId, paragraphIndex, localOffset, contextWords }
}

export function recoverFromStructuralAnchor(
  anchor: StructuralAnchor,
  anchorText: string,
  article: Element,
): { startIdx: number; endIdx: number } | null {
  let section: Element | null = null
  if (anchor.headingId) {
    section =
      article.querySelector(`[data-heading-id="${anchor.headingId}"]`) ||
      article.querySelector(`#${anchor.headingId}`)
  }

  const searchWithinElement = (element: Element): { startIdx: number; endIdx: number } | null => {
    const elementText = element.textContent || ""
    const matches: number[] = []
    let searchStart = 0
    while (true) {
      const idx = elementText.indexOf(anchorText, searchStart)
      if (idx === -1) break
      matches.push(idx)
      searchStart = idx + 1
    }

    if (matches.length === 0) return null

    const walker = document.createTreeWalker(article, NodeFilter.SHOW_TEXT)
    let globalOffset = 0
    let elementStartOffset = -1
    while (walker.nextNode()) {
      const currentNode = walker.currentNode
      if (!(currentNode instanceof Text)) continue
      if (element.contains(currentNode)) {
        if (elementStartOffset === -1) {
          elementStartOffset = globalOffset
        }
      }
      globalOffset += currentNode.length
    }

    if (elementStartOffset === -1) return null

    let bestMatch = matches[0]
    if (matches.length > 1 && anchor.contextWords) {
      for (const match of matches) {
        const beforeStart = Math.max(0, match - 50)
        const afterEnd = Math.min(elementText.length, match + anchorText.length + 50)
        const context = elementText.slice(beforeStart, afterEnd)
        if (context.includes(anchor.contextWords[0]) || context.includes(anchor.contextWords[1])) {
          bestMatch = match
          break
        }
      }
    }

    return {
      startIdx: elementStartOffset + bestMatch,
      endIdx: elementStartOffset + bestMatch + anchorText.length,
    }
  }

  if (anchor.blockId) {
    const block = article.querySelector(`#${anchor.blockId}`)
    if (block) {
      const result = searchWithinElement(block)
      if (result) return result
    }
  }

  if (section && anchor.paragraphIndex >= 0) {
    const paragraphs = section.querySelectorAll("p, li, blockquote, td, th")
    if (anchor.paragraphIndex < paragraphs.length) {
      const targetParagraph = paragraphs[anchor.paragraphIndex]
      const result = searchWithinElement(targetParagraph)
      if (result) return result
    }
  }

  if (section) {
    const result = searchWithinElement(section)
    if (result) return result
  }

  return null
}

export function getRangeOffsets(range: Range, root: Element) {
  const text = range.toString()
  if (!text) return null
  const startRange = document.createRange()
  startRange.setStart(root, 0)
  startRange.setEnd(range.startContainer, range.startOffset)
  const startOffset = startRange.toString().length
  const endRange = document.createRange()
  endRange.setStart(root, 0)
  endRange.setEnd(range.endContainer, range.endOffset)
  const endOffset = endRange.toString().length
  if (endOffset <= startOffset) return null
  return { text, startOffset, endOffset }
}
