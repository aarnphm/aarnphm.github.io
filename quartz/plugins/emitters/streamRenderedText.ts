import type { Element, Root } from 'hast'
import { fromHtmlIsomorphic } from 'hast-util-from-html-isomorphic'
import { toText as hastToText } from 'hast-util-to-text'

const hasClass = (element: Element, className: string): boolean => {
  const value = element.properties.className
  return Array.isArray(value) && value.includes(className)
}

const findElement = (
  node: Root | Element,
  predicate: (element: Element) => boolean,
): Element | null => {
  for (const child of node.children) {
    if (child.type !== 'element') continue
    if (predicate(child)) return child
    const descendant = findElement(child, predicate)
    if (descendant) return descendant
  }
  return null
}

export interface RenderedStreamEntry {
  content: string
  wordCount: number
}

export const renderedStreamEntries = (html: string): Map<string, RenderedStreamEntry> => {
  const root = fromHtmlIsomorphic(html) as Root
  const entries = new Map<string, RenderedStreamEntry>()
  const visit = (node: Root | Element): void => {
    for (const child of node.children) {
      if (child.type !== 'element') continue
      const entryId = child.properties.dataEntryId
      if (typeof entryId === 'string' && hasClass(child, 'stream-entry')) {
        const content = findElement(child, element => hasClass(element, 'stream-entry-content'))
        const wordCount = findElement(child, element => hasClass(element, 'stream-entry-wordcount'))
        const wordCountMatch = wordCount ? hastToText(wordCount).match(/^([\d,]+) words?$/) : null
        if (content && wordCountMatch) {
          entries.set(entryId, {
            content: hastToText(content).replace(/\s+/g, ' ').trim(),
            wordCount: Number(wordCountMatch[1].replaceAll(',', '')),
          })
        }
      }
      visit(child)
    }
  }
  visit(root)
  return entries
}
