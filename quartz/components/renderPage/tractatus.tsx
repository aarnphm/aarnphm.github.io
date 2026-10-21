import { Element, ElementContent, Root, RootContent, Text } from 'hast'
import { h } from 'hastscript'

type ManualNumberInfo = {
  value: string
  container: Element
  index: number
  kind: 'element' | 'text'
  offset?: number
}

const inlineNumberPattern = /^\s*(\d+(?:\.\d+)+)\s*$/
const leadingNumberPattern = /^\s*(\d+(?:\.\d+)+)\s*/

function isListElement(node: Element): boolean {
  return node.tagName === 'ul' || node.tagName === 'ol'
}

function isListItem(node: ElementContent): node is Element {
  return node.type === 'element' && (node as Element).tagName === 'li'
}

function isMetaSection(node: Element): boolean {
  if (node.tagName !== 'section') return false
  return (
    node.properties?.dataReferences === '' ||
    node.properties?.dataFootnotes === '' ||
    node.properties?.dataBacklinks === ''
  )
}

function inlineText(node: ElementContent): string | null {
  if (node.type === 'text') return node.value
  if (node.type === 'element') {
    let value = ''
    for (const child of node.children ?? []) {
      const piece = inlineText(child)
      if (piece === null) return null
      value += piece
    }
    return value
  }
  return null
}

function findFirstParagraph(li: Element): Element | null {
  for (const child of li.children ?? []) {
    if (child.type !== 'element') continue
    const element = child as Element
    if (element.tagName === 'p') return element
  }
  return null
}

function readManualNumber(li: Element): ManualNumberInfo | null {
  const paragraph = findFirstParagraph(li)
  const container = paragraph ?? li
  if (!container.children || container.children.length === 0) return null
  let index = -1
  for (let i = 0; i < container.children.length; i += 1) {
    const child = container.children[i]
    if (child.type === 'text' && child.value.trim().length === 0) continue
    index = i
    break
  }
  if (index < 0) return null
  const first = container.children[index]
  if (first.type === 'text') {
    const match = first.value.match(leadingNumberPattern)
    if (!match) return null
    return { value: match[1], container, index, kind: 'text', offset: match[0].length }
  }
  if (first.type === 'element') {
    const element = first as Element
    if (isListElement(element)) return null
    const allowed =
      element.tagName === 'em' ||
      element.tagName === 'strong' ||
      element.tagName === 'span' ||
      element.tagName === 'code'
    if (!allowed) return null
    const text = inlineText(element)
    if (!text) return null
    if (!inlineNumberPattern.test(text)) return null
    return { value: text.trim(), container, index, kind: 'element' }
  }
  return null
}

function normalizeLeadingWhitespace(container: Element): void {
  if (!container.children || container.children.length === 0) return
  while (container.children.length > 0) {
    const node = container.children[0]
    if (node.type !== 'text') break
    const textNode = node as Text
    const trimmed = textNode.value.replace(/^\s+/, '')
    if (trimmed.length === 0) {
      container.children.shift()
      continue
    }
    textNode.value = trimmed
    break
  }
}

function stripManualNumber(info: ManualNumberInfo): void {
  const { container, kind } = info
  if (!container.children || container.children.length === 0) return
  if (kind === 'element') {
    container.children.splice(info.index, 1)
    normalizeLeadingWhitespace(container)
    return
  }
  if (kind === 'text') {
    const textNode = container.children[info.index] as Text
    textNode.value = textNode.value.slice(info.offset ?? 0)
    if (textNode.value.length === 0) {
      container.children.splice(info.index, 1)
    }
    normalizeLeadingWhitespace(container)
  }
}

function computeChildNumber(parentNumber: string, index: number): string {
  const segments = parentNumber.split('.')
  const last = segments[segments.length - 1]
  const prefix = segments.slice(0, -1).join('.')
  const suffix = `${last}${index + 1}`
  if (prefix.length === 0) {
    return `${last}.${suffix}`
  }
  return `${prefix}.${suffix}`
}

function wrapTractatusItem(li: Element, numberLabel: string, depth: number): Element[] {
  const props = li.properties ?? (li.properties = {})
  const className = Array.isArray(props.className)
    ? props.className.filter((value): value is string | number => {
        return typeof value === 'string' || typeof value === 'number'
      })
    : typeof props.className === 'string' || typeof props.className === 'number'
      ? [props.className]
      : []
  if (!className.includes('tractatus-item')) className.push('tractatus-item')
  props.className = className
  if (!props.id) {
    const safeId = numberLabel.replace(/[^0-9A-Za-z.-]/g, '')
    props.id = `tractatus-${safeId}`
  }
  const style = typeof props.style === 'string' ? props.style : ''
  const depthStyle = `--tractatus-depth: ${depth};`
  props.style = style ? `${style.trim().replace(/;?$/, ';')} ${depthStyle}` : depthStyle
  props['data-tractatus-number'] = numberLabel
  props['data-tractatus-depth'] = depth

  const nestedLists: Element[] = []
  const contentBlocks: ElementContent[] = []
  for (const child of li.children ?? []) {
    if (child.type === 'element') {
      const element = child as Element
      if (isListElement(element)) {
        nestedLists.push(element)
        continue
      }
    }
    contentBlocks.push(child)
  }

  const row = h('div', { className: 'tractatus-row' }, [
    h('span', { className: 'tractatus-number' }, [{ type: 'text', value: numberLabel }]),
    h('div', { className: 'tractatus-text' }, contentBlocks),
  ])

  li.children = [row, ...nestedLists]
  return nestedLists
}

function processTractatusList(list: Element, parentNumber: string | null, depth: number): void {
  const props = list.properties ?? (list.properties = {})
  const className = Array.isArray(props.className)
    ? props.className.filter((value): value is string | number => {
        return typeof value === 'string' || typeof value === 'number'
      })
    : typeof props.className === 'string' || typeof props.className === 'number'
      ? [props.className]
      : []
  if (!className.includes('tractatus-list')) className.push('tractatus-list')
  props.className = className

  let itemIndex = 0
  for (const child of list.children ?? []) {
    if (!isListItem(child)) continue
    const li = child as Element
    const manualInfo = readManualNumber(li)
    const manualValue = manualInfo?.value ?? null
    const numberLabel = manualValue
      ? manualValue
      : parentNumber
        ? computeChildNumber(parentNumber, itemIndex)
        : String(itemIndex + 1)

    if (manualInfo) {
      stripManualNumber(manualInfo)
    }

    const nestedLists = wrapTractatusItem(li, numberLabel, depth)
    for (const nested of nestedLists) {
      processTractatusList(nested, numberLabel, depth + 1)
    }
    itemIndex += 1
  }
}

export function applyTractatusLayout(root: Root): void {
  const walk = (node: RootContent, inMeta: boolean, inList: boolean): void => {
    if (node.type !== 'element') return
    const element = node as Element
    const nextMeta = inMeta || isMetaSection(element)
    if (isListElement(element)) {
      if (!nextMeta && !inList) {
        processTractatusList(element, null, 0)
      }
      return
    }
    for (const child of element.children ?? []) {
      walk(child, nextMeta, inList || isListElement(element))
    }
  }
  for (const child of root.children ?? []) {
    walk(child, false, false)
  }
}
