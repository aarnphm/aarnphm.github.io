import type { Element } from 'hast'

/**
 * A ```poetry (or ```poem) fence's lines as `pre.poetry` holds them: text as written, `~~struck~~`
 * runs as `del`, and a line opening with `+ ` or `- ` as a `span.diff-add` / `span.diff-del`.
 * Shared by the Aarnphm transformer and the /embeddings reader.
 */
export function strikethroughChildren(value: string): Element['children'] {
  if (!value) return []
  const markers: number[] = []
  for (let index = 0; index < value.length - 1; ) {
    if (value[index] === '~' && value[index + 1] === '~') {
      markers.push(index)
      index += 2
    } else index++
  }
  if (markers.length < 2) return [{ type: 'text', value }]
  const children: Element['children'] = []
  let cursor = 0
  for (let index = 0; index < markers.length - (markers.length % 2); index += 2) {
    const start = markers[index]
    const end = markers[index + 1]
    const before = value.slice(cursor, start)
    if (before) children.push({ type: 'text', value: before })
    const struck = value.slice(start + 2, end)
    if (struck)
      children.push({
        type: 'element',
        tagName: 'del',
        properties: {},
        children: [{ type: 'text', value: struck }],
      })
    cursor = end + 2
  }
  const tail = value.slice(cursor)
  if (tail) children.push({ type: 'text', value: tail })
  return children.length === 0 ? [{ type: 'text', value: '' }] : children
}

export function poetryChildren(value: string): Element['children'] {
  const lines = value.split('\n')
  const children: Element['children'] = []
  for (const [index, line] of lines.entries()) {
    const diff = line.match(/^([+-])\s/)
    const content = strikethroughChildren(diff ? ` ${line.slice(2)}` : line)
    if (diff)
      children.push({
        type: 'element',
        tagName: 'span',
        properties: { className: [diff[1] === '+' ? 'diff-add' : 'diff-del'] },
        children: content.length === 0 ? [{ type: 'text', value: '' }] : content,
      })
    else children.push(...content)
    if (index < lines.length - 1) children.push({ type: 'text', value: '\n' })
  }
  return children.length === 0 ? [{ type: 'text', value: '' }] : children
}
