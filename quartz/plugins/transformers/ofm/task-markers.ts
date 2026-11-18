import type { Nodes, Root } from 'mdast'

/** Retain source markers independently of GFM's boolean completion state. */
export function lowerTaskMarkers(tree: Root, source: string): void {
  function visit(node: Nodes) {
    if (node.type === 'listItem' && node.position?.start.offset !== undefined) {
      const authored = source.slice(node.position.start.offset)
      const match = /^(?:[*+-]|\d+[.)])[\t ]+\[([^\]\r\n])\](?=[\t \r\n]|$)/u.exec(authored)
      const marker = match?.[1]
      if (marker !== undefined) {
        if (node.checked == null) {
          const paragraph = node.children[0]
          const text = paragraph?.type === 'paragraph' ? paragraph.children[0] : undefined
          const prefix =
            text?.type === 'text' ? /^\[[^\]\r\n]\](?:[\t ]+|(?=\r?\n|$))/u.exec(text.value) : null
          if (!prefix || text?.type !== 'text' || paragraph?.type !== 'paragraph') return
          text.value = text.value.slice(prefix[0].length)
          if (text.value.length === 0) paragraph.children.shift()
          else if (text.position) {
            text.position.start.column += prefix[0].length
            if (text.position.start.offset !== undefined)
              text.position.start.offset += prefix[0].length
          }
          node.checked = marker.trim().length > 0
        }
        node.data = {
          ...node.data,
          taskMarker: marker,
          hProperties: { ...node.data?.hProperties, 'data-task-marker': marker },
        }
      }
    }
    if ('children' in node) node.children.forEach(visit)
  }
  visit(tree)
}

declare module 'mdast' {
  interface ListItemData {
    taskMarker?: string
  }
}
