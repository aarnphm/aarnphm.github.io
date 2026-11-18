import type { Nodes, Root } from 'mdast'

const colors = new Map([
  ['🔴', 'red'],
  ['🟠', 'orange'],
  ['🟡', 'yellow'],
  ['🟢', 'green'],
  ['🔵', 'blue'],
  ['🟣', 'purple'],
])

export function lowerHighlightColors(tree: Root): void {
  function visit(node: Nodes) {
    if (node.type === 'ofmHighlight') {
      const text = node.children[0]
      if (text?.type === 'text') {
        const emoji = Array.from(text.value)[0]
        const color = colors.get(emoji)
        if (color) {
          text.value = text.value.slice(emoji.length)
          if (text.position) {
            text.position.start.column += emoji.length
            if (text.position.start.offset !== undefined) text.position.start.offset += emoji.length
          }
          node.data = {
            ...node.data,
            hProperties: {
              ...node.data?.hProperties,
              className: ['highlight', `highlight-${color}`],
              'data-highlight-color': color,
            },
          }
        }
      }
    }
    if ('children' in node) node.children.forEach(visit)
  }
  visit(tree)
}
