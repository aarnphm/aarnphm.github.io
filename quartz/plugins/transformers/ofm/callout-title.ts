import type { Paragraph, PhrasingContent } from 'mdast'

type Split = { title: PhrasingContent[]; body: PhrasingContent[]; found: boolean }

/** Separate the physical title line without inserting bytes before Markdown parsing. */
export function splitCalloutParagraph(paragraph: Paragraph): Split {
  return splitChildren(paragraph.children)
}

function splitChildren(children: PhrasingContent[]): Split {
  const title: PhrasingContent[] = []
  const body: PhrasingContent[] = []
  let found = false
  for (const child of children) {
    if (found) {
      body.push(child)
    } else if (child.type === 'break') {
      found = true
    } else if (child.type === 'text' && child.value.includes('\n')) {
      const newline = child.value.indexOf('\n')
      const head = child.value.slice(0, newline).replace(/\r$/, '')
      const tail = child.value.slice(newline + 1)
      if (head) title.push({ ...child, value: head, position: undefined })
      if (tail) body.push({ ...child, value: tail, position: undefined })
      found = true
    } else if (
      child.type === 'emphasis' ||
      child.type === 'strong' ||
      child.type === 'delete' ||
      child.type === 'link' ||
      child.type === 'linkReference'
    ) {
      const split = splitChildren(child.children)
      if (!split.found) title.push(child)
      else {
        if (split.title.length) title.push({ ...child, children: split.title, position: undefined })
        if (split.body.length) body.push({ ...child, children: split.body, position: undefined })
        found = true
      }
    } else {
      title.push(child)
    }
  }
  return { title, body, found }
}
