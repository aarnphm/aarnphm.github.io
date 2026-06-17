import type { Comment, Element, Root as HtmlRoot, RootContent } from 'hast'
import { toHtml } from 'hast-util-to-html'

export interface TrainingPlan {
  id: string
  meta: string
  distance: string
  date: string
  target: string
  author: string
  html: string
}

export interface TrainingPayload {
  plans: TrainingPlan[]
}

const START = 'training plan start'
const END = 'training plan end'

const isComment = (n: RootContent): n is Comment => n.type === 'comment'
const isFootnotes = (n: RootContent): n is Element =>
  n.type === 'element' && n.tagName === 'section' && n.properties?.dataFootnotes === ''

const DATE_CELL = /^(?:\d{4}-)?\d{2}-\d{2}(?:\s+to\s+(?:\d{4}-)?\d{2}-\d{2})?$/

function cellText(node: Element): string {
  let out = ''
  for (const child of node.children) {
    if (child.type === 'text') out += child.value
    else if (child.type === 'element') out += cellText(child)
  }
  return out
}

function tagDateCells(nodes: RootContent[]): void {
  for (const node of nodes) {
    if (node.type !== 'element') continue
    if (node.tagName === 'td' || node.tagName === 'th') {
      const text = cellText(node).trim()
      if (DATE_CELL.test(text)) {
        node.properties ??= {}
        node.properties.dataDateType = text.includes(' to ') ? 'range' : 'day'
      }
    }
    tagDateCells(node.children)
  }
}

function parseMeta(value: string): Omit<TrainingPlan, 'id' | 'html'> {
  const fields = { meta: '', distance: '', date: '', target: '', author: '' }
  for (const line of value.split('\n')) {
    const m = /^\s*(meta|distance|date|target|author)\s*:\s*(.+?)\s*$/.exec(line)
    if (m) (fields as Record<string, string>)[m[1]] = m[2]
  }
  return fields
}

export function parseTrainingPlans(tree: HtmlRoot): TrainingPlan[] {
  const kids = tree.children
  const footnotes = kids.find(isFootnotes) ?? null
  const plans: TrainingPlan[] = []
  let i = 0
  while (i < kids.length) {
    const node = kids[i]
    if (!isComment(node) || !node.value.trimStart().startsWith(START)) {
      i++
      continue
    }
    const fields = parseMeta(node.value)
    const body: RootContent[] = []
    let j = i + 1
    while (j < kids.length) {
      const sib = kids[j]
      if (isComment(sib) && sib.value.trim() === END) break
      body.push(sib)
      j++
    }
    if (footnotes && !body.includes(footnotes)) body.push(footnotes)
    tagDateCells(body)
    const html = toHtml({ type: 'root', children: body } as HtmlRoot, { allowDangerousHtml: true })
    plans.push({ id: `plan-${plans.length}`, ...fields, html })
    i = j + 1
  }
  return plans
}
