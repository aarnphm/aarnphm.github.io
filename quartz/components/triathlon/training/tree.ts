import type { Element, RootContent } from 'hast'
import { fromHtml } from 'hast-util-from-html'
import { toHtml } from 'hast-util-to-html'
import { toText } from 'hast-util-to-text'
import type { TrainingPlan } from '../../../plugins/stores/training'

export interface TrainingTreeNode {
  id: string
  label: string
  level: number
  children: TrainingTreeNode[]
}

export interface TrainingDocument {
  html: string
  tree: readonly TrainingTreeNode[]
}

const headingLevel = (node: Element): number | null => {
  const match = /^h([2-4])$/.exec(node.tagName)
  return match ? Number(match[1]) : null
}

export const deriveTrainingDocument = (plan: TrainingPlan): TrainingDocument => {
  const root = fromHtml(plan.html, { fragment: true })
  const headings: Element[] = []
  const visit = (nodes: RootContent[], inFootnotes: boolean): void => {
    for (const node of nodes) {
      if (node.type !== 'element') continue
      const footnotes = inFootnotes || node.properties?.dataFootnotes === ''
      if (!footnotes && headingLevel(node) != null) headings.push(node)
      visit(node.children, footnotes)
    }
  }
  visit(root.children, false)
  const tree: TrainingTreeNode[] = []
  const stack: TrainingTreeNode[] = []
  headings.forEach((heading, index) => {
    const level = headingLevel(heading)
    if (level == null) return
    const id = `tri-h-${plan.id}-${index}`
    heading.properties ??= {}
    heading.properties.id = id
    const node: TrainingTreeNode = { id, label: toText(heading).trim(), level, children: [] }
    while (stack.length > 0 && stack[stack.length - 1].level >= level) stack.pop()
    const siblings = stack.length > 0 ? stack[stack.length - 1].children : tree
    siblings.push(node)
    stack.push(node)
  })
  return { html: toHtml(root, { allowDangerousHtml: true }), tree }
}
