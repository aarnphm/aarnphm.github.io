import type { Element, ElementContent, Root as HtmlRoot } from 'hast'
import type { Root } from 'mdast'
import { toString } from 'mdast-util-to-string'
import { visit } from 'unist-util-visit'
import type { QuartzTransformerPlugin } from '../../types/plugin'
import {
  BOLD_LEAD,
  STATEMENT_KINDS,
  STATEMENT_SOURCE,
  statementRowId,
  termAnchor,
} from '../../util/course-statements'
import { FullSlug, resolveRelative } from '../../util/path'

const statementsSlug = (dir: string) => `courses/${dir}/statements/index` as FullSlug

function rowLink(from: FullSlug, dir: string, rowId: string, children: ElementContent[]): Element {
  const slug = statementsSlug(dir)
  return {
    type: 'element',
    tagName: 'a',
    properties: {
      href: `${resolveRelative(from, slug)}#${rowId}`,
      className: ['internal', 'course-statement-link'],
      dataSlug: slug,
      title: 'this statement on the statement index',
    },
    children,
  }
}

const isPageLink = (node: ElementContent) =>
  node.type === 'element' &&
  node.tagName === 'p' &&
  node.children.length === 1 &&
  node.children[0].type === 'element' &&
  node.children[0].tagName === 'a' &&
  /^p\. \d+/.test(
    node.children[0].children.map(child => (child.type === 'text' ? child.value : '')).join(''),
  )

/**
 * Links each statement of a course note to its row on `courses/<dir>/statements/`, which links
 * back: numbered callouts in `notes/<nn>` get the link beside their `p. N` source link; bold-lead
 * terms in `lectures/<nn>` get an anchor and their lead becomes the link.
 */
export const CourseStatements: QuartzTransformerPlugin = () => ({
  name: 'CourseStatements',
  markdownPlugins: ctx => [
    () => (tree: Root, file) => {
      const match = STATEMENT_SOURCE.exec(file.data.slug ?? '')
      if (!match || match[2] !== 'lectures') return
      if (!ctx.allSlugs.includes(statementsSlug(match[1]))) return
      // the same leads the statement index reads from the raw source
      const leads = new Set(
        Array.from(String(file.value).matchAll(BOLD_LEAD), lead => termAnchor(lead[1])),
      )
      visit(tree, 'paragraph', node => {
        const lead = node.children[0]
        if (lead?.type !== 'strong') return
        const text = toString(lead)
        if (!text.endsWith('.')) return
        const anchor = termAnchor(text.slice(0, -1))
        if (!leads.has(anchor)) return
        node.data = { ...node.data, hProperties: { ...node.data?.hProperties, id: anchor } }
        lead.data = {
          ...lead.data,
          hProperties: { ...lead.data?.hProperties, dataStatementRow: anchor },
        }
      })
    },
  ],
  htmlPlugins: ctx => [
    () => (tree: HtmlRoot, file) => {
      const from = file.data.slug
      const match = STATEMENT_SOURCE.exec(from ?? '')
      if (!from || !match || !ctx.allSlugs.includes(statementsSlug(match[1]))) return
      const [, dir, , n] = match
      const unit = Number(n)
      visit(tree, 'element', (node, index, parent) => {
        const term = node.properties.dataStatementRow
        if (
          node.tagName === 'strong' &&
          typeof term === 'string' &&
          parent &&
          index !== undefined
        ) {
          delete node.properties.dataStatementRow
          parent.children[index] = rowLink(from, dir, statementRowId(unit, term), [node])
          return 'skip'
        }
        const kind = node.properties.dataCallout
        const id = node.properties.id
        if (
          node.tagName !== 'blockquote' ||
          typeof kind !== 'string' ||
          typeof id !== 'string' ||
          !STATEMENT_KINDS.has(kind) ||
          !id.startsWith(`${kind}-`)
        )
          return
        const content = node.children.find(
          (child): child is Element =>
            child.type === 'element' &&
            String(child.properties.className ?? '').includes('callout-content'),
        )
        if (!content) return
        const link = rowLink(from, dir, statementRowId(unit, id), [
          { type: 'text', value: 'statements' },
        ])
        const last = content.children.findLast(child => child.type === 'element')
        if (last && isPageLink(last))
          (last as Element).children.push({ type: 'text', value: ' · ' }, link)
        else
          content.children.push({ type: 'element', tagName: 'p', properties: {}, children: [link] })
      })
    },
  ],
})
