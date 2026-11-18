import type { Element, Root } from 'hast'
import { toString } from 'hast-util-to-string'
import { h } from 'hastscript'
import { visit } from 'unist-util-visit'
import type { QuartzComponentProps } from '../../types/component'
import type { CompiledQuery, QueryExpression } from '../../util/embedded-search/types'
import { compileEmbeddedQuery } from '../../util/embedded-search/compile'
import { embeddedSearchCorpus } from '../../util/embedded-search/corpus'
import { evaluateEmbeddedQuery } from '../../util/embedded-search/evaluate'
import { isFullSlug, resolveRelative } from '../../util/path'

function isQueryCode(node: Element): boolean {
  const classes = node.properties.className
  return (
    node.tagName === 'code' &&
    ((Array.isArray(classes) && classes.includes('language-query')) ||
      (node.properties.dataLanguage ?? node.properties['data-language']) === 'query')
  )
}

function contentTerms(node: QueryExpression): string[] {
  if (node.kind === 'term') return [node.value]
  if (node.kind === 'and' || node.kind === 'or') return node.children.flatMap(contentTerms)
  if (node.kind === 'scope' && !['file', 'path', 'tag'].includes(node.scope))
    return contentTerms(node.child)
  return []
}

function sourceExcerpt(source: string, expression: QueryExpression): string {
  const text = source.replace(/\s+/g, ' ').trim()
  const lower = text.toLowerCase()
  const indices = contentTerms(expression)
    .map(term => lower.indexOf(term.toLowerCase()))
    .filter(index => index >= 0)
  const index = indices.length ? Math.min(...indices) : 0
  const points = Array.from(text)
  const matched = Array.from(text.slice(0, index)).length
  const start = Math.max(0, matched - 80)
  const end = Math.min(points.length, start + 240)
  return `${start > 0 ? '…' : ''}${points.slice(start, end).join('')}${end < points.length ? '…' : ''}`
}

export function resolveEmbeddedSearch(root: Root, componentData: QuartzComponentProps): void {
  let corpus: ReturnType<typeof embeddedSearchCorpus> | undefined
  let queries = 0
  visit(root, 'element', node => {
    if (node.data?.embeddedSearch === true) queries++
  })
  visit(root, 'element', node => {
    if (node.tagName !== 'pre' && node.tagName !== 'figure') return
    const container =
      node.tagName === 'figure'
        ? node.children.find(child => child.type === 'element' && child.tagName === 'pre')
        : node
    if (!container || container.type !== 'element') return
    const code = container.children.find(child => child.type === 'element' && isQueryCode(child))
    if (!code || code.type !== 'element') return
    const source = toString(code)
    const compiled: CompiledQuery =
      ++queries > 32
        ? { kind: 'error', message: 'page exceeds 32 embedded queries; reduce the query fences' }
        : compileEmbeddedQuery(source)
    const result =
      compiled.kind === 'error'
        ? compiled
        : evaluateEmbeddedQuery(
            compiled.expression,
            (corpus ??= embeddedSearchCorpus(componentData.allFiles)),
          )
    const query = h('details', [h('summary', 'query'), h('pre', [h('code', source)])])
    const children: Element[] =
      result.kind === 'error'
        ? [h('p.embedded-search-error', [h('strong', 'search error: '), result.message]), query]
        : [
            h(
              'p',
              result.documents.length === 0
                ? 'no results'
                : `${result.documents.length}${result.truncated ? '+' : ''} ${result.documents.length === 1 ? 'result' : 'results'}`,
            ),
            ...(result.documents.length
              ? [
                  h(
                    'ul',
                    result.documents.map(document => {
                      const target =
                        isFullSlug(document.slug) && componentData.fileData.slug
                          ? resolveRelative(componentData.fileData.slug, document.slug)
                          : `/${document.slug}`
                      const excerpt =
                        compiled.kind === 'compiled'
                          ? sourceExcerpt(document.text, compiled.expression)
                          : ''
                      return h('li', { 'data-query-result': document.slug }, [
                        h(
                          'a.internal',
                          { href: target, 'data-slug': document.slug },
                          document.title,
                        ),
                        h('p', excerpt),
                      ])
                    }),
                  ),
                ]
              : []),
            ...(result.truncated
              ? [h('p', 'results limited to 100; narrow the query to see the remaining matches')]
              : []),
            query,
          ]
    node.tagName = 'section'
    node.properties = { className: ['embedded-search'], ariaLabel: 'search results' }
    node.data ??= { position: {} }
    node.data.embeddedSearch = true
    node.children = children
  })
}
