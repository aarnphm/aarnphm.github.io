import { Element, Text as HastText } from 'hast'
import { h } from 'hastscript'
import { Root, Link, Text } from 'mdast'
import rehypeCitation from 'rehype-citation'
import { visit } from 'unist-util-visit'
import { QuartzTransformerPlugin } from '../../types/plugin'
import { cacheState, makeBibKey, normalizeArxivId } from '../stores/citations'
import '@citation-js/plugin-bibtex'
import '@citation-js/plugin-doi'
import { extractArxivId } from './links'

const URL_PATTERN = /https?:\/\/[^\s<>)"]+/g

interface LinkType {
  type: string
  pattern: (url: string) => boolean | string | null
  label: string
}

const LINK_TYPES: LinkType[] = [
  { type: 'arxiv', pattern: extractArxivId, label: '[arXiv]' },
  {
    type: 'lesswrong',
    pattern: (url: string) => url.toLowerCase().includes('lesswrong.com'),
    label: '[lesswrong]',
  },
  {
    type: 'github',
    pattern: (url: string) => url.toLowerCase().includes('github.com'),
    label: '[GitHub]',
  },
  {
    type: 'transformer',
    pattern: (url: string) => url.toLowerCase().includes('transformer-circuits.pub'),
    label: '[transformer circuit]',
  },
  {
    type: 'alignment',
    pattern: (url: string) => url.toLowerCase().includes('alignmentforum.org'),
    label: '[alignment forum]',
  },
]

function createTextNode(value: string): HastText {
  return { type: 'text', value }
}

function getLinkType(url: string): LinkType | undefined {
  return LINK_TYPES.find(type => type.pattern(url))
}

function createLinkElement(href: string): Element {
  const linkType = getLinkType(href)
  const displayText = linkType ? linkType.label : href

  return h(
    'a.csl-external-link',
    { href, target: '_blank', rel: 'noopener noreferrer' },
    createTextNode(displayText),
  )
}

function processTextNode(node: HastText): (Element | HastText)[] {
  const text = node.value
  const matches = Array.from(text.matchAll(URL_PATTERN))

  if (matches.length === 0) {
    return [node]
  }

  const result: (Element | HastText)[] = []
  let lastIndex = 0

  matches.forEach(match => {
    const href = match[0]
    const startIndex = match.index!

    if (startIndex > lastIndex) {
      result.push(createTextNode(text.slice(lastIndex, startIndex)))
    }

    const arxivId = extractArxivId(href)
    if (arxivId) {
      result.push(createTextNode(`arXiv preprint arXiv:${arxivId} `))
    }

    result.push(createLinkElement(href))
    lastIndex = startIndex + href.length
  })

  if (lastIndex < text.length) {
    result.push(createTextNode(text.slice(lastIndex)))
  }

  return result
}

function processNodes(nodes: (Element | HastText)[]): (Element | HastText)[] {
  return nodes.flatMap(node => {
    if (node.type === 'text') {
      return processTextNode(node)
    }
    if (node.type === 'element') {
      return { ...node, children: processNodes(node.children as (Element | HastText)[]) }
    }
    return [node]
  })
}

export const checkBib = ({ tagName, properties }: Element) =>
  tagName === 'a' && typeof properties?.href === 'string' && properties.href.startsWith('#bib')

export const checkBibSection = ({ type, tagName, properties }: Element) =>
  type === 'element' && tagName === 'section' && properties.dataReferences == ''

interface Options {
  bibliography: string
}

declare module 'vfile' {
  interface DataMap {
    citations?: { arxivIds: string[] }
    citationsDisabled?: boolean
  }
}

export const Citations: QuartzTransformerPlugin<Options> = (opts?: Options) => {
  const bibliography = opts?.bibliography ?? 'content/References.bib'
  return {
    name: 'Citations',
    markdownPlugins: () => [
      () => (tree: Root, file: any) => {
        const frontmatter = file.data?.frontmatter ?? {}
        const disableCitations = frontmatter.citations === false || frontmatter.noCitations === true
        if (disableCitations) {
          file.data.citationsDisabled = true
          delete file.data.citations
          return
        }
        file.data.citationsDisabled = false
        const arxivNodes: { node: Link; index: number; parent: any; id: string }[] = []

        visit(tree, 'link', (node: Link, index: number | undefined, parent: any) => {
          if (index === undefined || !parent) return
          const arxivId = extractArxivId(node.url)
          if (!arxivId) return
          arxivNodes.push({ node, index, parent, id: normalizeArxivId(arxivId) })
        })

        const docIds = Array.from(new Set(arxivNodes.map(entry => entry.id))).sort()
        if (docIds.length > 0) {
          file.data.citations = { arxivIds: docIds }
        } else {
          delete file.data.citations
        }

        if (arxivNodes.length === 0) return

        for (const id of docIds) {
          if (!cacheState.papers.has(id)) {
            cacheState.papers.set(id, {
              title: id,
              bibkey: makeBibKey(id),
              lastVerified: 0,
              inBibFile: false,
            })
            cacheState.dirty = true
          }
        }

        for (const { node, index, parent, id } of arxivNodes) {
          const entry = cacheState.papers.get(id)
          if (!entry) continue

          node.children = [{ type: 'text', value: entry.title } as Text]
          parent.children.splice(index, 1, node, {
            type: 'text',
            value: ` [@${entry.bibkey}] `,
          } as Text)
        }
      },
    ],
    htmlPlugins: ({ cfg }) => [
      [
        rehypeCitation,
        {
          bibliography,
          suppressBibliography: false,
          linkCitations: true,
          csl: 'apa',
          lang:
            cfg.configuration.locale !== 'en-US'
              ? `https://raw.githubusercontent.com/citation-style-language/locales/refs/heads/master/locales-${cfg.configuration.locale}.xml`
              : 'en-US',
        },
      ],
      () => (tree, file: any) => {
        if (file?.data?.citationsDisabled) return
        visit(
          tree,
          node => checkBib(node as Element),
          (node, _index, parent) => {
            node.properties['data-bib'] = true
            parent.tagName = 'cite'
          },
        )
      },
      () => (tree, file: any) => {
        if (file?.data?.citationsDisabled) return
        visit(
          tree,
          node => {
            const className = (node as Element).properties?.className
            return Array.isArray(className) && className.includes('references')
          },
          (node, index, parent) => {
            const entries: Element[] = []
            visit(
              node,
              node => {
                const className = (node as Element).properties?.className
                return Array.isArray(className) && className.includes('csl-entry')
              },
              node => {
                const { properties, children } = node as Element
                entries.push(h('li', properties, processNodes(children as Element[])))
              },
            )

            parent!.children.splice(
              index!,
              1,
              h(
                'section.bibliography',
                { dataReferences: true },
                h('h2#reference-label', [{ type: 'text', value: 'bibliographie' }]),
                h('ul', ...entries),
              ),
            )
          },
        )
      },
    ],
  }
}
