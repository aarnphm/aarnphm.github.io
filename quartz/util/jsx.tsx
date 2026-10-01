import type { ElementContent, Node, Root } from 'hast'
import type { JSX } from 'preact'
import { Components, Jsx, toJsxRuntime } from 'hast-util-to-jsx-runtime'
import { Fragment, jsx, jsxs } from 'preact/jsx-runtime'
import { EXIT, visit } from 'unist-util-visit'
import { getMdxComponentEntries } from '../components/mdx/registry'
import { clone } from './clone'
import { type FilePath } from './path'
import '../components/mdx'
import { trace } from './trace'

const componentTableClasses = new Set([
  'tri-act-stats',
  'tri-environment-table',
  'tri-effort-table',
  'tri-health-score-table',
])

const baseComponents = {
  table: (props: JSX.HTMLAttributes<HTMLTableElement>) => {
    const table = <table {...props} />
    const classes = typeof props.class === 'string' ? props.class.split(/\s+/) : []
    if (classes.some(cls => componentTableClasses.has(cls))) return table
    return <div class="table-container">{table}</div>
  },
}

let cachedComponents: Components | undefined
let cachedMdxComponents: Components | undefined

function resolveMdxComponents(): Components {
  cachedMdxComponents ??= Object.fromEntries(getMdxComponentEntries()) as Components
  return cachedMdxComponents
}

function resolveComponents(wrapTables: boolean): Components {
  if (!wrapTables) return resolveMdxComponents()
  cachedComponents ??= { ...baseComponents, ...resolveMdxComponents() } as Components
  return cachedComponents
}

export function htmlToJsx(fp: FilePath, tree: Node, options: { wrapTables?: boolean } = {}) {
  try {
    const sourceTree = tree as Root
    let renderTree = sourceTree
    visit(sourceTree, 'element', node => {
      if (node.tagName !== 'template' || !node.content) return
      renderTree = clone(sourceTree)
      return EXIT
    })
    if (renderTree !== sourceTree)
      visit(renderTree, 'element', node => {
        if (node.tagName !== 'template' || !node.content) return
        // HAST stores template content separately; the JSX converter reads children.
        node.children = node.content.children.filter(
          (child): child is ElementContent => child.type !== 'doctype' && child.type !== 'mdxjsEsm',
        )
        delete node.content
      })
    return toJsxRuntime(renderTree, {
      Fragment,
      jsx: jsx as Jsx,
      jsxs: jsxs as Jsx,
      elementAttributeNameCase: 'html',
      components: resolveComponents(options.wrapTables ?? true),
    })
  } catch (e) {
    trace(`Failed to parse Markdown in \`${fp}\` into JSX`, e as Error)
    return undefined
  }
}
