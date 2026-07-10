import { Node, Root } from 'hast'
import { Components, Jsx, toJsxRuntime } from 'hast-util-to-jsx-runtime'
import { Fragment, jsx, jsxs } from 'preact/jsx-runtime'
import { getMdxComponentEntries } from '../components/mdx/registry'
import { type FilePath } from './path'
import '../components/mdx'
import { trace } from './trace'

const baseComponents: Record<string, any> = {
  table: (props: any) => (
    <div class="table-container">
      <table {...props} />
    </div>
  ),
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
    return toJsxRuntime(tree as Root, {
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
