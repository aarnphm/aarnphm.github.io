import { Components, Jsx, toJsxRuntime } from "hast-util-to-jsx-runtime"
import { Node, Root } from "hast"
import { Fragment, jsx, jsxs } from "preact/jsx-runtime"
import { trace } from "./trace"
import { type FilePath } from "./path"
import "../components/mdx"
import { getMdxComponentEntries } from "../components/mdx/registry"

const baseComponents: Record<string, any> = {
  table: (props: any) => (
    <div class="table-container">
      <table {...props} />
    </div>
  ),
}

let cachedComponents: Components | undefined

function resolveComponents(): Components {
  if (!cachedComponents) {
    const mdxEntries = Object.fromEntries(getMdxComponentEntries())
    cachedComponents = {
      ...baseComponents,
      ...mdxEntries,
    } as Components
  }
  return cachedComponents
}

export function htmlToJsx(fp: FilePath, tree: Node) {
  try {
    return toJsxRuntime(tree as Root, {
      Fragment,
      jsx: jsx as Jsx,
      jsxs: jsxs as Jsx,
      elementAttributeNameCase: "html",
      components: resolveComponents(),
    })
  } catch (e) {
    trace(`Failed to parse Markdown in \`${fp}\` into JSX`, e as Error)
  }
}
