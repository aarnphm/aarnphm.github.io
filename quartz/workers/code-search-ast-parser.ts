import {
  Parser as TreeSitterParser,
  type ParseCallback,
  type ParseOptions,
  type Tree,
} from 'web-tree-sitter'
export { Language } from 'web-tree-sitter'

// ast-grep retains its registered parsers. Trees include temporary pattern trees,
// which its Rust wrappers do not delete. A synchronous match owns this allocation set.
const trees = new Set<Tree>()
export class Parser extends TreeSitterParser {
  override parse(
    input: string | ParseCallback,
    oldTree?: Tree | null,
    options?: ParseOptions,
  ): Tree | null {
    const tree = super.parse(input, oldTree, options)
    if (tree) trees.add(tree)
    return tree
  }
}

export function releaseStructuralTrees(): void {
  for (const tree of trees) tree.delete()
  trees.clear()
}
