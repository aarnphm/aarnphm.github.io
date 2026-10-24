import type { Root } from 'mdast'
import { visit } from 'unist-util-visit'
import type { QuartzTransformerPlugin } from '../../types/plugin'
import { anchorCodeBlocks, codeBlockRecords, type PublishedCodeBlock } from './code-search-blocks'

export const CodeSearchBlocks: QuartzTransformerPlugin = () => ({
  name: 'CodeSearchBlocks',
  markdownPlugins() {
    return [
      () => (tree: Root, file) => {
        const notebook = file.data.relativePath?.endsWith('.ipynb')
        if (!notebook) codeBlockRecords(tree, String(file.value))
        const records: PublishedCodeBlock[] = []
        visit(tree, 'code', node => {
          const record = node.data?.codeSearchRecord
          if (!record) return
          records.push(record)
        })
        anchorCodeBlocks(tree)
        file.data.codeSearchBlocks = records
      },
    ]
  },
})
