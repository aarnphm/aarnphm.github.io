import { Root } from 'mdast'
import { QuartzTransformerPlugin } from '../../types/plugin'
import { compileBaseConfig } from '../../util/base/compile'
import { BaseExpressionDiagnostic, BasesExpressions } from '../../util/base/compiler'
import { BaseFile } from '../../util/base/types'

export const ObsidianBases: QuartzTransformerPlugin = () => {
  return {
    name: 'ObsidianBases',
    externalResources() {
      return {
        css: [
          {
            content: 'https://unpkg.com/lucide-static@latest/font/lucide.css',
            classPrefixes: ['icon-', 'base-view'],
          },
        ],
      }
    },
    markdownPlugins() {
      return [
        () => {
          return async (tree: Root, file) => {
            // Detect .base files by extension
            const isBaseFile = file.path?.endsWith('.base')

            if (!isBaseFile) {
              file.data.bases = false
              return
            }

            file.data.bases = true

            const source = String(file.value)
            const compiled = compileBaseConfig(source, file.path)
            const config: BaseFile = compiled.config

            file.data.basesConfig = config
            file.data.basesSource = source
            file.data.basesDiagnostics = compiled.diagnostics
            file.data.basesExpressions = compiled.expressions

            tree.children = []

            file.data.frontmatter = {
              title: file.path?.replace('.base', '').split('/').pop() || '',
              pageLayout: 'default' as const,
              tags: ['bases'],
            }
          }
        },
      ]
    },
  }
}

declare module 'vfile' {
  interface DataMap {
    bases?: boolean
    basesConfig?: BaseFile
    basesSource?: string
    basesDiagnostics?: BaseExpressionDiagnostic[]
    basesExpressions?: BasesExpressions
  }
}
