import type { QuartzPluginData } from '../vfile'
import content from '../../components/styles/protected.scss'
import { QuartzTransformerPlugin } from '../../types/plugin'
import { resolveProtectedPassword } from '../../util/protected'

export const Protected: QuartzTransformerPlugin = () => {
  return {
    name: 'Protected',
    htmlPlugins: ctx => [
      () => {
        return async (_tree, file) => {
          if (ctx.argv.watch && !ctx.argv.force) return

          const frontmatter = file.data.frontmatter
          if (!frontmatter?.protected) return

          file.data.protectedPassword = resolveProtectedPassword(file.data as QuartzPluginData)
        }
      },
    ],
    externalResources() {
      return { css: [{ content, inline: true }] }
    },
  }
}
