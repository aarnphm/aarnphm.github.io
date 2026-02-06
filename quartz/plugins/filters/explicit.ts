import { QuartzFilterPlugin } from '../../types/plugin'

export const ExplicitPublish: QuartzFilterPlugin = () => ({
  name: 'ExplicitPublish',
  shouldPublish(_ctx, [_tree, vfile]) {
    return vfile.data?.frontmatter?.publish ?? false
  },
})
