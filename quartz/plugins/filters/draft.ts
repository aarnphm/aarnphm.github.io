import { QuartzFilterPlugin } from '../../types/plugin'

export const RemoveDrafts: QuartzFilterPlugin<{}> = () => ({
  name: 'RemoveDrafts',
  shouldPublish(_ctx, [_tree, vfile]) {
    const draftFlag: boolean =
      vfile.data?.frontmatter?.draft || vfile.data?.frontmatter?.private || false
    return !draftFlag
  },
})
