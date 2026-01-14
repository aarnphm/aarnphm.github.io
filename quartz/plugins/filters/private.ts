import { QuartzFilterPlugin } from "../../types/plugin"

export const RemovePrivate: QuartzFilterPlugin<{}> = () => ({
  name: "RemovePrivate",
  shouldPublish(ctx, [_tree, vfile]) {
    if (ctx.argv.serve) return true
    const privateFlag = vfile.data?.frontmatter?.private || false
    return !privateFlag
  },
})
