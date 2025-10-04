import { QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { classNames } from "../util/lang"
import { resolveRelative, FullSlug } from "../util/path"
import { ArenaData, ArenaChannel } from "../plugins/transformers/arena"
import { toArenaHeadingJsx } from "../util/arena"

export default (() => (componentData: QuartzComponentProps) => {
  const { fileData, displayClass } = componentData
  const title = fileData.frontmatter?.title
  const slug = fileData.slug!
  const isArenaIndex = slug === "arena"
  const isArenaChannel = slug.startsWith("arena/") && slug !== "arena"

  if (isArenaIndex) {
    const arenaData = fileData.arenaData as ArenaData | undefined

    return (
      <hgroup class={classNames(displayClass, "title-col", "arena-title-block")} data-article-title>
        <h1 class="article-title">are.na</h1>
        <p class="description">
          {arenaData
            ? `${arenaData.channels.length} channels · ${arenaData.channels.reduce((sum, ch) => sum + ch.blocks.length, 0)} blocks`
            : ""}
        </p>
      </hgroup>
    )
  }

  if (isArenaChannel) {
    const channel = fileData.arenaChannel as ArenaChannel | undefined
    const arenaRootSlug = "arena" as FullSlug

    return (
      <hgroup class={classNames(displayClass, "title-col", "arena-title-block")} data-article-title>
        <h1 class="article-title">
          <a
            href={resolveRelative(slug, arenaRootSlug)}
            class="internal"
            data-no-popover
            data-slug={arenaRootSlug}
          >
            are.na
          </a>
          {" / "}
          {channel?.titleHtmlNode
            ? toArenaHeadingJsx(
                fileData.filePath!,
                channel.titleHtmlNode,
                fileData.slug! as FullSlug,
                `arena/${channel.slug}` as FullSlug,
                componentData,
              )
            : channel?.name || title}
        </h1>
        <p class="description">{channel ? `${channel.blocks.length} blocks` : ""}</p>
      </hgroup>
    )
  }

  if (title) {
    return (
      <hgroup class={classNames(displayClass, "title-col")} data-article-title>
        <h1 class="article-title">{title}</h1>
        <p class="description">{fileData.description}</p>
      </hgroup>
    )
  }

  return <></>
}) satisfies QuartzComponentConstructor
