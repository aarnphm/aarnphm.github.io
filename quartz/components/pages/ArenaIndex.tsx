import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "../types"
import { ArenaData } from "../../plugins/transformers/arena"
import { resolveRelative, joinSegments, FullSlug } from "../../util/path"
import style from "../styles/arena.scss"
import { classNames } from "../../util/lang"
import { toArenaHeadingInlineJsx, toArenaJsx, arenaBlockTimestamp } from "../../util/arena"
// @ts-ignore
import modalScript from "../scripts/arena.inline"

export default (() => {
  const ArenaIndex: QuartzComponent = (componentData: QuartzComponentProps) => {
    const { fileData } = componentData
    const arenaData = fileData.arenaData as ArenaData | undefined

    if (!arenaData || !arenaData.channels) {
      return <article class="arena-content">No arena data found</article>
    }

    const arenaBase = "arena" as FullSlug
    const currentSlug = (fileData.slug ?? arenaBase) as FullSlug
    const limits = 5

    const sortedChannels = [...arenaData.channels].sort((a, b) => b.blocks.length - a.blocks.length)

    return (
      <article class="arena-index main-col popover-hint">
        <div class="arena-channels-list">
          {sortedChannels.map((channel) => {
            const channelPath = joinSegments(arenaBase, channel.slug) as FullSlug
            return (
              <div class="arena-channel-row" key={channel.slug} data-slug={channelPath}>
                <div class="arena-channel-row-header">
                  <h2>
                    <a
                      href={resolveRelative(currentSlug, channelPath)}
                      class="internal"
                      data-slug={channelPath}
                      data-no-popover
                    >
                      {channel.titleHtmlNode
                        ? toArenaHeadingInlineJsx(
                            fileData.filePath!,
                            channel.titleHtmlNode,
                            currentSlug,
                            channelPath,
                            componentData,
                          )
                        : channel.name}
                    </a>
                  </h2>
                  <span class="arena-channel-row-count">
                    {channel.blocks.length - limits > 0
                      ? channel.blocks.length - limits
                      : channel.blocks.length}
                  </span>
                </div>
                <div class="arena-channel-row-preview">
                  {[...channel.blocks]
                    .sort((a, b) => arenaBlockTimestamp(b) - arenaBlockTimestamp(a))
                    .slice(0, limits)
                    .map((block) => {
                      return (
                        <div
                          key={block.id}
                          class={classNames(
                            undefined,
                            `arena-channel-row-preview-item`,
                            block.highlighted ? "highlighted" : "",
                          )}
                          data-block-id={block.id}
                          role="button"
                          tabIndex={0}
                        >
                          <div class="arena-channel-row-preview-text">
                            {block.titleHtmlNode
                              ? toArenaJsx(
                                  fileData.filePath!,
                                  block.titleHtmlNode,
                                  currentSlug,
                                  componentData,
                                )
                              : block.title || block.content}
                          </div>
                        </div>
                      )
                    })}
                </div>
              </div>
            )
          })}
        </div>
      </article>
    )
  }

  ArenaIndex.css = style
  ArenaIndex.afterDOMLoaded = modalScript

  return ArenaIndex
}) satisfies QuartzComponentConstructor
