import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "../types"
import { ArenaData } from "../../plugins/transformers/arena"
import { resolveRelative, joinSegments, FullSlug } from "../../util/path"
import style from "../styles/arena.scss"
import { classNames } from "../../util/lang"

export default (() => {
  const ArenaIndex: QuartzComponent = ({ fileData }: QuartzComponentProps) => {
    const arenaData = fileData.arenaData as ArenaData | undefined

    if (!arenaData || !arenaData.channels) {
      return <article class="arena-content">No arena data found</article>
    }

    const arenaBase = "arena" as FullSlug
    const currentSlug = (fileData.slug ?? arenaBase) as FullSlug
    const limits = 5

    return (
      <article class="arena-index main-col popover-hint">
        <div class="arena-channels-list">
          {arenaData.channels.map((channel) => {
            const channelPath = joinSegments(arenaBase, channel.slug) as FullSlug
            return (
              <a
                href={resolveRelative(currentSlug, channelPath)}
                class="arena-channel-row internal"
                data-slug={channelPath}
                data-no-popover
                key={channel.slug}
              >
                <div class="arena-channel-row-header">
                  <h2>{channel.name}</h2>
                  <span class="arena-channel-row-count">
                    {channel.blocks.length - limits > 0
                      ? channel.blocks.length - limits
                      : channel.blocks.length}
                  </span>
                </div>
                <div class="arena-channel-row-preview">
                  {channel.blocks.slice(0, limits).map((block) => {
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
                          {block.title || block.content}
                        </div>
                      </div>
                    )
                  })}
                </div>
              </a>
            )
          })}
        </div>
      </article>
    )
  }

  ArenaIndex.css = style

  return ArenaIndex
}) satisfies QuartzComponentConstructor
