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
        <div class="arena-search">
          <input
            type="text"
            id="arena-search-bar"
            class="arena-search-input"
            placeholder="rechercher tous les canaux..."
            data-search-scope="index"
            aria-label="Rechercher tous les canaux"
          />
          <svg
            class="arena-search-icon"
            width="18"
            height="18"
            viewBox="0 0 15 15"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              d="M10 6.5C10 8.433 8.433 10 6.5 10C4.567 10 3 8.433 3 6.5C3 4.567 4.567 3 6.5 3C8.433 3 10 4.567 10 6.5ZM9.30884 10.0159C8.53901 10.6318 7.56251 11 6.5 11C4.01472 11 2 8.98528 2 6.5C2 4.01472 4.01472 2 6.5 2C8.98528 2 11 4.01472 11 6.5C11 7.56251 10.6318 8.53901 10.0159 9.30884L12.8536 12.1464C13.0488 12.3417 13.0488 12.6583 12.8536 12.8536C12.6583 13.0488 12.3417 13.0488 12.1464 12.8536L9.30884 10.0159Z"
              fill="currentColor"
              fill-rule="evenodd"
              clip-rule="evenodd"
            />
          </svg>
          <div id="arena-search-container" class="arena-search-results" />
        </div>
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

        <div class="arena-search-modal" id="arena-search-modal">
          <div class="arena-modal-content">
            <div class="arena-modal-nav">
              <button type="button" class="arena-modal-close" aria-label="Close search">
                <svg
                  width="15"
                  height="15"
                  viewBox="0 0 15 15"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    d="M11.7816 4.03157C12.0062 3.80702 12.0062 3.44295 11.7816 3.2184C11.5571 2.99385 11.193 2.99385 10.9685 3.2184L7.50005 6.68682L4.03164 3.2184C3.80708 2.99385 3.44301 2.99385 3.21846 3.2184C2.99391 3.44295 2.99391 3.80702 3.21846 4.03157L6.68688 7.49999L3.21846 10.9684C2.99391 11.193 2.99391 11.557 3.21846 11.7816C3.44301 12.0061 3.80708 12.0061 4.03164 11.7816L7.50005 8.31316L10.9685 11.7816C11.193 12.0061 11.5571 12.0061 11.7816 11.7816C12.0062 11.557 12.0062 11.193 11.7816 10.9684L8.31322 7.49999L11.7816 4.03157Z"
                    fill="currentColor"
                    fill-rule="evenodd"
                    clip-rule="evenodd"
                  />
                </svg>
              </button>
            </div>
            <div class="arena-search-results-container" />
          </div>
        </div>
      </article>
    )
  }

  ArenaIndex.css = style
  ArenaIndex.afterDOMLoaded = modalScript

  return ArenaIndex
}) satisfies QuartzComponentConstructor
