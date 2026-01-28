import type { ComponentChild } from "preact"
import { ArenaData, ArenaBlock } from "../../plugins/transformers/arena"
import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from "../../types/component"
import { toArenaHeadingInlineJsx, toArenaJsx, arenaBlockTimestamp } from "../../util/arena"
import { classNames } from "../../util/lang"
import { resolveRelative, joinSegments, FullSlug } from "../../util/path"
import { extractWikilinksWithPositions, resolveWikilinkTarget } from "../../util/wikilinks"
// @ts-ignore
import script from "../scripts/arena.inline"
import style from "../styles/arena.scss"

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

    const renderInlineText = (text: string) => {
      if (!text) return ""
      const parts: ComponentChild[] = []
      const ranges = extractWikilinksWithPositions(text)
      let lastIndex = 0
      for (const range of ranges) {
        const start = range.start
        if (start > lastIndex) {
          parts.push(text.slice(lastIndex, start))
        }

        const parsed = range.wikilink
        const resolved = resolveWikilinkTarget(parsed, "" as FullSlug)
        const raw = text.slice(range.start, range.end)

        if (resolved) {
          const hrefBase = resolveRelative(fileData.slug! as FullSlug, resolved.slug)
          const href = parsed.anchor ? `${hrefBase}${parsed.anchor}` : hrefBase
          parts.push(
            <a
              href={href}
              class="internal"
              data-no-popover
              data-slug={resolved.slug}
              key={`arena-wikilink-${parts.length}`}
            >
              {parsed.alias ?? parsed.target ?? raw}
            </a>,
          )
        } else {
          parts.push(parsed.alias ?? parsed.target ?? raw)
        }

        lastIndex = range.end
      }

      if (lastIndex < text.length) {
        parts.push(text.slice(lastIndex))
      }

      if (parts.length === 1) {
        return parts[0]
      }

      return <>{parts}</>
    }

    // Collect all preview blocks for modal data
    const allPreviewBlocks: Array<{ block: ArenaBlock; channelSlug: string }> = []
    sortedChannels.forEach((channel) => {
      const previewBlocks = [...channel.blocks]
        .sort((a, b) => arenaBlockTimestamp(b) - arenaBlockTimestamp(a))
        .slice(0, limits)
      previewBlocks.forEach((block) => {
        allPreviewBlocks.push({ block, channelSlug: channel.slug })
      })
    })

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
            aria-keyshortcuts="Meta+K Control+K"
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
                        : renderInlineText(channel.name)}
                    </a>
                  </h2>
                  <div class="arena-channel-row-metadata">
                    {(channel.metadata?.json === "true" || channel.metadata?.json === true) && (
                      <a
                        href={resolveRelative(currentSlug, `${channelPath}/json` as FullSlug)}
                        class="arena-channel-json-link"
                        title="JSON export available"
                        data-no-popover
                      >
                        <span>{"{ }"}</span>
                      </a>
                    )}
                    <span class="arena-channel-row-count">
                      {channel.blocks.length - limits > 0
                        ? channel.blocks.length - limits
                        : channel.blocks.length}
                    </span>
                  </div>
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
                              : renderInlineText(block.title || block.content || "")}
                          </div>
                        </div>
                      )
                    })}
                </div>
              </div>
            )
          })}
        </div>

        <div class="arena-block-modal" id="arena-modal">
          <div class="arena-modal-content">
            <div class="arena-modal-nav">
              <button type="button" class="arena-modal-nav-btn arena-modal-collapse">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="15"
                  height="15"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  stroke-width="2"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                >
                  <line x1="4" x2="20" y1="12" y2="12" />
                  <line x1="4" x2="20" y1="6" y2="6" />
                  <line x1="4" x2="20" y1="18" y2="18" />
                </svg>
              </button>
              <button
                type="button"
                class="arena-modal-nav-btn arena-modal-prev"
                aria-label="Previous block"
              >
                <svg
                  width="15"
                  height="15"
                  viewBox="0 0 15 15"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    d="M8.84182 3.13514C9.04327 3.32401 9.05348 3.64042 8.86462 3.84188L5.43521 7.49991L8.86462 11.1579C9.05348 11.3594 9.04327 11.6758 8.84182 11.8647C8.64036 12.0535 8.32394 12.0433 8.13508 11.8419L4.38508 7.84188C4.20477 7.64955 4.20477 7.35027 4.38508 7.15794L8.13508 3.15794C8.32394 2.95648 8.64036 2.94628 8.84182 3.13514Z"
                    fill="currentColor"
                    fill-rule="evenodd"
                    clip-rule="evenodd"
                  />
                </svg>
              </button>
              <button
                type="button"
                class="arena-modal-nav-btn arena-modal-next"
                aria-label="Next block"
              >
                <svg
                  width="15"
                  height="15"
                  viewBox="0 0 15 15"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    d="M6.1584 3.13508C6.35985 2.94621 6.67627 2.95642 6.86514 3.15788L10.6151 7.15788C10.7954 7.3502 10.7954 7.64949 10.6151 7.84182L6.86514 11.8418C6.67627 12.0433 6.35985 12.0535 6.1584 11.8646C5.95694 11.6757 5.94673 11.3593 6.1356 11.1579L9.565 7.49985L6.1356 3.84182C5.94673 3.64036 5.95694 3.32394 6.1584 3.13508Z"
                    fill="currentColor"
                    fill-rule="evenodd"
                    clip-rule="evenodd"
                  />
                </svg>
              </button>
              <button type="button" class="arena-modal-close" aria-label="Close">
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
            <div class="arena-modal-body" />
          </div>
        </div>
      </article>
    )
  }

  ArenaIndex.css = style
  ArenaIndex.afterDOMLoaded = script

  return ArenaIndex
}) satisfies QuartzComponentConstructor
