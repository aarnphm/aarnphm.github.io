import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "../types"
import { ArenaData } from "../../plugins/transformers/arena"
import { resolveRelative, joinSegments, FullSlug } from "../../util/path"
import { htmlToJsx } from "../../util/jsx"
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

    let globalBlockIndex = 0

    return (
      <article class="arena-index main-col">
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
                    const hasSnapshot = Boolean(block.url)
                    const snapshotSrc =
                      hasSnapshot && block.snapshotKey
                        ? `/api/arena-snapshot?id=${encodeURIComponent(block.snapshotKey)}&source=${encodeURIComponent(
                            block.url!,
                          )}`
                        : undefined
                    const blockDiv = (
                      <div
                        key={block.id}
                        class={classNames(
                          undefined,
                          `arena-channel-row-preview-item`,
                          block.highlighted ? "highlighted" : "",
                        )}
                        data-block-id={block.id}
                        data-block-index={globalBlockIndex}
                        role="button"
                        tabIndex={0}
                      >
                        <div class="arena-channel-row-preview-text">
                          {snapshotSrc ? (
                            <img
                              src={snapshotSrc}
                              loading="lazy"
                              alt={`Snapshot of ${block.title || block.content}`}
                            />
                          ) : block.htmlNode ? (
                            htmlToJsx(fileData.filePath!, block.htmlNode)
                          ) : block.titleHtmlNode ? (
                            htmlToJsx(fileData.filePath!, block.titleHtmlNode)
                          ) : (
                            block.title || block.content
                          )}
                        </div>
                      </div>
                    )
                    globalBlockIndex++
                    return blockDiv
                  })}
                </div>
                <div style="display: none;">
                  {channel.blocks.slice(0, limits).map((block) => {
                    const snapshotSrc =
                      block.url && block.snapshotKey
                        ? `/api/arena-snapshot?id=${encodeURIComponent(block.snapshotKey)}&source=${encodeURIComponent(
                            block.url,
                          )}`
                        : undefined

                    return (
                      <div
                        key={`modal-${block.id}`}
                        class="arena-block-modal-data"
                        id={`arena-modal-data-${block.id}`}
                      >
                        <div class="arena-modal-layout">
                          <div class="arena-modal-main">
                            <div class="arena-modal-main-content">
                              {snapshotSrc ? (
                                <img
                                  src={snapshotSrc}
                                  loading="lazy"
                                  alt={`Snapshot of ${block.title || block.content}`}
                                />
                              ) : block.htmlNode ? (
                                htmlToJsx(fileData.filePath!, block.htmlNode)
                              ) : (
                                block.title || block.content
                              )}
                            </div>
                          </div>
                          <div class="arena-modal-sidebar">
                            <div class="arena-modal-info">
                              <h3 class="arena-modal-title">{block.title || block.content}</h3>
                              {block.url && (
                                <div class="arena-modal-url-bar">
                                  <a
                                    href={block.url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    class="arena-modal-link"
                                  >
                                    <div class="arena-modal-link-text">{block.url}</div>
                                    <svg
                                      width="15"
                                      height="15"
                                      viewBox="0 0 15 15"
                                      fill="none"
                                      xmlns="http://www.w3.org/2000/svg"
                                    >
                                      <path
                                        fill-rule="evenodd"
                                        clip-rule="evenodd"
                                        d="M12 13C12.5523 13 13 12.5523 13 12V3C13 2.44771 12.5523 2 12 2H3C2.44771 2 2 2.44771 2 3V6.5C2 6.77614 2.22386 7 2.5 7C2.77614 7 3 6.77614 3 6.5V3H12V12H8.5C8.22386 12 8 12.2239 8 12.5C8 12.7761 8.22386 13 8.5 13H12ZM9 6.5C9 6.5001 9 6.50021 9 6.50031V6.50035V9.5C9 9.77614 8.77614 10 8.5 10C8.22386 10 8 9.77614 8 9.5V7.70711L2.85355 12.8536C2.65829 13.0488 2.34171 13.0488 2.14645 12.8536C1.95118 12.6583 1.95118 12.3417 2.14645 12.1464L7.29289 7H5.5C5.22386 7 5 6.77614 5 6.5C5 6.22386 5.22386 6 5.5 6H8.5C8.56779 6 8.63244 6.01349 8.69139 6.03794C8.74949 6.06198 8.80398 6.09744 8.85143 6.14433C8.94251 6.23434 8.9992 6.35909 8.99999 6.49708L8.99999 6.49738"
                                        fill="currentColor"
                                      />
                                    </svg>
                                  </a>
                                </div>
                              )}
                            </div>
                            {block.subItems && block.subItems.length > 0 && (
                              <div class="arena-modal-connections">
                                <div class="arena-modal-connections-header">
                                  <span class="arena-modal-connections-title">notes</span>
                                  <span class="arena-modal-connections-count">
                                    {block.subItems.length}
                                  </span>
                                </div>
                                <ul class="arena-modal-connections-list">
                                  {block.subItems.map((subItem) => (
                                    <li key={subItem.id}>
                                      {subItem.titleHtmlNode
                                        ? htmlToJsx(fileData.filePath!, subItem.titleHtmlNode)
                                        : subItem.title || subItem.content}
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            )}
                          </div>
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
