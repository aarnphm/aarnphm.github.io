import type { ComponentChild } from 'preact'
import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../../types/component'
import { classNames } from '../../util/lang'
import {
  ARENA_CARD_PAGE_SIZE,
  arenaChannelAssets,
  type ArenaSectionName,
} from '../arena/channel-data'
import { arenaChannelSections, createArenaChannelRenderer } from '../arena/ChannelBlock'
import { ArenaReaderLink } from '../arena/reader-link'
import style from '../styles/arena.scss'

const ArenaChannelSection = ({
  title,
  children,
}: {
  title: 'pinned' | 'later'
  children: ComponentChild
}) => (
  <details class="arena-channel-section" open>
    <summary class="arena-section-header">
      <h3>{title}</h3>
      <svg
        class="arena-section-chevron"
        width="12"
        height="12"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="1.5"
        stroke-linecap="round"
        stroke-linejoin="round"
        aria-hidden="true"
      >
        <path d="m9 5 7 7-7 7" />
      </svg>
    </summary>
    {children}
  </details>
)

export default (() => {
  const ChannelContent: QuartzComponent = (componentData: QuartzComponentProps) => {
    const { fileData, displayClass } = componentData
    const channel = fileData.arenaChannel

    if (!channel) {
      return <article class="arena-content">Channel not found</article>
    }

    const {
      pinned: pinnedBlocks,
      later: laterBlocks,
      blocks: regularBlocks,
    } = arenaChannelSections(channel)
    const renderBlock = createArenaChannelRenderer(componentData, channel)
    const assetBase = arenaChannelAssets(channel.slug)
    const blockOrder = [...pinnedBlocks, ...laterBlocks, ...regularBlocks].map(block => block.id)
    const channelViewPreferenceRaw = channel.metadata?.['view'] ?? channel.metadata?.['layout']
    const channelViewPreference =
      typeof channelViewPreferenceRaw === 'string'
        ? channelViewPreferenceRaw.trim().toLowerCase()
        : undefined
    const defaultViewMode: 'grid' | 'list' =
      channelViewPreference === 'list' || channelViewPreference === 'lists' ? 'list' : 'grid'
    const isListDefault = defaultViewMode === 'list'

    const renderSection = (
      name: ArenaSectionName,
      blocks: typeof channel.blocks,
      startIndex: number,
      view: 'grid' | 'list',
    ) => (
      <div
        class={`arena-channel-grid arena-${name}-section`}
        id={name === 'blocks' ? 'arena-block-collection' : undefined}
        data-arena-grid={name}
        data-arena-count={blocks.length}
        data-arena-start={startIndex}
        data-view-mode={view}
        data-view-default={view}
      >
        {blocks
          .slice(0, ARENA_CARD_PAGE_SIZE)
          .map((block, index) => renderBlock(block, startIndex + index))}
      </div>
    )

    return (
      <article
        class="arena-channel-page main-col"
        data-view-mode={defaultViewMode}
        data-view-default={defaultViewMode}
        data-arena-assets={assetBase}
        data-arena-block-order={JSON.stringify(blockOrder)}
      >
        <div class="arena-channel-controls">
          <div class="arena-search-row">
            <div class="arena-search">
              <input
                type="text"
                id="arena-search-bar"
                class="arena-search-input"
                placeholder="rechercher ce canal..."
                data-search-scope="channel"
                data-channel-slug={channel.slug}
                aria-label="Rechercher ce canal"
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
            <ArenaReaderLink />
          </div>
          <div class="arena-view-toggle" role="group" aria-label="Toggle channel layout">
            <button
              type="button"
              class={classNames(
                displayClass,
                'arena-view-toggle-button',
                !isListDefault ? 'active' : '',
              )}
              data-view-mode="grid"
              aria-label="Grid view"
              aria-pressed={!isListDefault}
            >
              <svg
                width="16"
                height="16"
                viewBox="0 0 15 15"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
                aria-hidden="true"
              >
                <path
                  fill-rule="evenodd"
                  clip-rule="evenodd"
                  d="M1.5 3A1.5 1.5 0 0 1 3 1.5h2A1.5 1.5 0 0 1 6.5 3v2A1.5 1.5 0 0 1 5 6.5H3A1.5 1.5 0 0 1 1.5 5V3Zm1 0A.5.5 0 0 1 3 2.5h2a.5.5 0 0 1 .5.5v2a.5.5 0 0 1-.5.5H3a.5.5 0 0 1-.5-.5V3ZM8.5 3A1.5 1.5 0 0 1 10 1.5h2A1.5 1.5 0 0 1 13.5 3v2A1.5 1.5 0 0 1 12 6.5h-2A1.5 1.5 0 0 1 8.5 5V3Zm1 0a.5.5 0 0 1 .5-.5h2a.5.5 0 0 1 .5.5v2a.5.5 0 0 1-.5.5h-2a.5.5 0 0 1-.5-.5V3ZM1.5 10a1.5 1.5 0 0 1 1.5-1.5h2A1.5 1.5 0 0 1 6.5 10v2A1.5 1.5 0 0 1 5 13.5H3A1.5 1.5 0 0 1 1.5 12v-2Zm1 0a.5.5 0 0 1 .5-.5h2a.5.5 0 0 1 .5.5v2a.5.5 0 0 1-.5.5H3a.5.5 0 0 1-.5-.5v-2ZM8.5 10A1.5 1.5 0 0 1 10 8.5h2a1.5 1.5 0 0 1 1.5 1.5v2a1.5 1.5 0 0 1-1.5 1.5h-2A1.5 1.5 0 0 1 8.5 12v-2Zm1 0a.5.5 0 0 1 .5-.5h2a.5.5 0 0 1 .5.5v2a.5.5 0 0 1-.5.5h-2a.5.5 0 0 1-.5-.5v-2Z"
                  fill="currentColor"
                />
              </svg>
            </button>
            <button
              type="button"
              class={classNames(
                displayClass,
                'arena-view-toggle-button',
                isListDefault ? 'active' : '',
              )}
              data-view-mode="list"
              aria-label="List view"
              aria-pressed={isListDefault}
            >
              <svg
                width="16"
                height="16"
                viewBox="0 0 15 15"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
                aria-hidden="true"
              >
                <path
                  fill-rule="evenodd"
                  clip-rule="evenodd"
                  d="M2 4.25A.75.75 0 0 1 2.75 3.5h9.5a.75.75 0 0 1 0 1.5h-9.5A.75.75 0 0 1 2 4.25Zm0 3.5A.75.75 0 0 1 2.75 7h9.5a.75.75 0 0 1 0 1.5h-9.5A.75.75 0 0 1 2 7.75Zm.75 3.5a.75.75 0 0 0 0 1.5h9.5a.75.75 0 0 0 0-1.5h-9.5Z"
                  fill="currentColor"
                />
              </svg>
            </button>
          </div>
        </div>
        {pinnedBlocks.length > 0 && (
          <ArenaChannelSection title="pinned">
            {renderSection('pinned', pinnedBlocks, 0, defaultViewMode)}
          </ArenaChannelSection>
        )}
        {laterBlocks.length > 0 && (
          <ArenaChannelSection title="later">
            {renderSection('later', laterBlocks, pinnedBlocks.length, 'list')}
          </ArenaChannelSection>
        )}
        {regularBlocks.length > 0 && (
          <>
            {pinnedBlocks.length > 0 && (
              <div class="arena-section-header">
                <h3>blocks</h3>
              </div>
            )}
            {renderSection(
              'blocks',
              regularBlocks,
              pinnedBlocks.length + laterBlocks.length,
              defaultViewMode,
            )}
          </>
        )}

        <div class="arena-block-modal" id="arena-modal">
          <div class="arena-modal-content">
            <div class="arena-modal-nav">
              <button
                type="button"
                class="arena-modal-nav-btn arena-modal-collapse"
                aria-label="Toggle item details"
              >
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

  ChannelContent.css = style

  return ChannelContent
}) satisfies QuartzComponentConstructor
