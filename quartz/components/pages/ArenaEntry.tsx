import { QuartzComponent, QuartzComponentConstructor } from '../../types/component'
import { FullSlug, joinSegments, resolveRelative } from '../../util/path'
import { arenaChannelSections, createArenaChannelRenderer } from '../arena/ChannelBlock'
// @ts-ignore
import script from '../scripts/arena.inline'
import style from '../styles/arena.scss'

const EntryArrow = ({ direction }: { direction: 'previous' | 'next' }) => (
  <svg width="15" height="15" viewBox="0 0 15 15" fill="none" aria-hidden="true">
    <path
      d={
        direction === 'previous'
          ? 'M8.84182 3.13514C9.04327 3.32401 9.05348 3.64042 8.86462 3.84188L5.43521 7.49991L8.86462 11.1579C9.05348 11.3594 9.04327 11.6758 8.84182 11.8647C8.64036 12.0535 8.32394 12.0433 8.13508 11.8419L4.38508 7.84188C4.20477 7.64955 4.20477 7.35027 4.38508 7.15794L8.13508 3.15794C8.32394 2.95648 8.64036 2.94628 8.84182 3.13514Z'
          : 'M6.1584 3.13508C6.35985 2.94621 6.67627 2.95642 6.86514 3.15788L10.6151 7.15788C10.7954 7.3502 10.7954 7.64949 10.6151 7.84182L6.86514 11.8418C6.67627 12.0433 6.35985 12.0535 6.1584 11.8646C5.95694 11.6757 5.94673 11.3593 6.1356 11.1579L9.565 7.49985L6.1356 3.84182C5.94673 3.64036 5.95694 3.32394 6.1584 3.13508Z'
      }
      fill="currentColor"
    />
  </svg>
)

export default (() => {
  const ArenaEntry: QuartzComponent = componentData => {
    const { arenaChannel: channel, arenaEntry: block, slug } = componentData.fileData
    if (!channel || !block || !block.entryId || !slug) return <article>Entry not found</article>

    const renderBlock = createArenaChannelRenderer(componentData, channel)
    const sections = arenaChannelSections(channel)
    const ordered = [...sections.pinned, ...sections.later, ...sections.blocks]
    const index = ordered.findIndex(item => item.id === block.id)
    if (index < 0) throw new Error(`Arena entry missing from ${channel.slug}: ${block.id}`)
    const entryHref = (offset: number) => {
      const adjacent = ordered[index + offset]
      if (!adjacent) return undefined
      if (!adjacent.entryId)
        throw new Error(`Arena entry ID missing in ${channel.slug}: ${adjacent.id}`)
      return resolveRelative(
        slug,
        joinSegments('arena', channel.slug, adjacent.entryId) as FullSlug,
      )
    }
    const previousHref = entryHref(-1)
    const nextHref = entryHref(1)

    return (
      <article class="arena-entry-page main-col" id={block.entryId} data-entry-id={block.entryId}>
        <div class="arena-entry-shell">
          <nav class="arena-modal-nav" aria-label="Entry controls">
            <button
              type="button"
              class="arena-modal-nav-btn arena-modal-collapse"
              aria-label="Toggle item details"
              aria-controls={`arena-entry-details-${block.entryId}`}
              aria-expanded="true"
            >
              <svg
                width="15"
                height="15"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
                stroke-linecap="round"
                stroke-linejoin="round"
                aria-hidden="true"
              >
                <line x1="4" x2="20" y1="12" y2="12" />
                <line x1="4" x2="20" y1="6" y2="6" />
                <line x1="4" x2="20" y1="18" y2="18" />
              </svg>
            </button>
            {previousHref ? (
              <a
                class="arena-modal-nav-btn arena-modal-prev internal"
                href={previousHref}
                aria-label="Previous block"
                data-no-popover
              >
                <EntryArrow direction="previous" />
              </a>
            ) : (
              <button
                type="button"
                class="arena-modal-nav-btn arena-modal-prev"
                disabled
                aria-label="Previous block"
              >
                <EntryArrow direction="previous" />
              </button>
            )}
            {nextHref ? (
              <a
                class="arena-modal-nav-btn arena-modal-next internal"
                href={nextHref}
                aria-label="Next block"
                data-no-popover
              >
                <EntryArrow direction="next" />
              </a>
            ) : (
              <button
                type="button"
                class="arena-modal-nav-btn arena-modal-next"
                disabled
                aria-label="Next block"
              >
                <EntryArrow direction="next" />
              </button>
            )}
          </nav>
          <div class="arena-modal-body">{renderBlock(block, 0, 'page')}</div>
        </div>
      </article>
    )
  }

  ArenaEntry.css = style
  ArenaEntry.afterDOMLoaded = script
  return ArenaEntry
}) satisfies QuartzComponentConstructor
