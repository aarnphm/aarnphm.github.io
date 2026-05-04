import { ArenaData, ArenaChannel } from '../plugins/transformers/arena'
import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../types/component'
import { toArenaHeadingJsx } from '../util/arena'
import { classNames } from '../util/lang'
import { renderDescription } from '../util/og'
import { resolveRelative, FullSlug } from '../util/path'
import { StreamUnlockIcon } from './stream/UnlockIcon'

export interface Options {
  enableDescription: boolean
}

const defaultOptions: Options = { enableDescription: true }

export default ((userOpts?: Options) => {
  const opts = { ...defaultOptions, ...userOpts }

  const ArticleTitle: QuartzComponent = (componentData: QuartzComponentProps) => {
    const { fileData, displayClass } = componentData
    const title = fileData.frontmatter?.title
    const slug = fileData.slug!
    const isArenaIndex = slug === 'arena'
    const isArenaChannel = slug.startsWith('arena/') && slug !== 'arena'
    const isStreamPage = slug === 'stream' || slug.startsWith('stream/on')
    const hasStreamProtectedContent =
      isStreamPage && Object.keys(fileData.streamData?.protectedPayloads ?? {}).length > 0

    if (isArenaIndex) {
      const arenaData = fileData.arenaData as ArenaData | undefined

      return (
        <hgroup
          class={classNames(displayClass, 'title-col', 'arena-title-block')}
          data-article-title
        >
          <h1 class="article-title">are.na</h1>
          <p class="description">
            {arenaData
              ? `${arenaData.channels.length} channels · ${arenaData.channels.reduce((sum, ch) => sum + ch.blocks.length, 0)} blocks`
              : ''}
          </p>
        </hgroup>
      )
    }

    if (isArenaChannel) {
      const channel = fileData.arenaChannel as ArenaChannel | undefined
      const arenaRootSlug = 'arena' as FullSlug

      return (
        <hgroup
          class={classNames(displayClass, 'title-col', 'arena-title-block')}
          data-article-title
        >
          <h1 class="article-title">
            <a
              href={resolveRelative(slug, arenaRootSlug)}
              class="internal"
              data-no-popover
              data-slug={arenaRootSlug}
              style={{ background: 'transparent' }}
            >
              are.na
            </a>
            {' / '}
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
          <p class="description">{channel ? `${channel.blocks.length} blocks` : ''}</p>
        </hgroup>
      )
    }

    if (title) {
      return (
        <hgroup class={classNames(displayClass, 'title-col')} data-article-title>
          <h1 class="article-title">{title}</h1>
          {opts.enableDescription && (
            <p
              class="description"
              dangerouslySetInnerHTML={{
                __html: renderDescription(fileData.description, slug) ?? '',
              }}
            />
          )}
          {isStreamPage && (
            <nav class="stream-title-actions" aria-label="stream links">
              {hasStreamProtectedContent && (
                <button
                  type="button"
                  class="stream-title-unlock"
                  data-protected-unlock-trigger
                  aria-label="unlock protected stream entries"
                  title="unlock protected stream entries"
                >
                  <StreamUnlockIcon />
                </button>
              )}
              <a
                href="/stream/index.xml"
                class="internal"
                data-no-popover
                style="font-style: italic"
              >
                rss
              </a>
              <a href="/stream/on" class="internal" style="font-style: italic" data-no-popover>
                list
              </a>
            </nav>
          )}
        </hgroup>
      )
    }

    return <></>
  }

  return ArticleTitle
}) satisfies QuartzComponentConstructor<Options>
