import type { QuartzComponent, QuartzComponentConstructor } from '../../types/component'
import { ReaderLoading } from '../arena-feed/loading'
// @ts-ignore
import script from '../scripts/arena-feed.inline'
import style from '../styles/arena-feed.scss'

export default (() => {
  const ArenaFeed: QuartzComponent = () => (
    <article class="arena-feed main-col" data-arena-feed>
      <div class="arena-feed-mount" data-arena-feed-mount>
        <div class="arena-reader">
          <header class="arena-reader-header">
            <div>
              <a href="/arena" class="internal">
                arena
              </a>
              <span aria-hidden="true"> / </span>
              <span>reader</span>
            </div>
          </header>
          <ReaderLoading />
        </div>
      </div>
      <noscript>
        The reader needs JavaScript to load saved pages and synchronize notes.{' '}
        <a href="/arena">Browse the channels.</a>
      </noscript>
    </article>
  )
  ArenaFeed.css = style
  ArenaFeed.afterDOMLoaded = script
  return ArenaFeed
}) satisfies QuartzComponentConstructor
