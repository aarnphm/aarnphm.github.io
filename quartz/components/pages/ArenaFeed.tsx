import type { QuartzComponent, QuartzComponentConstructor } from '../../types/component'
// @ts-ignore
import script from '../scripts/arena-feed.inline'
import style from '../styles/arena-feed.scss'

export default (() => {
  const ArenaFeed: QuartzComponent = () => (
    <article class="arena-feed main-col" data-arena-feed>
      <div class="arena-feed-mount" data-arena-feed-mount>
        <header class="arena-reader-header">
          <a href="/arena" class="internal">
            Arena
          </a>
          <span>Reader</span>
        </header>
        <div class="arena-reader-empty" role="status">
          <h1>Your reading queue</h1>
          <p>Loading your saved links. Later links come first.</p>
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
