import { type QuartzComponent, type QuartzComponentConstructor } from '../types/component'
// @ts-expect-error Quartz's resource loader imports inline modules as script text.
import script from './scripts/homepage-trail.inline'
import style from './styles/homepage-trail.scss'

export default (() => {
  const HomepageTrail: QuartzComponent = ({ fileData }) => {
    if (fileData.slug !== 'index') return null
    return (
      <div class="homepage-trail">
        <time class="trail-local-time" aria-label="Local time" aria-live="off" />
        <span class="trail-visitor-count" title="Connected visitors · PlayHTML">
          <svg viewBox="0 0 12 14" aria-hidden="true">
            <path d="m1 1 9 6-4 1-2 4Z" />
          </svg>
          <span
            data-trail-visitors
            data-state="connecting"
            aria-live="off"
            aria-label="Connecting visitors"
          >
            ·
          </span>
        </span>
        <button
          class="homepage-moon-toggle"
          type="button"
          popovertarget="homepage-moon-panel"
          aria-label="Moon calendar"
          aria-haspopup="dialog"
          aria-expanded="false"
          aria-controls="homepage-moon-panel"
          title="Moon calendar"
        >
          <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
            <path d="M12 6c-3-2-6-2-9-1v14c3-1 6-1 9 1 3-2 6-2 9-1V5c-3-1-6-1-9 1Zm0 0v14" />
          </svg>
        </button>
        <div
          id="homepage-moon-panel"
          class="homepage-moon-panel"
          popover="auto"
          role="dialog"
          aria-label="Moon calendar"
        />
        <button
          class="homepage-trail-toggle"
          type="button"
          aria-label="Walking trail"
          aria-expanded="false"
          aria-controls="homepage-trail-scene"
          title="Show walking trail"
        >
          <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
            <path d="M4 20c8 1 11-4 6-7S7 5 17 4" />
            <path d="m15 2 3 2-2 3" />
          </svg>
        </button>
        <div
          id="homepage-trail-scene"
          class="homepage-trail-scene"
          role="region"
          aria-label="Shared trail to the campfire"
          hidden
        />
      </div>
    )
  }
  HomepageTrail.css = style
  HomepageTrail.afterDOMLoaded = script
  return HomepageTrail
}) satisfies QuartzComponentConstructor
