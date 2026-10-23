import { type QuartzComponent, type QuartzComponentConstructor } from '../types/component'
import { resolveAsset } from '../util/asset-manifest'
import { joinSegments } from '../util/path'
// @ts-expect-error Quartz's resource loader imports inline modules as script text.
import script from './scripts/homepage-trail.inline'
import style from './styles/homepage-trail.scss'

export default (() => {
  const HomepageTrail: QuartzComponent = ({ fileData, ctx }) => {
    if (fileData.slug !== 'index') return null
    const source = joinSegments('/', resolveAsset(ctx, 'static/scripts/shared-trail.js'))
    return (
      <div class="homepage-trail">
        <time class="trail-local-time" aria-label="Local time" aria-live="off" />
        <button
          class="trail-visitor-count"
          type="button"
          aria-label="Visitors"
          aria-describedby="homepage-visitor-count"
          aria-haspopup="dialog"
          aria-expanded="false"
          aria-controls="homepage-visitor-panel"
          popovertarget="homepage-visitor-panel"
          title="Visitors · arrivals and departures"
        >
          <span class="trail-visitor-marker" aria-hidden="true" />
          <span
            id="homepage-visitor-count"
            data-trail-visitors
            data-state="connecting"
            aria-live="off"
            aria-label="Connecting visitors"
          >
            ·
          </span>
        </button>
        <div
          id="homepage-visitor-panel"
          class="homepage-visitor-panel"
          popover="auto"
          role="dialog"
          aria-label="Visitor register"
        >
          <div class="visitor-register-columns" aria-hidden="true">
            <span>visitor</span>
            <span>utc</span>
            <span>arrived</span>
            <span>left</span>
          </div>
          <ol class="visitor-register-list" data-visitor-register />
          <p data-visitor-register-empty>waiting for visitors</p>
          <div class="visitor-register-total" data-visitor-register-total>
            local time
          </div>
          <div class="visitor-register-connection">
            <span data-trail-connection role="status">
              connecting
            </span>
            <button type="button" data-trail-retry aria-label="Reconnect visitors" hidden>
              reconnect
            </button>
          </div>
        </div>
        <div class="trail-shared-cursors" aria-hidden="true" />
        <div class="trail-almanac">
          <button
            class="homepage-moon-toggle"
            type="button"
            popovertarget="homepage-moon-panel"
            aria-label="Moon calendar"
            aria-haspopup="dialog"
            aria-expanded="false"
            aria-controls="homepage-moon-panel"
            title="Moon calendar"
            hidden
          >
            moon calendar
          </button>
          <span class="trail-grove-total" data-grove-water-total aria-live="off" hidden />
        </div>
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
          <span class="homepage-trail-holiday" data-holiday-marker aria-hidden="true" hidden />
        </button>
        <div
          id="homepage-trail-scene"
          class="homepage-trail-scene"
          role="region"
          aria-label="Shared trail to the campfire"
          hidden
        >
          <iframe
            data-trail-frame
            srcDoc={`<!doctype html><html lang="en"><head><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="style-src 'self' 'unsafe-inline'"><title>Shared trail</title></head><body><script src="${source}"></script></body></html>`}
            title="Trail connection"
            hidden
            tabIndex={-1}
            aria-hidden="true"
          />
        </div>
      </div>
    )
  }
  HomepageTrail.css = style
  HomepageTrail.afterDOMLoaded = script
  return HomepageTrail
}) satisfies QuartzComponentConstructor
