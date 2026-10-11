import { type QuartzComponent, type QuartzComponentConstructor } from '../types/component'
// @ts-expect-error Quartz's resource loader imports inline modules as script text.
import script from './scripts/homepage-trail.inline'
import style from './styles/homepage-trail.scss'

export default (() => {
  const HomepageTrail: QuartzComponent = ({ fileData }) => {
    if (fileData.slug !== 'index') return null
    return (
      <div class="homepage-trail">
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
