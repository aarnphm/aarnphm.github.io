import { type QuartzComponent, type QuartzComponentConstructor } from '../types/component'
// @ts-expect-error Quartz's resource loader imports inline modules as script text.
import script from './scripts/visitor-weather.inline'
import style from './styles/visitor-weather.scss'

export default (() => {
  const VisitorWeather: QuartzComponent = ({ fileData }) => {
    if (fileData.slug !== 'index') return null
    return (
      <div class="visitor-weather" data-weather-state="loading">
        <button
          class="visitor-weather-toggle"
          type="button"
          popovertarget="visitor-weather-panel"
          aria-label="Switch theme"
          aria-expanded="false"
          aria-controls="visitor-weather-panel"
          title="Switch theme"
        >
          <svg
            class="visitor-weather-icon"
            viewBox="0 0 24 24"
            aria-hidden="true"
            focusable="false"
          >
            <g class="visitor-weather-sun">
              <circle cx="12" cy="12" r="4" />
              <path
                class="visitor-weather-rays"
                d="M12 3v2M12 19v2M3 12h2M19 12h2M5.6 5.6 7 7M17 17l1.4 1.4M5.6 18.4 7 17M17 7l1.4-1.4"
              />
            </g>
            <g class="visitor-weather-moon">
              <g transform="translate(12 12)">
                <g class="visitor-weather-moon-motion">
                  <circle class="lunar-disc-outline" r="7" />
                  <path class="lunar-disc-lit" data-lunar-disc="" />
                </g>
              </g>
            </g>
          </svg>
        </button>
        <div
          id="visitor-weather-panel"
          class="visitor-weather-panel"
          popover="auto"
          role="region"
          aria-label="Weather near you"
        >
          <div class="visitor-weather-heading" role="status">
            <span class="visitor-weather-summary">
              <span data-weather-temperature /> <span data-weather-condition>loading weather</span>
            </span>
            <span class="visitor-weather-place">
              <span class="visitor-weather-country" data-weather-country />
            </span>
          </div>
          <div
            class="visitor-weather-map"
            role="region"
            aria-label="Map near your approximate location"
          >
            <div data-weather-map aria-hidden="true" />
            <button
              type="button"
              class="visitor-weather-map-status"
              data-weather-map-status
              disabled
            >
              map near you
            </button>
          </div>
          <div class="visitor-weather-readout">
            <span data-weather-countdown>local daylight</span>
            <span data-weather-wind />
          </div>
          <button class="visitor-weather-retry" type="button" data-weather-retry hidden>
            try again
          </button>
        </div>
      </div>
    )
  }
  VisitorWeather.css = style
  VisitorWeather.afterDOMLoaded = script
  return VisitorWeather
}) satisfies QuartzComponentConstructor
