import { QuartzComponent, QuartzComponentConstructor } from "./types"
import style from "./styles/streamSearch.scss"
// @ts-ignore
import script from "./scripts/streamSearch.inline"

export interface StreamSearchOptions {
  placeholder?: string
}

const defaultOptions: StreamSearchOptions = {
  placeholder: "search stream...",
}

export default ((userOpts?: Partial<StreamSearchOptions>) => {
  const StreamSearch: QuartzComponent = () => {
    const opts = { ...defaultOptions, ...userOpts }

    return (
      <div class="stream-search-container">
        <form class="stream-search-form">
          <input
            type="text"
            name="q"
            class="stream-search-input"
            placeholder={opts.placeholder}
            autocomplete="off"
            aria-label="Search stream entries"
          />
        </form>
      </div>
    )
  }

  StreamSearch.afterDOMLoaded = script
  StreamSearch.css = style

  return StreamSearch
}) satisfies QuartzComponentConstructor
