import { i18n } from "../i18n"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"

export const SearchConstructor = (() => {
  const Search: QuartzComponent = ({ cfg }: QuartzComponentProps) => {
    const searchPlaceholder = i18n(cfg.locale).components.search.searchBarPlaceholder
    return (
      <div class="search">
        <div id="search-container">
          <div id="search-space">
            <input
              autocomplete="off"
              id="search-bar"
              name="search"
              type="text"
              aria-label={searchPlaceholder}
              placeholder={searchPlaceholder}
            />
            <div id="search-layout" data-preview={true}></div>
          </div>
        </div>
      </div>
    )
  }
  return Search
}) satisfies QuartzComponentConstructor

interface Options {
  enableSearch?: boolean
}

const defaultOptions: Options = {
  enableSearch: true,
}

export default ((userOpts?: Partial<Options>) => {
  const opts = { ...defaultOptions, ...userOpts }
  const Search = SearchConstructor()
  const Meta: QuartzComponent = (componentData: QuartzComponentProps) => {
    return <>{opts.enableSearch ? <Search {...componentData} /> : <></>}</>
  }
  return Meta
}) satisfies QuartzComponentConstructor<Options>
