import { i18n } from "../i18n"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"

import { defaultOptions as GraphOptions } from "./Graph"

export const SearchConstructor = (() => {
  const Search: QuartzComponent = ({ cfg }: QuartzComponentProps) => {
    const searchPlaceholder = i18n(cfg.locale).components.search.searchBarPlaceholder
    return (
      <div class="search">
        <search id="search-container">
          <form id="search-space">
            <input
              autocomplete="off"
              id="search-bar"
              name="search"
              type="text"
              aria-label={searchPlaceholder}
              placeholder={searchPlaceholder}
            />
          </form>
          <output id="search-layout" data-preview={true}></output>
        </search>
      </div>
    )
  }
  return Search
}) satisfies QuartzComponentConstructor

export const GraphConstructor = (() => {
  const Graph: QuartzComponent = () => {
    return (
      <div class="graph">
        <div id="global-graph-outer">
          <div
            id="global-graph-container"
            data-cfg={JSON.stringify(GraphOptions.globalGraph)}
          ></div>
        </div>
      </div>
    )
  }
  return Graph
}) satisfies QuartzComponentConstructor

export default (() => {
  const Search = SearchConstructor()
  const Graph = GraphConstructor()
  const Meta: QuartzComponent = (componentData: QuartzComponentProps) => {
    return (
      <>
        <Search {...componentData} />
        <Graph {...componentData} />
      </>
    )
  }
  return Meta
}) satisfies QuartzComponentConstructor
