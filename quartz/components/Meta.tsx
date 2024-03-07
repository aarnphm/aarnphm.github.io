import { i18n } from "../i18n"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { D3Config, defaultOptions as graphOptions } from "./Graph"

export const globalGraphConfig: Partial<D3Config> = {
  repelForce: 0.3732,
  centerForce: 0.088,
  fontSize: 0.5375,
}

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

export const GraphConstructor = (() => {
  const Graph: QuartzComponent = (componentData: QuartzComponentProps) => {
    const opts = { ...graphOptions.globalGraph, ...globalGraphConfig }

    return (
      <div class="graph">
        <div id="global-graph-icon"></div>
        <div id="global-graph-outer">
          <div id="global-graph-container" data-cfg={JSON.stringify(opts)}></div>
        </div>
      </div>
    )
  }
  return Graph
}) satisfies QuartzComponentConstructor

export const DarkmodeConstructor = (() => {
  const Darkmode: QuartzComponent = (componentData: QuartzComponentProps) => {
    return (
      <div class="darkmode">
        <input class="toggle" id="darkmode-toggle" type="checkbox" tabIndex={-1} />
      </div>
    )
  }
  return Darkmode
}) satisfies QuartzComponentConstructor

interface Options {
  enableSearch?: boolean
  enableGraph?: boolean
  enableDarkMode?: boolean
}

const defaultOptions: Options = {
  enableSearch: true,
  enableGraph: true,
  enableDarkMode: false,
}

export default ((userOpts?: Partial<Options>) => {
  const opts = { ...defaultOptions, ...userOpts }
  const Search = SearchConstructor()
  const Graph = GraphConstructor()
  const Darkmode = DarkmodeConstructor()

  const Meta: QuartzComponent = (componentData: QuartzComponentProps) => {
    return (
      <>
        {opts.enableSearch ? <Search {...componentData} /> : <></>}
        {opts.enableGraph ? <Graph {...componentData} /> : <></>}
        {opts.enableDarkMode ? <Darkmode {...componentData} /> : <></>}
      </>
    )
  }
  return Meta
}) satisfies QuartzComponentConstructor
