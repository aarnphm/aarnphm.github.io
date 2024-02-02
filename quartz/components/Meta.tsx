import { QuartzComponentConstructor, QuartzComponentProps } from "./types"

const globalGraph = {
  drag: true,
  zoom: true,
  depth: -1,
  scale: 0.9,
  repelForce: 0.5,
  centerForce: 0.3,
  linkDistance: 30,
  fontSize: 0.6,
  opacityScale: 1,
  showTags: true,
  removeTags: [],
}

export const Search = () => (
  <div class="search">
    <div id="search-container">
      <div id="search-space">
        <input
          autocomplete="off"
          id="search-bar"
          name="search"
          type="text"
          aria-label="Search for something"
          placeholder="Search for something"
        />
        <div id="search-layout" data-preview={true}></div>
      </div>
    </div>
  </div>
)

export const Graph = () => (
  <div class="graph">
    <div id="global-graph-icon"></div>
    <div id="global-graph-outer">
      <div id="global-graph-container" data-cfg={JSON.stringify(globalGraph)}></div>
    </div>
  </div>
)

export const DarkMode = () => (
  <div class="darkmode">
    <input class="toggle" id="darkmode-toggle" type="checkbox" tabIndex={-1} />
  </div>
)

interface Options {
  enableSearch?: boolean
}

const defaultOptions: Options = {
  enableSearch: true,
}

export default ((userOpts?: Partial<Options>) => {
  const opts = { ...defaultOptions, ...userOpts }
  function Meta(componentData: QuartzComponentProps) {
    return (
      <>
        {opts.enableSearch ? <Search /> : null}
        <Graph />
        <DarkMode />
      </>
    )
  }
  return Meta
}) satisfies QuartzComponentConstructor
