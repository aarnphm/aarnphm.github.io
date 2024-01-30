import { QuartzComponentConstructor, QuartzComponentProps } from "./types"
import landingStyle from "./styles/landing.scss"
//@ts-ignore
import landingScript from "./scripts/landing.inline"
//@ts-ignore
import darkModeScript from "./scripts/darkmode.inline"

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

export const HyperAlias = {
  books: "/books",
  mailbox: "/posts/",
  notes: "/dump/",
  projects: "/dump/projects",
  uses: "/uses",
  advices: "/dump/quotes",
  affecter: "/influence",
}
export const SocialAlias = {
  github: "https://github.com/aarnphm",
  twitter: "https://x.com/aarnphm_",
  curius: "/curius",
}
export const KeybindAlias = {
  "cmd+k": "search",
  "cmd+g": "graph",
  "cmd+a": "toggle dark mode",
  "cmd+/": "homepage",
}

type AliasLinkProp = {
  name: string
  url: string
  isInternal?: boolean
  newTab?: boolean
}

const AliasLink = (props: AliasLinkProp) => {
  const opts = { isInternal: false, newTab: false, ...props }
  const className = ["landing-links"]
  if (opts.isInternal) className.push("internal")
  return (
    <li>
      <a href={opts.url} target={opts.newTab ? "_blank" : "_self"} className={className.join(" ")}>
        {opts.name}
      </a>
    </li>
  )
}

const Search = () => (
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

const Content = () => (
  <div class="content-container">
    <h1 class="landing-header">My name is Aaron.</h1>
    <p>
      Beige and <a class="rose">rosé</a> are my two favorite colours.{" "}
      <a href={"/dump/Chaos"} target="_self" class="internal landing-links">
        Chaos
      </a>{" "}
      constructs the id and form the ego. I enjoy treating my friends with{" "}
      <a href={"/dump/Dishes"} target="_self" class="internal landing-links">
        cooking
      </a>
      . I spend a lot of time{" "}
      <a href={"/dump/writing"} target="_self" class="internal landing-links">
        writing
      </a>{" "}
      and{" "}
      <a href={"/books"} target="_self" class="internal landing-links">
        reading
      </a>{" "}
      when I'm not coding. I'm pretty bullish on investing into self and fullfil one's desire in
      life.
    </p>
    <p class="landing-job">
      Currently, I'm building{" "}
      <a href="https://bentoml.com" target="_blank" rel="noopener noreferrer">
        serving infrastructure
      </a>{" "}
      and explore our interaction with large language models.
    </p>
    <hr />
    <p class="landing-subhead">
      <h3>garden:</h3>
      <ul id="garden">
        {Object.entries(HyperAlias).map(([name, url], index, array) => (
          <AliasLink key={name} name={name} url={url} isInternal />
        ))}
      </ul>
    </p>
    <p>
      <h3>socials:</h3>
      <ul id="socials">
        {Object.entries(SocialAlias).map(([name, url], index, array) => (
          <AliasLink key={name} name={name} url={url} newTab={name !== "curius"} />
        ))}
      </ul>
    </p>
    <hr />
    <p class="landing-usage">
      <ul class="keybinds">
        {Object.entries(KeybindAlias).map(([key, value], index, array) => (
          <li>
            <a id="landing-keybind" data-keybind={key.replaceAll("+", "--")}>
              {key}
            </a>
            : {value}
          </li>
        ))}
      </ul>
    </p>
  </div>
)

export default (() => {
  function LandingComponent(componentData: QuartzComponentProps) {
    return (
      <div class="popover-hint">
        <div class="landing">
          <Search />
          <Graph />
          <DarkMode />
          <Content />
        </div>
      </div>
    )
  }
  LandingComponent.css = landingStyle
  LandingComponent.beforeDOMLoaded = darkModeScript
  LandingComponent.afterDOMLoaded = landingScript

  return LandingComponent
}) satisfies QuartzComponentConstructor
