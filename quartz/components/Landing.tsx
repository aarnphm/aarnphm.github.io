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
  curius: "https://curius.app/aaron-pham",
}
export const KeybindAlias = {
  "cmd+k": "search",
  "cmd+g": "graph",
  "cmd+a": "toggle dark mode",
}

export default (() => {
  function LandingComponent({ displayClass }: QuartzComponentProps) {
    return (
      <div class="popover-hint">
        <div class="landing">
          {/* search components */}
          <div class={`search ${displayClass ?? ""}`}>
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
                <div id="search-layout">
                  <div id="results-container"></div>
                  <div id="preview-container"></div>
                </div>
              </div>
            </div>
          </div>
          {/* graph components */}
          <div class={`graph ${displayClass ?? ""}`}>
            <div id="global-graph-icon"></div>
            <div id="global-graph-outer">
              <div id="global-graph-container" data-cfg={JSON.stringify(globalGraph)}></div>
            </div>
          </div>
          {/* darkmode components */}
          <div class={`darkmode ${displayClass ?? ""}`}>
            <input class="toggle" id="darkmode-toggle" type="checkbox" tabIndex={-1} />
          </div>
          {/* landing content */}
          <div class="content-container">
            <img src="/static/avatar.png" class="landing-logo" />
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
              when I'm not coding. I'm pretty bullish on investing into self and fullfil one's
              desire in life.
            </p>
            <p class="landing-job">
              Currently, I'm building <a href="https://bentoml.com">serving infrastructure</a> and
              explore our interaction with large language models.
            </p>
            <p class="landing-subhead">
              <em>garden</em>
              {": "}
              {Object.entries(HyperAlias).map(([key, value]) => {
                const isFinal = key === Object.keys(HyperAlias).at(-1)
                return (
                  <>
                    <a href={value} target="_self" class="internal landing-links">
                      {key}
                    </a>
                    {/* make sure to only append " · " if key and value is not the last item*/}
                    {!isFinal ? " · " : ""}
                  </>
                )
              })}
            </p>
            <p>
              <em>socials</em>
              {": "}
              {Object.entries(SocialAlias).map(([key, value]) => {
                const isFinal = key === Object.keys(SocialAlias).at(-1)
                return (
                  <>
                    <a href={value} target="_blank">
                      {key}
                    </a>
                    {!isFinal ? " · " : ""}
                  </>
                )
              })}
            </p>
            <hr />
            <p class="landing-usage">
              🖥️
              {" · "}
              {Object.entries(KeybindAlias).map(([key, value]) => {
                const isFinal = key === Object.keys(KeybindAlias).at(-1)
                return (
                  <>
                    <em>{key}</em> for {value} {!isFinal ? " · " : ""}
                  </>
                )
              })}
            </p>
          </div>
          {/* <div class="curius-container"> */}
          {/*   <p> */}
          {/*     Rabbit hole. More on{" "} */}
          {/*     <a href="https://curius.app/aaron-pham" target="_blank"> */}
          {/*       curius.app/aaron-pham */}
          {/*     </a> */}
          {/*   </p> */}
          {/* </div> */}
        </div>
      </div>
    )
  }
  LandingComponent.css = landingStyle
  LandingComponent.beforeDOMLoaded = darkModeScript
  LandingComponent.afterDOMLoaded = landingScript

  return LandingComponent
}) satisfies QuartzComponentConstructor
