import { QuartzComponentConstructor, QuartzComponentProps } from "./types"
import landingStyle from "./styles/landing.scss"
//@ts-ignore
import popoverScript from "./scripts/popover.inline"

export default (() => {
  function LandingComponent() {
    return (
      <div class="popover-hint">
        <div class="content-container">
          <img src="/static/avatar.png" class="landing-logo" />
          <h1 class="landing-header">My name is Aaron.</h1>
          <p>
            Beige and <a class="rose">rosé</a> are my two favorite colours.{" "}
            <a href={"/dump/Chaos"} target="_blank" class="internal landing-links">
              Chaos
            </a>{" "}
            constructs the id and form the ego. I enjoy treating my friends with{" "}
            <a href={"/dump/Dishes"} target="_blank" class="internal landing-links">
              cooking
            </a>
            . I spend a lot of time{" "}
            <a href={"/dump/writing"} target="_blank" class="internal landing-links">
              writing
            </a>{" "}
            and{" "}
            <a href={"/books"} target="_blank" class="internal landing-links">
              reading
            </a>
            . I'm pretty bullish on investing on my curiosity and fullfil one's desire in life.
          </p>
          <p class="landing-job">
            Currently, I'm building <a href="https://bentoml.com">serving infrastructure</a> and
            explore our interaction with large language models.
          </p>
          <p class="landing-subhead">
            <em>garden</em>
            {": "}
            <a href={"/books"} target="_blank" class="internal landing-links">
              books
            </a>
            {" · "}
            <a href={"/posts/"} target="_blank" class="internal landing-links">
              mailbox
            </a>
            {" · "}
            <a href={"/dump/"} target="_blank" class="internal landing-links">
              notes
            </a>
            {" · "}
            <a href={"/dump/projects"} target="_blank" class="internal landing-links">
              projects
            </a>
            {" · "}
            <a href={"/dump/Scents"} target="_blank" class="internal landing-links">
              scent
            </a>
            {" · "}
            <a href={"/uses"} target="_blank" class="internal landing-links">
              uses
            </a>
            {" · "}
            <a href={"/influence"} target="_blank" class="internal landing-links">
              influence
            </a>
          </p>
          <p>
            <em>socials</em>
            {": "}
            <a href="https://github.com/aarnphm" target="_blank">
              github
            </a>
            {" · "}
            <a href="https://x.com/aarnphm_" target="_blank">
              twitter
            </a>
            {" · "}
            <a href="https://curius.app/aaron-pham" target="_blank">
              curius
            </a>
          </p>
        </div>
      </div>
    )
  }
  LandingComponent.css = landingStyle
  LandingComponent.afterDOMLoaded = popoverScript

  return LandingComponent
}) satisfies QuartzComponentConstructor
