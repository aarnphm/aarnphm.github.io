import { QuartzComponentConstructor } from "./types"
import landingStyle from "./styles/landing.scss"

export default (() => {
  function LandingComponent() {
    return (
      <div>
        <div class="content-container">
          <img src="/static/avatar.png" class="landing-logo" />
          <h1 class="landing-header">My name is Aaron.</h1>
          <p>
            Beige and pink are my two favorite colours. <a href={"/dump/Chaos"}>Chaos</a> constructs
            both the ego and the id. Reading and <a href={"/dump/Dishes"}>cooking</a> is how I
            nurture my friendship. Nurturing one's curiosity is the source for a happy life and
            adventure.
          </p>
          <p class="landing-job">
            Currently, I'm building <a href="https://bentoml.com">serving infrastructure</a> and
            explore our interaction with large language models.
          </p>
          <p class="landing-subhead">
            <em>garden</em>:{" "}
            <a href={"/books"} target="_blank">
              books
            </a>{" "}
            •{" "}
            <a href={"/posts/"} target="_self">
              mailbox
            </a>{" "}
            •{" "}
            <a href={"/dump"} target="_self">
              notes
            </a>{" "}
            •{" "}
            <a href={"/dump/projects"} target="_self">
              projects
            </a>{" "}
            •{" "}
            <a href={"/dump/Scents"} target="_self">
              scent
            </a>{" "}
            •{" "}
            <a href={"/uses"} target="_self">
              uses
            </a>{" "}
            •{" "}
            <a href={"/influence"} target="_self">
              influence
            </a>
          </p>
          <p>
            <em>socials:</em>{" "}
            <a href="https://github.com/aarnphm" target="_blank">
              github
            </a>{" "}
            •{" "}
            <a href="https://x.com/aarnphm_" target="_blank">
              twitter
            </a>
          </p>
        </div>
      </div>
    )
  }
  LandingComponent.css = landingStyle
  return LandingComponent
}) satisfies QuartzComponentConstructor
