import { QuartzComponentConstructor, QuartzComponentProps, QuartzComponent } from "./types"
import style from "./styles/curiusFriends.scss"
//@ts-ignore
import script from "./scripts/curius-friends.inline"
import { classNames } from "../util/lang"

export default (() => {
  const CuriusFriends: QuartzComponent = (props: QuartzComponentProps) => {
    const { displayClass } = props
    return (
      <div class={classNames(displayClass, "curius-friends")}>
        <h4 style={["font-size: initial", "margin-top: unset", "margin-bottom: 0.5rem"].join(";")}>
          mes amis.
        </h4>
        <ul class="section-ul" id="friends-list" style="margin-top: unset"></ul>
        <div id="see-more-friends">
          Void{" "}
          <span id="more" style="text-decoration: none !important">
            de plus
          </span>
          <svg
            fill="currentColor"
            preserveAspectRatio="xMidYMid meet"
            height="1rem"
            width="1rem"
            viewBox="0 -10 40 40"
          >
            <g>
              <path d="m31 12.5l1.5 1.6-12.5 13.4-12.5-13.4 1.5-1.6 11 11.7z"></path>
            </g>
          </svg>
        </div>
      </div>
    )
  }

  CuriusFriends.css = style
  CuriusFriends.beforeDOMLoaded = script

  return CuriusFriends
}) satisfies QuartzComponentConstructor
