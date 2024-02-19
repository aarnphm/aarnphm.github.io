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
        <h4 style="font-size: initial">mes amis.</h4>
        <ul class="section-ul" id="friends-ul"></ul>
      </div>
    )
  }

  CuriusFriends.css = style
  CuriusFriends.beforeDOMLoaded = script

  return CuriusFriends
}) satisfies QuartzComponentConstructor
