import { classNames } from "../util/lang"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"

export default (() => {
  const Header: QuartzComponent = ({ children }: QuartzComponentProps) => {
    return children.length > 0 ? (
      <section class={classNames(undefined, "header", "full-col")}>
        <header class={classNames(undefined, "header-content", "full-col")}>{children}</header>
      </section>
    ) : null
  }

  return Header
}) satisfies QuartzComponentConstructor
