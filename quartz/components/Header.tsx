import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from "../types/component"
import { classNames } from "../util/lang"

export default (() => {
  const Header: QuartzComponent = ({ children }: QuartzComponentProps) => {
    return children.length > 0 ? (
      <section class={classNames(undefined, "header", "grid", "all-col")}>
        <header class={classNames(undefined, "header-content")}>{children}</header>
      </section>
    ) : null
  }

  return Header
}) satisfies QuartzComponentConstructor
