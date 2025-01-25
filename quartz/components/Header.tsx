import { classNames } from "../util/lang"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"

type Props = {
  headerStyle?: "main-col" | "full-col"
} & QuartzComponentProps

export default (() => {
  const Header: QuartzComponent = ({ children, headerStyle }: Props) => {
    headerStyle = headerStyle ?? "main-col"
    return children.length > 0 ? (
      <section class={classNames(undefined, "header", headerStyle)} data-column={headerStyle}>
        <header class={classNames(undefined, "header-content", headerStyle)}>{children}</header>
      </section>
    ) : null
  }

  return Header
}) satisfies QuartzComponentConstructor
