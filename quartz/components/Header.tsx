import { classNames } from "../util/lang"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"

type Props = {
  headerStyle?: "main-col" | "full-col"
} & QuartzComponentProps

const Header: QuartzComponent = ({ children, headerStyle }: Props) => {
  headerStyle = headerStyle ?? "full-col"
  return children.length > 0 ? (
    <section class={classNames(undefined, "header", headerStyle)} data-column={headerStyle}>
      <header class={classNames(undefined, "header-content", headerStyle)}>{children}</header>
    </section>
  ) : null
}

export default (() => Header) satisfies QuartzComponentConstructor
