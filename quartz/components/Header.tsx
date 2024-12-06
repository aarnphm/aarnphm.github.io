import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"

const Header: QuartzComponent = ({ children }: QuartzComponentProps) => {
  return children.length > 0 ? (
    <section class="header full-col">
      <header class="header-content">{children}</header>
    </section>
  ) : null
}

export default (() => Header) satisfies QuartzComponentConstructor
