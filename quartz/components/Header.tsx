import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"

const Header: QuartzComponent = ({ children }: QuartzComponentProps) => {
  return children.length > 0 ? (
    <section class="header main-col">
      <header class="header-content main-col">{children}</header>
    </section>
  ) : null
}

export default (() => Header) satisfies QuartzComponentConstructor
