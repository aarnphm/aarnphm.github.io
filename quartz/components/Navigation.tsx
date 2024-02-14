import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import navigationCss from "./styles/navigation.scss"
import { FullSlug, TransformOptions, transformLink } from "../util/path"

interface Options {
  prev: string
  next: string
}

const defaultOptions: Options = {
  prev: "/",
  next: "/tags",
}

export default ((userOpts?: Partial<Options>) => {
  const Navigation: QuartzComponent = ({ fileData, allFiles }: QuartzComponentProps) => {
    const transformOpts: TransformOptions = {
      strategy: "absolute",
      allSlugs: allFiles.map((f) => f.slug as FullSlug),
    }

    const transformNav = (nav: string) =>
      transformLink(fileData.slug!, nav.replace(/['"\[\]]+/g, ""), transformOpts)

    const navigation = fileData.frontmatter?.navigation
    let baseOpts: Options = defaultOptions
    if (navigation) {
      const [next, prev] = navigation
      baseOpts = {
        ...defaultOptions,
        prev: transformNav(prev),
        next: transformNav(next),
      }
    }

    const getALink = (text: string, href: string) => (
      <a href={href} rel="noopener noreferrer">
        {text}
      </a>
    )

    const frontmatter = fileData.frontmatter
    const opts = { ...baseOpts, ...userOpts }
    return (
      <footer class="navigation-container">
        <p>
          Vous pourriez être intéressé par {getALink("cela", opts.prev)} ou{" "}
          {getALink("peut-être cela", opts.next)}.
        </p>
      </footer>
    )
  }

  Navigation.css = navigationCss
  return Navigation
}) satisfies QuartzComponentConstructor
