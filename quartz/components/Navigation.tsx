import { QuartzComponentConstructor, QuartzComponentProps } from "./types"
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
  function Navigation({ fileData, allFiles }: QuartzComponentProps) {
    const transformOpts: TransformOptions = {
      strategy: "absolute",
      allSlugs: allFiles.map((f) => f.slug as FullSlug),
    }

    const transformNav = (nav: string) =>
      transformLink(fileData.slug!, nav.replace(/['"\[\]]+/g, ""), transformOpts)

    const navigation = fileData.frontmatter?.navigation
    let baseOpts: Options = defaultOptions
    if (navigation) {
      baseOpts = {
        ...defaultOptions,
        prev: transformNav(navigation.prev),
        next: transformNav(navigation.next),
      }
    }

    const frontmatter = fileData.frontmatter
    const opts = { ...baseOpts, ...userOpts }
    return (
      <footer class="navigation-container">
        <p>
          You might be interested in{" "}
          <a href={opts.prev} rel="noopener noreferrer">
            this
          </a>{" "}
          or{" "}
          <a href={opts.next} rel="noopener noreferrer">
            maybe this.
          </a>
        </p>
      </footer>
    )
  }

  Navigation.css = navigationCss
  return Navigation
}) satisfies QuartzComponentConstructor
