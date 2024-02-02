import { QuartzComponentConstructor, QuartzComponentProps } from "./types"
import navigationCss from "./styles/navigation.scss"

interface Options {
  prev: string
  next: string
}

const defaultOptions: Options = {
  prev: "/",
  next: "/tags",
}

export default ((userOpts?: Partial<Options>) => {
  function Navigation({ fileData, componentData }: QuartzComponentProps) {
    const frontmatter = fileData.frontmatter
    const opts = { ...defaultOptions, ...frontmatter?.navigation, ...userOpts }
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
