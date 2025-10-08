import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import navigationCss from "./styles/navigation.scss"
import { FullSlug, TransformOptions, transformLink, resolveRelative } from "../util/path"
import { parseWikilink, resolveWikilinkTarget } from "../util/wikilinks"

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

    const transformNavLegacy = (nav: string) =>
      transformLink(fileData.slug!, nav.replace(/['"\[\]]+/g, ""), transformOpts)

    const resolveNavEntry = (nav: string) => {
      const parsed = parseWikilink(nav)
      if (parsed) {
        const resolved = resolveWikilinkTarget(parsed, fileData.slug as FullSlug)
        if (resolved) {
          const href = resolveRelative(fileData.slug!, resolved.slug)
          return parsed.anchor ? `${href}${parsed.anchor}` : href
        }
      }
      return transformNavLegacy(nav)
    }

    const navigation = fileData.frontmatter?.navigation as [string, string]
    let baseOpts: Options = defaultOptions
    if (navigation) {
      const [next, prev] = navigation
      baseOpts = {
        ...defaultOptions,
        prev: resolveNavEntry(prev),
        next: resolveNavEntry(next),
      }
    }

    const getALink = (text: string, href: string) => (
      <a href={href} rel="noopener noreferrer">
        {text}
      </a>
    )

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
