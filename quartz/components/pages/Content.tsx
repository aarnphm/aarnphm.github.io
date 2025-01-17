import { htmlToJsx } from "../../util/jsx"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "../types"
import { FullSlug, joinSegments, resolveRelative } from "../../util/path"

export default (() => {
  const Content: QuartzComponent = ({ fileData, tree }: QuartzComponentProps) => {
    const hasSlides = fileData.frontmatter?.slides! || false
    const content = htmlToJsx(fileData.filePath!, tree)
    const classes: string[] = fileData.frontmatter?.cssclasses ?? []
    const classString = ["popover-hint", "main-col", ...classes].join(" ")
    return (
      <article class={classString}>
        {hasSlides && (
          <p>
            goto:{" "}
            <a
              data-no-popover
              data-slug={resolveRelative(
                fileData.slug!,
                joinSegments(fileData.slug!, "/slides") as FullSlug,
              )}
              href={resolveRelative(
                fileData.slug!,
                joinSegments(fileData.slug!, "/slides") as FullSlug,
              )}
            >
              slides deck
            </a>{" "}
            or{" "}
            <a data-no-popover data-slug="/" href="/">
              back home
            </a>
          </p>
        )}
        {content}
      </article>
    )
  }

  return Content
}) satisfies QuartzComponentConstructor
