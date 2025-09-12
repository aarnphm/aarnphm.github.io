import { htmlToJsx } from "../../util/jsx"
import { FilePath, resolveRelative, slugifyFilePath } from "../util/path"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "../types"
import { joinSegments } from "../../util/path"

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
            see also:{" "}
            <a
              data-no-popover
              data-slug={joinSegments(fileData.slug!, "/slides")}
              href={joinSegments(fileData.slug!, "/slides")}
            >
              slides deck
            </a>
          </p>
        )}
        {content}
      </article>
    )
  }

  return Content
}) satisfies QuartzComponentConstructor
