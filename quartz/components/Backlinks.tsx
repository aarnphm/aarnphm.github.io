import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import style from "./styles/backlinks.scss"
import { resolveRelative, simplifySlug } from "../util/path"
import { i18n } from "../i18n"
import { classNames } from "../util/lang"
import { unescapeHTML } from "../util/escape"

const Backlinks: QuartzComponent = ({
  fileData,
  allFiles,
  displayClass,
  cfg,
}: QuartzComponentProps) => {
  const slug = simplifySlug(fileData.slug!)
  const backlinkFiles = allFiles.filter((file) => file.links?.includes(slug))
  if (backlinkFiles.length === 0) {
    return <></>
  }

  return (
    <section data-backlinks={true} class={classNames(displayClass, "backlinks")}>
      <h2>{i18n(cfg.locale).components.backlinks.title}</h2>
      <div class="overflow">
        {backlinkFiles.map((f) => (
          <a href={resolveRelative(fileData.slug!, f.slug!)} data-backlink={f.slug!}>
            <div class="small">{f.frontmatter?.title}</div>
            <div class="description">
              {unescapeHTML(
                f.frontmatter?.description ??
                  f.description?.trim() ??
                  i18n(cfg.locale).propertyDefaults.description,
              )}
            </div>
          </a>
        ))}
      </div>
    </section>
  )
}

Backlinks.css = style
export default (() => Backlinks) satisfies QuartzComponentConstructor
