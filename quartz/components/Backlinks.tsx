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
      <h2 id="backlinks-label">
        {i18n(cfg.locale).components.backlinks.title}
        <a
          role="anchor"
          aria-hidden="true"
          tabindex="-1"
          data-no-popover="true"
          href="#backlinks-label"
          class="internal"
        >
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
          >
            <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"></path>
            <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"></path>
          </svg>
        </a>
      </h2>
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
