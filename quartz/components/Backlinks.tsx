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
    return null
  }

  return (
    <section data-backlinks={true} class={classNames(displayClass, "backlinks", "side-col")}>
      <h2 id="backlinks-label">
        {i18n(cfg.locale).components.backlinks.title}
        <a
          data-role="anchor"
          aria-hidden="true"
          tabindex={-1}
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
            <use href="#github-anchor" />
          </svg>
        </a>
      </h2>
      <div class="overflow">
        {backlinkFiles.length > 0 ? (
          backlinkFiles.map((f) => (
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
          ))
        ) : (
          <div>{i18n(cfg.locale).components.backlinks.noBacklinksFound}</div>
        )}
      </div>
    </section>
  )
}

Backlinks.css = style
export default (() => Backlinks) satisfies QuartzComponentConstructor
