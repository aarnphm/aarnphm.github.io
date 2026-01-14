import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from "../types/component"
import style from "./styles/backlinks.scss"
import { resolveRelative, simplifySlug } from "../util/path"
import { i18n } from "../i18n"
import { classNames } from "../util/lang"

export default (() => {
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
      <section data-backlinks class={classNames(displayClass, "backlinks", "main-col")}>
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
              <a
                href={resolveRelative(fileData.slug!, f.slug!)}
                data-backlink={f.slug!}
                data-slug={f.slug!}
                data-no-popover
                class="internal"
              >
                <div class="small">{f.frontmatter?.title}</div>
                <div class="description">{f.description}</div>
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
  return Backlinks
}) satisfies QuartzComponentConstructor
