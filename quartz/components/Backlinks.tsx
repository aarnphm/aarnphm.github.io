import { i18n } from '../i18n'
import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../types/component'
import { renderDataFor } from '../util/ctx'
import { classNames } from '../util/lang'
import { resolveRelative, simplifySlug } from '../util/path'
import style from './styles/backlinks.scss'

export default (() => {
  const Backlinks: QuartzComponent = ({
    fileData,
    allFiles,
    displayClass,
    cfg,
    ctx,
  }: QuartzComponentProps) => {
    const slug = simplifySlug(fileData.slug!)
    const backlinkFiles = renderDataFor(ctx, allFiles).backlinksBySlug.get(slug) ?? []
    if (backlinkFiles.length === 0) {
      return null
    }

    return (
      <section data-backlinks class={classNames(displayClass, 'backlinks', 'main-col')}>
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
          {backlinkFiles.map(f => (
            <div class="backlink-card">
              <a
                href={resolveRelative(fileData.slug!, f.slug!)}
                data-backlink={f.slug!}
                data-slug={f.slug!}
                data-no-popover
                class="internal"
              >
                <span class="small">{f.frontmatter?.title}</span>
              </a>
              <div class="description" dangerouslySetInnerHTML={{ __html: f.description ?? '' }} />
            </div>
          ))}
        </div>
      </section>
    )
  }

  Backlinks.css = style
  return Backlinks
}) satisfies QuartzComponentConstructor
