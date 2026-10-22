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
      <section
        id="backlinks"
        data-backlinks
        class={classNames(displayClass, 'backlinks', 'main-col')}
        aria-label={i18n(cfg.locale).components.backlinks.title}
      >
        <div class="overflow">
          <table class="backlinks-table" aria-label={i18n(cfg.locale).components.backlinks.title}>
            <thead>
              <tr>
                <th scope="col">backlinks</th>
                <th scope="col">description</th>
              </tr>
            </thead>
            <tbody>
              {backlinkFiles.map(f => (
                <tr class="backlink-card">
                  <th scope="row">
                    <a
                      href={resolveRelative(fileData.slug!, f.slug!)}
                      data-backlink={f.slug!}
                      data-slug={f.slug!}
                      data-no-popover
                      class="internal"
                    >
                      <span class="small">{f.frontmatter?.title}</span>
                    </a>
                  </th>
                  <td>
                    <div
                      class="description"
                      dangerouslySetInnerHTML={{ __html: f.description ?? '' }}
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    )
  }

  Backlinks.css = style
  return Backlinks
}) satisfies QuartzComponentConstructor
