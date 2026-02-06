import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../../types/component'
import { htmlToJsx } from '../../util/jsx'
import { FullSlug, joinSegments, resolveRelative } from '../../util/path'
import { concatenateResources } from '../../util/resources'
//@ts-ignore
import lydiaScript from '../scripts/lydia.inline'
import SeeAlsoComponent from '../SeeAlso'

export default (() => {
  const SeeAlso = SeeAlsoComponent()

  const Content: QuartzComponent = (props: QuartzComponentProps) => {
    const { fileData, tree } = props
    const hasSlides = (fileData.frontmatter && fileData.frontmatter.slides!) || false
    const content = htmlToJsx(fileData.filePath!, tree)
    const classes: string[] = fileData.frontmatter?.cssclasses ?? []
    const classString = ['popover-hint', 'main-col', ...classes].join(' ')
    return (
      <>
        <article class={classString}>
          {hasSlides && (
            <p>
              goto:{' '}
              <a
                data-no-popover
                data-slug={resolveRelative(
                  fileData.slug!,
                  joinSegments(fileData.slug!, '/slides') as FullSlug,
                )}
                href={resolveRelative(
                  fileData.slug!,
                  joinSegments(fileData.slug!, '/slides') as FullSlug,
                )}
              >
                slides deck
              </a>{' '}
              or{' '}
              <a data-no-popover data-slug="/" href="/">
                back home
              </a>
            </p>
          )}
          {content}
        </article>
        <SeeAlso {...props} />
      </>
    )
  }

  Content.afterDOMLoaded = lydiaScript
  Content.css = concatenateResources(SeeAlso.css)

  return Content
}) satisfies QuartzComponentConstructor
