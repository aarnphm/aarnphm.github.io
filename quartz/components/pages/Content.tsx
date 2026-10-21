import type { Element } from 'hast'
import { EXIT, visit } from 'unist-util-visit'
import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../../types/component'
import { inheritComponentSourceNames } from '../../util/component-source'
import { unitOf } from '../../util/course'
import { htmlToJsx } from '../../util/jsx'
import { concatenateResources } from '../../util/resources'
import { CourseUnitWorkspace } from '../CourseUnit'
//@ts-ignore
import frMarginsScript from '../scripts/fr-margins.inline'
//@ts-ignore
import lydiaScript from '../scripts/lydia.inline'
//@ts-ignore
import unbadgeScript from '../scripts/unbadge.inline'
import SeeAlsoComponent from '../SeeAlso'
import frMarginsStyle from '../styles/fr-margins.scss'

export default (() => {
  const SeeAlso = SeeAlsoComponent()
  const Workspace = CourseUnitWorkspace()

  const Content: QuartzComponent = (props: QuartzComponentProps) => {
    const { fileData, tree } = props
    const content = htmlToJsx(fileData.filePath!, tree)
    let hasNotebookRuntime = false
    visit(tree, 'element', (node: Element) => {
      if (
        Array.isArray(node.properties.className) &&
        node.properties.className.includes('notebook-runtime')
      ) {
        hasNotebookRuntime = true
        return EXIT
      }
    })
    const classes: string[] = fileData.frontmatter?.cssclasses ?? []
    const unitKind = unitOf(fileData.slug ?? '')?.kind
    // authored and converted lecture notes and problem sets open the workspace
    const isLecture = unitKind === 'lectures' || unitKind === 'notes' || unitKind === 'psets'
    const classString = [
      'popover-hint',
      'main-col',
      ...(hasNotebookRuntime ? ['notebook-runtime-page'] : []),
      ...(isLecture ? ['course-lecture'] : []),
      ...classes,
    ].join(' ')
    const article = (
      <article
        class={classString}
        id={isLecture ? 'course-lecture-reading' : undefined}
        tabIndex={isLecture ? -1 : undefined}
      >
        {content}
      </article>
    )
    return (
      <>
        {isLecture ? (
          <div class="course-reading-split">
            {article}
            <Workspace {...props} />
          </div>
        ) : (
          article
        )}
        <SeeAlso {...props} />
      </>
    )
  }

  Content.afterDOMLoaded = concatenateResources(
    frMarginsScript,
    lydiaScript,
    unbadgeScript,
    Workspace.afterDOMLoaded,
  )
  Content.css = concatenateResources(SeeAlso.css, Workspace.css, frMarginsStyle)
  Content.sourceNames = inheritComponentSourceNames('Content', [SeeAlso, Workspace])

  return Content
}) satisfies QuartzComponentConstructor
