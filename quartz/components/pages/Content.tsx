import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../../types/component'
import { inheritComponentSourceNames } from '../../util/component-source'
import { htmlToJsx } from '../../util/jsx'
import { concatenateResources } from '../../util/resources'
//@ts-ignore
import lydiaScript from '../scripts/lydia.inline'
import SeeAlsoComponent from '../SeeAlso'

export default (() => {
  const SeeAlso = SeeAlsoComponent()

  const Content: QuartzComponent = (props: QuartzComponentProps) => {
    const { fileData, tree } = props
    const content = htmlToJsx(fileData.filePath!, tree)
    const classes: string[] = fileData.frontmatter?.cssclasses ?? []
    const classString = ['popover-hint', 'main-col', ...classes].join(' ')
    return (
      <>
        <article class={classString}>{content}</article>
        <SeeAlso {...props} />
      </>
    )
  }

  Content.afterDOMLoaded = lydiaScript
  Content.css = concatenateResources(SeeAlso.css)
  Content.sourceNames = inheritComponentSourceNames('Content', [SeeAlso])

  return Content
}) satisfies QuartzComponentConstructor
