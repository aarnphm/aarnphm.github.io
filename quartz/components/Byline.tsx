import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../types/component'
import { concatenateResources } from '../util/resources'

export default ((...components: QuartzComponent[]) => {
  const Components = Array.from(components)
  const Byline: QuartzComponent = (props: QuartzComponentProps) => {
    return (
      <section class="byline all-col grid">
        {Components.map(Inner => (
          <Inner {...props} />
        ))}
      </section>
    )
  }

  Byline.displayName = 'Byline'

  Byline.afterDOMLoaded = concatenateResources(...Components.map(c => c.afterDOMLoaded))
  Byline.beforeDOMLoaded = concatenateResources(...Components.map(c => c.beforeDOMLoaded))
  Byline.css = concatenateResources(...Components.map(c => c.css))
  return Byline
}) satisfies QuartzComponentConstructor<any>
