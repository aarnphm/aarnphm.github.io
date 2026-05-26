import { QuartzComponent, QuartzComponentProps } from '../types/component'
import { inheritComponentSourceNames } from '../util/component-source'
import { concatenateResources } from '../util/resources'

type BylineConstructor = (...components: QuartzComponent[]) => QuartzComponent

export default ((...components: QuartzComponent[]) => {
  const Components = Array.from(components)
  const Byline: QuartzComponent = (props: QuartzComponentProps) => {
    return (
      <section class="byline">
        {Components.map(Inner => (
          <Inner {...props} />
        ))}
      </section>
    )
  }

  Byline.displayName = 'Byline'
  Byline.sourceNames = inheritComponentSourceNames('Byline', Components)
  Byline.afterDOMLoaded = concatenateResources(...Components.map(c => c.afterDOMLoaded))
  Byline.beforeDOMLoaded = concatenateResources(...Components.map(c => c.beforeDOMLoaded))
  Byline.css = concatenateResources(...Components.map(c => c.css))
  return Byline
}) satisfies BylineConstructor
