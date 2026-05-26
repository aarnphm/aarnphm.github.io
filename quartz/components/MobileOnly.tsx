import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../types/component'
import { componentSourceNames } from '../util/component-source'

export default ((component: QuartzComponent) => {
  const Component = component
  const MobileOnly: QuartzComponent = (props: QuartzComponentProps) => {
    return <Component displayClass="mobile-only" {...props} />
  }

  MobileOnly.displayName = component.displayName
  MobileOnly.sourceNames = componentSourceNames(component)
  MobileOnly.afterDOMLoaded = component?.afterDOMLoaded
  MobileOnly.beforeDOMLoaded = component?.beforeDOMLoaded
  MobileOnly.css = component?.css
  return MobileOnly
}) satisfies QuartzComponentConstructor<QuartzComponent>
