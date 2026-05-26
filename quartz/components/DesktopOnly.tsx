import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../types/component'
import { componentSourceNames } from '../util/component-source'

export default ((component: QuartzComponent) => {
  const Component = component
  const DesktopOnly: QuartzComponent = (props: QuartzComponentProps) => {
    return <Component displayClass="desktop-only" {...props} />
  }

  DesktopOnly.displayName = component.displayName
  DesktopOnly.sourceNames = componentSourceNames(component)
  DesktopOnly.afterDOMLoaded = component?.afterDOMLoaded
  DesktopOnly.beforeDOMLoaded = component?.beforeDOMLoaded
  DesktopOnly.css = component?.css
  return DesktopOnly
}) satisfies QuartzComponentConstructor<QuartzComponent>
