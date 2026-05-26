import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../types/component'
import { componentSourceNames } from '../util/component-source'

type ConditionalRenderConfig = {
  component: QuartzComponent
  condition: (props: QuartzComponentProps) => boolean
}

export default ((config: ConditionalRenderConfig) => {
  const ConditionalRender: QuartzComponent = (props: QuartzComponentProps) => {
    if (config.condition(props)) {
      return <config.component {...props} />
    }

    return null
  }

  ConditionalRender.sourceNames = componentSourceNames(config.component)
  ConditionalRender.afterDOMLoaded = config.component.afterDOMLoaded
  ConditionalRender.beforeDOMLoaded = config.component.beforeDOMLoaded
  ConditionalRender.css = config.component.css

  return ConditionalRender
}) satisfies QuartzComponentConstructor<ConditionalRenderConfig>
