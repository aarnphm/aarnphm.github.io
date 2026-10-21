import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../types/component'
import { inquirySpecOf } from '../util/inquiry-map'
import { classNames } from '../util/lang'
//@ts-ignore
import script from './scripts/inquiry-map.inline'
import style from './styles/inquiry-map.scss'

// The graph itself comes from `<page>/inquiries.json` (InquiryMap emitter); the browser lays it out.
export default (() => {
  const InquiryMap: QuartzComponent = ({ fileData, displayClass }: QuartzComponentProps) => {
    if (!inquirySpecOf(fileData.frontmatter?.inquiries)) return null
    return (
      <figure
        class={classNames(displayClass, 'inquiry-map')}
        data-inquiry-map
        role="group"
        aria-label="carte"
      >
        <canvas class="inquiry-map-halo" aria-hidden="true" />
        <svg class="inquiry-map-links" aria-hidden="true" />
        <div class="inquiry-map-list" />
      </figure>
    )
  }
  InquiryMap.css = style
  InquiryMap.afterDOMLoaded = script
  return InquiryMap
}) satisfies QuartzComponentConstructor
