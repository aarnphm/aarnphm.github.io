import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../types/component'
import { classNames } from '../util/lang'
// @ts-ignore
import script from './scripts/headings-modal.inline'
import style from './styles/headings.scss'

export default (() => {
  const Headings: QuartzComponent = ({ displayClass }: QuartzComponentProps) => {
    return (
      <div class={classNames(displayClass, 'headings-modal-container')} style={{ display: 'none' }}>
        <div class="headings-modal">
          <div class="headings-modal-content">
            <div class="headings-modal-header">
            </div>
            <div class="headings-list" />
          </div>
        </div>
      </div>
    )
  }

  Headings.afterDOMLoaded = script
  Headings.css = style

  return Headings
}) satisfies QuartzComponentConstructor
