import { type ComponentChildren } from 'preact'
//@ts-ignore
import script from '../scripts/zoomable.inline'
import style from '../styles/zoomable.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { label?: string; children?: ComponentChildren }

const ExpandIcon = () => (
  <svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    stroke-width="1.75"
    stroke-linecap="round"
    stroke-linejoin="round"
    aria-hidden="true"
  >
    <path d="M15 3h6v6" />
    <path d="M9 21H3v-6" />
    <path d="M21 3l-7 7" />
    <path d="M3 21l7-7" />
  </svg>
)

const CollapseIcon = () => (
  <svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    stroke-width="1.75"
    stroke-linecap="round"
    stroke-linejoin="round"
    aria-hidden="true"
  >
    <path d="M4 14h6v6" />
    <path d="M20 10h-6V4" />
    <path d="M14 10l7-7" />
    <path d="M3 21l7-7" />
  </svg>
)

const ZoomableImpl: QuartzMdxComponent<Props> = ({ label, children }) => {
  const triggerLabel = label ? `Expand ${label}` : 'Expand figure'
  return (
    <div class="zoomable" data-zoomable>
      <div class="zoomable-content">{children}</div>
      <button
        type="button"
        class="zoomable-trigger"
        data-zoomable-trigger
        aria-label={triggerLabel}
        title={triggerLabel}
      >
        <span class="zoomable-trigger-icon zoomable-trigger-icon--expand">
          <ExpandIcon />
        </span>
        <span class="zoomable-trigger-icon zoomable-trigger-icon--collapse">
          <CollapseIcon />
        </span>
      </button>
    </div>
  )
}

const ZoomableComponent = ZoomableImpl as QuartzMdxComponent<Props>
ZoomableComponent.css = style
ZoomableComponent.afterDOMLoaded = script

export const Zoomable = registerMdxComponent('Zoomable', ZoomableComponent)

export default (() => Zoomable) satisfies (opts: undefined) => QuartzMdxComponent<Props>
