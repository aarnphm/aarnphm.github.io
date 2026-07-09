import type { QuartzComponent, QuartzComponentConstructor } from '../types/component'
import { BaseViewSelectorMarkup } from './BaseViewSelectorMarkup'
import script from './scripts/base-view-selector.inline'
import baseViewSelectorStyle from './styles/baseViewSelector.scss'

const BaseViewSelector: QuartzComponent = BaseViewSelectorMarkup

BaseViewSelector.css = baseViewSelectorStyle
BaseViewSelector.afterDOMLoaded = script

export default (() => BaseViewSelector) satisfies QuartzComponentConstructor
