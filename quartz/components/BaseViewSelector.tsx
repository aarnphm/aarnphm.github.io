import type { QuartzComponent, QuartzComponentConstructor } from '../types/component'
import { BaseViewSelectorMarkup } from './BaseViewSelectorMarkup'
import bookcaseScript from './scripts/base-bookcase.inline'
import kanbanScript from './scripts/base-kanban.inline'
import vaultScript from './scripts/base-vault.inline'
import script from './scripts/base-view-selector.inline'
import baseViewSelectorStyle from './styles/baseViewSelector.scss'

const BaseViewSelector: QuartzComponent = BaseViewSelectorMarkup

BaseViewSelector.css = baseViewSelectorStyle
BaseViewSelector.afterDOMLoaded = [script, kanbanScript, bookcaseScript, vaultScript]

export default (() => BaseViewSelector) satisfies QuartzComponentConstructor
