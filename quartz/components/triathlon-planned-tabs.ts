import type { Element, ElementContent } from 'hast'
import { h } from 'hastscript'
import {
  PLANNED_LABEL,
  PLANNED_PANEL_CLASS,
  PLANNED_SHORT_LABEL,
  PLANNED_TAB_CLASS,
  plannedPanelAttributes,
  plannedTabAttributes,
  plannedTabForm,
  plannedTabStyle,
  workoutAnalysisOwner,
} from './triathlon/activity/planned-tab'

const hasClass = (node: ElementContent, name: string): node is Element =>
  node.type === 'element' &&
  Array.isArray(node.properties.className) &&
  node.properties.className.includes(name)

const textOf = (node: Element): string =>
  node.children.map(child => (child.type === 'text' ? child.value : '')).join('')

const plannedTab = (model: Element, base: string): Element => {
  const form = plannedTabForm(
    model.children.some(child => hasClass(child, 'tri-map-tab-shortcut')),
    textOf(model),
  )
  const attributes = { ...plannedTabAttributes(base), class: PLANNED_TAB_CLASS }
  if (form === 'expanded')
    return h('button', { ...attributes, style: plannedTabStyle(PLANNED_LABEL) }, [
      h('span.tri-map-tab-shortcut', { 'aria-hidden': 'true' }, PLANNED_SHORT_LABEL),
      h('span.tri-map-tab-label', { 'aria-hidden': 'true' }, PLANNED_LABEL),
    ])
  return h('button', attributes, form === 'short' ? PLANNED_SHORT_LABEL : PLANNED_LABEL)
}

/**
 * Adds the planned tab, first in the tab row, to each workout analysis section whose activity is
 * linked to a TrainingPeaks workout. The panel stays empty: the plan is private and the client
 * fills it after the training calendar session is unlocked.
 */
export const addPlannedTabs = (node: Element, planned: readonly number[] | undefined): void => {
  if (!planned?.length) return
  const ids = new Set(planned)
  const visit = (element: Element): void => {
    if (element.properties.dataWorkoutAnalysis !== undefined) {
      const tabs = element.children.find(child => hasClass(child, 'tri-workout-analysis-tabs'))
      const stage = element.children.find(child => hasClass(child, 'tri-workout-analysis-stage'))
      const model = tabs?.children.find((child): child is Element => child.type === 'element')
      const owner = workoutAnalysisOwner(String(model?.properties.id ?? ''))
      if (!tabs || !stage || !model || !owner || !ids.has(owner.activityId)) return
      tabs.children.unshift(plannedTab(model, owner.base))
      stage.children.unshift(
        h('div', { ...plannedPanelAttributes(owner.base), class: PLANNED_PANEL_CLASS }),
      )
      return
    }
    for (const child of element.children) if (child.type === 'element') visit(child)
  }
  visit(node)
}
