import { QuartzComponent, QuartzComponentConstructor } from '../types/component'
// @ts-ignore
import script from './scripts/notebook-runtime.inline'
import style from './styles/svgDiagram.scss'

export default (() => {
  const NotebookRuntimeLoader: QuartzComponent = () => null
  NotebookRuntimeLoader.afterDOMLoaded = script
  NotebookRuntimeLoader.css = style
  return NotebookRuntimeLoader
}) satisfies QuartzComponentConstructor
