import { QuartzComponent, QuartzComponentConstructor } from '../types/component'
//@ts-ignore
import script from './scripts/notebook-runtime.inline'

export default (() => {
  const NotebookRuntimeLoader: QuartzComponent = () => null
  NotebookRuntimeLoader.afterDOMLoaded = script
  return NotebookRuntimeLoader
}) satisfies QuartzComponentConstructor
