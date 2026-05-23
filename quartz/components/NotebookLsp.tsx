import { QuartzComponent, QuartzComponentConstructor } from '../types/component'
import script from './scripts/notebook-lsp.inline'

export default (() => {
  const NotebookLsp: QuartzComponent = () => null
  NotebookLsp.afterDOMLoaded = script
  return NotebookLsp
}) satisfies QuartzComponentConstructor
