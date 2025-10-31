import { QuartzComponent, QuartzComponentConstructor } from "./types"
// @ts-ignore
import script from "./scripts/code-copy.inline"

export default (() => {
  const CodeCopy: QuartzComponent = () => {
    return <></>
  }

  CodeCopy.afterDOMLoaded = script

  return CodeCopy
}) satisfies QuartzComponentConstructor
