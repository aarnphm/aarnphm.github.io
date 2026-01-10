import { QuartzComponent, QuartzComponentConstructor } from "./types"
import style from "./styles/reader.scss"
// @ts-ignore
import readerScript from "./scripts/reader.inline"

export default (() => {
  const Reader: QuartzComponent = () => <></>
  Reader.css = style
  Reader.beforeDOMLoaded = readerScript

  return Reader
}) satisfies QuartzComponentConstructor
