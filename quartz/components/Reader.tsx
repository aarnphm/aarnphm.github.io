import { QuartzComponent, QuartzComponentConstructor } from "../types/component"
// @ts-ignore
import readerScript from "./scripts/reader.inline"
import style from "./styles/reader.scss"

export default (() => {
  const Reader: QuartzComponent = () => <></>
  Reader.css = style
  Reader.beforeDOMLoaded = readerScript

  return Reader
}) satisfies QuartzComponentConstructor
