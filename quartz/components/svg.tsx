import { render } from 'preact-render-to-string'
import { FilePath } from '../util/path'
import { quartzIconsJsx } from './svg/sprite-definition'
export {
  bskySvg,
  githubSvg,
  githubWhiteSvg,
  substackSvg,
  twitterSvg,
  youtubeSvg,
} from './svg/social-marks'
export { hfSvg } from './svg/hugging-face-mark'
export { anthropicSvg, modularSvg, openaiSvg, ycSvg } from './svg/technology-marks'
export { doiSvg, gwernSvg, obsidianSvg } from './svg/knowledge-marks'
export { bentomlHoverSvg, bentomlSvg } from './svg/bentoml-marks'

export { svgOptions } from '../util/svg-options'

const quartzIconsRawHtml = render(quartzIconsJsx)
const quartzIconsInnerHtml = quartzIconsRawHtml.slice(
  quartzIconsRawHtml.indexOf('>') + 1,
  quartzIconsRawHtml.lastIndexOf('</svg>'),
)
const quartzIconsFastJsx = (
  <svg
    class="quartz-icons"
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    style="height: 0;"
    data-singleton={true}
    dangerouslySetInnerHTML={{ __html: quartzIconsInnerHtml }}
  />
)

export const QuartzIcon = (_props?: { filePath?: FilePath }) => quartzIconsFastJsx
