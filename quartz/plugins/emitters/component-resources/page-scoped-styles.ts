import triathlonStyle from '../../../components/styles/triathlon.scss'
import latentStyle from '../../../styles/pages/latent-sheet.scss'

export type PageScopedStyle = {
  name: string
  css: string
  classPrefixes: string[]
  /**
   * false ships the sheet outside `@layer quartz-base`: the latent sheet was part of the unlayered
   * custom.scss, and its rules must keep beating the site's unlayered ones as they did
   */
  layer?: boolean
}

// Component sheets kept out of component.css. Each ships as its own file, and only pages that
// render its class prefixes link it (util/page-styles.ts).
export const pageScopedComponentStyles: PageScopedStyle[] = [
  { name: 'triathlon', css: triathlonStyle, classPrefixes: ['triathlon', 'tri-'] },
  { name: 'latent', css: latentStyle, classPrefixes: ['latent', 'lat-'], layer: false },
]
