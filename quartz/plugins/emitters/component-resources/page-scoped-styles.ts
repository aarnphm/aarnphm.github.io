import triathlonStyle from '../../../components/styles/triathlon.scss'

// Component sheets kept out of component.css. Each ships as its own file, and only pages that
// render its class prefixes link it (util/page-styles.ts).
export const pageScopedComponentStyles = [
  { name: 'triathlon', css: triathlonStyle, classPrefixes: ['triathlon', 'tri-'] },
]
