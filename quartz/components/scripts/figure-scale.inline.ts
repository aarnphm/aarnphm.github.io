import { observeSvgScale } from './figure-scale'
import { registerMdxContent } from './mdx-content'

// Figures without a script of their own name the property on each scaled svg, as in
// data-fig-scale="--rs-u"; their stylesheet multiplies label sizes by it.
registerMdxContent('svg[data-fig-scale]', node => {
  const svg = node as unknown as SVGSVGElement
  const property = svg.dataset.figScale
  if (property) window.addCleanup?.(observeSvgScale([svg], property))
})
