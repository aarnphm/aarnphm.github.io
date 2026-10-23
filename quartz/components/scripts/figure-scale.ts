// Type inside a scaled SVG is sized in user units, so it grows and shrinks with the rendered
// width. This keeps `property` on each svg at user units per rendered px; CSS multiplies a px size
// by it (`font-size: calc(var(--x-u) * 12px)`) to hold that size in every frame. The observer
// fires again when Zoomable moves the figure into its expanded frame.
const unitsPerPx = (svg: SVGSVGElement, width: number, height: number): number | null => {
  const box = svg.viewBox.baseVal
  if (!box || box.width <= 0 || width <= 0) return null
  const sx = box.width / width
  if (box.height <= 0 || height <= 0) return sx
  const sy = box.height / height
  const { align, meetOrSlice } = svg.preserveAspectRatio.baseVal
  if (align === SVGPreserveAspectRatio.SVG_PRESERVEASPECTRATIO_NONE) return sx
  return meetOrSlice === SVGPreserveAspectRatio.SVG_MEETORSLICE_SLICE
    ? Math.min(sx, sy)
    : Math.max(sx, sy)
}

// `onChange` runs after the properties are set, for figures that also fit labels to their boxes.
export function observeSvgScale(
  svgs: Iterable<SVGSVGElement>,
  property: string,
  onChange?: () => void,
): () => void {
  const observer = new ResizeObserver(entries => {
    for (const entry of entries) {
      const svg = entry.target as SVGSVGElement
      const u = unitsPerPx(svg, entry.contentRect.width, entry.contentRect.height)
      if (u !== null) svg.style.setProperty(property, u.toPrecision(5))
    }
    onChange?.()
  })
  for (const svg of svgs) observer.observe(svg)
  return () => observer.disconnect()
}
