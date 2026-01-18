import { annotate } from "rough-notation"
import type { RoughAnnotation } from "rough-notation/lib/model"

let annotations: RoughAnnotation[] = []

function cleanup() {
  annotations.forEach((annotation) => annotation.remove())
  annotations = []
}

function setupMarkers() {
  cleanup()
  window.addCleanup(cleanup)

  const markers = document.querySelectorAll<HTMLDivElement>(".marker")
  if (markers.length === 0) return

  const cssVars = ["--rose", "--love", "--lime", "--gold", "--pine", "--foam", "--iris"] as const

  const style = getComputedStyle(document.documentElement)
  const computedStyleMap = cssVars.reduce(
    (acc, key) => {
      acc[key] = style.getPropertyValue(key)
      return acc
    },
    {} as Record<(typeof cssVars)[number], string>,
  )

  const intensityColors: Record<string, string> = {
    h1: computedStyleMap["--rose"],
    h2: computedStyleMap["--love"],
    h3: computedStyleMap["--lime"],
    h4: computedStyleMap["--gold"],
    h5: computedStyleMap["--pine"],
    h6: computedStyleMap["--foam"],
    h7: computedStyleMap["--iris"],
  }

  for (const marker of markers) {
    const intensityClass = Array.from(marker.classList).find((cls) => cls.startsWith("marker-h"))
    if (!intensityClass) continue

    const intensity = intensityClass.replace("marker-", "")
    const color = intensityColors[intensity] || intensityColors["h1"]

    const annotation = annotate(marker, {
      type: "box",
      color,
      iterations: 2,
      animate: false,
      multiline: true,
      brackets: ["left", "right"],
    })

    annotation.show()
    annotations.push(annotation)
  }
}

document.addEventListener("nav", setupMarkers)
document.addEventListener("contentdecrypted", setupMarkers)
document.addEventListener("readermodechange", setupMarkers)
document.addEventListener("resize", setupMarkers)
