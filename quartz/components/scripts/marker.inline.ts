import { annotate } from "rough-notation"
import type { RoughAnnotation } from "rough-notation/lib/model"

// map intensity levels to colors
var intensityColors: Record<string, string> = {
  h1: "rgb(209, 77, 65)", // --rose
  h2: "rgb(218, 112, 44)", // --love
  h3: "rgb(135, 154, 57)", // --lime
  h4: "rgb(208, 162, 21)", // --gold
  h5: "rgb(58, 169, 159)", // --pine
  h6: "rgb(67, 133, 190)", // --foam
  h7: "rgb(139, 126, 200)", // --iris
}

var annotations: RoughAnnotation[] = []

async function setupMarkers() {
  // clean up previous annotations
  annotations.forEach((annotation) => annotation.remove())
  annotations = []

  // find all marker elements
  var markers = document.querySelectorAll<HTMLDivElement>(".marker")

  markers.forEach((marker) => {
    // extract intensity level from class (marker-h1, marker-h2, etc.)
    var classList = Array.from(marker.classList)
    var intensityClass = classList.find((cls) => cls.startsWith("marker-h"))

    if (!intensityClass) return

    var intensity = intensityClass.replace("marker-", "")
    var color = intensityColors[intensity] || intensityColors["h1"]

    // create rough notation annotation
    var annotation = annotate(marker, {
      type: "bracket",
      color,
      iterations: 2,
      animate: false,
      multiline: true,
      brackets: ["left", "bottom"],
    })

    annotation.show()
    annotations.push(annotation)
  })
}

document.addEventListener("nav", setupMarkers)
document.addEventListener("content-decrypted", setupMarkers)
