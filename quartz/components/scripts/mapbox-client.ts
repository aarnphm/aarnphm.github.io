const MAPBOX_SCRIPT_SRC = "https://api.mapbox.com/mapbox-gl-js/v3.15.0/mapbox-gl.js"
const MAPBOX_STYLESHEET_HREF = "https://api.mapbox.com/mapbox-gl-js/v3.15.0/mapbox-gl.css"
const MAPBOX_TOKEN_ENDPOINT = "/api/secrets?key=MAPBOX_API_KEY"

let mapboxTokenPromise: Promise<string | null> | null = null
let mapboxReady: Promise<any | null> | null = null

async function fetchMapboxToken() {
  try {
    const res = await fetch(MAPBOX_TOKEN_ENDPOINT, {
      headers: { Accept: "application/json" },
      credentials: "same-origin",
    })
    if (!res.ok) return null
    const { value } = (await res.json()) as { value?: string }
    return value?.trim() || null
  } catch {
    return null
  }
}

export function getMapboxToken() {
  return (mapboxTokenPromise ??= fetchMapboxToken())
}

export async function loadMapbox() {
  const token = await getMapboxToken()
  if (!token) return null

  if (!document.querySelector(`link[href="${MAPBOX_STYLESHEET_HREF}"]`)) {
    const link = document.createElement("link")
    link.rel = "stylesheet"
    link.href = MAPBOX_STYLESHEET_HREF
    document.head.appendChild(link)
  }

  if (window.mapboxgl) {
    window.mapboxgl.accessToken = token
    return window.mapboxgl
  }

  if (!mapboxReady) {
    mapboxReady = new Promise((resolve) => {
      let script = document.querySelector(`script[src="${MAPBOX_SCRIPT_SRC}"]`) as HTMLScriptElement
      if (!script) {
        script = document.createElement("script")
        script.src = MAPBOX_SCRIPT_SRC
        script.async = true
        script.defer = true
        document.head.appendChild(script)
      }

      script.addEventListener("load", () => resolve(window.mapboxgl), { once: true })
      script.addEventListener("error", () => resolve(null), { once: true })
    })
  }

  const mapbox = await mapboxReady
  if (mapbox) mapbox.accessToken = token
  return mapbox
}

export function applyMonochromeMapPalette(map: any) {
  const layers = map.getStyle()?.layers ?? []
  for (const layer of layers) {
    const { id, type } = layer
    if (type === "background") {
      map.setPaintProperty(id, "background-color", "#fff9f3")
    } else if (type === "fill") {
      const isWater = id.includes("water")
      map.setPaintProperty(id, "fill-color", isWater ? "#e2e8ee" : "#fef6ee")
      map.setPaintProperty(id, "fill-opacity", isWater ? 0.96 : 0.85)
    } else if (type === "line") {
      map.setPaintProperty(id, "line-color", "#cbbfb1")
      map.setPaintProperty(id, "line-opacity", 0.35)
    } else if (type === "symbol") {
      map.setPaintProperty(id, "text-color", "#7c7468")
      map.setPaintProperty(id, "icon-color", "#7c7468")
    } else if (type === "circle") {
      map.setPaintProperty(id, "circle-color", "#7c7468")
      map.setPaintProperty(id, "circle-opacity", 0.4)
    }
  }
}
