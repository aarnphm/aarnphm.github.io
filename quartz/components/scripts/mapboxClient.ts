const MAPBOX_SCRIPT_SRC = "https://api.mapbox.com/mapbox-gl-js/v3.15.0/mapbox-gl.js"
const MAPBOX_STYLESHEET_HREF = "https://api.mapbox.com/mapbox-gl-js/v3.15.0/mapbox-gl.css"
const MAPBOX_TOKEN_ENDPOINT = "/api/secrets?key=MAPBOX_API_KEY"

let mapboxTokenPromise: Promise<string | null> | null = null
let mapboxReady: Promise<any | null> | null = null

async function fetchMapboxToken(): Promise<string | null> {
  try {
    const response = await fetch(MAPBOX_TOKEN_ENDPOINT, {
      method: "GET",
      headers: { Accept: "application/json" },
      credentials: "same-origin",
    })

    if (!response.ok) {
      return null
    }

    const payload = (await response.json().catch(() => null)) as { value?: unknown } | null
    if (!payload || typeof payload.value !== "string") {
      return null
    }

    const token = payload.value.trim()
    return token.length > 0 ? token : null
  } catch (error) {
    console.error(error)
    return null
  }
}

export async function getMapboxToken(): Promise<string | null> {
  if (!mapboxTokenPromise) {
    mapboxTokenPromise = fetchMapboxToken()
  }

  const token = await mapboxTokenPromise
  if (!token) {
    mapboxTokenPromise = Promise.resolve(null)
  }
  return token
}

export async function loadMapbox(): Promise<any | null> {
  const token = await getMapboxToken()
  if (!token) {
    return null
  }

  const ensureStylesheet = () => {
    if (document.querySelector(`link[href="${MAPBOX_STYLESHEET_HREF}"]`)) {
      return
    }
    const link = document.createElement("link")
    link.rel = "stylesheet"
    link.href = MAPBOX_STYLESHEET_HREF
    document.head.appendChild(link)
  }
  ensureStylesheet()

  const applyToken = (mapboxgl: any | null) => {
    if (mapboxgl && mapboxgl.Map) {
      if (window.mapboxgl && window.mapboxgl.accessToken !== token) {
        window.mapboxgl.accessToken = token
      }
      return mapboxgl
    }
    return null
  }

  const immediate = applyToken(window.mapboxgl ?? null)
  if (immediate) {
    return immediate
  }

  if (!mapboxReady) {
    let script: HTMLScriptElement | null = document.querySelector(
      `script[src="${MAPBOX_SCRIPT_SRC}"]`,
    )

    if (!script) {
      script = document.createElement("script")
      script.src = MAPBOX_SCRIPT_SRC
      script.async = true
      script.defer = true
      document.head.appendChild(script)
    }

    if (script) {
      mapboxReady = new Promise((resolve) => {
        const resolveWithMap = () => resolve(window.mapboxgl ?? null)
        const state = (script as any).readyState as string | undefined
        if (state === "complete" || state === "loaded") {
          resolveWithMap()
        } else {
          script!.addEventListener("load", resolveWithMap, { once: true })
          script!.addEventListener("error", () => resolve(null), { once: true })
        }
      })
    } else {
      mapboxReady = Promise.resolve(null)
    }
  }

  const loaded = await mapboxReady
  return applyToken(loaded)
}
