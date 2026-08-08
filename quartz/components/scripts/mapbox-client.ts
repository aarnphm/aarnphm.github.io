const MAPBOX_SCRIPT_SRC = 'https://api.mapbox.com/mapbox-gl-js/v3.15.0/mapbox-gl.js'
const MAPBOX_STYLESHEET_HREF = 'https://api.mapbox.com/mapbox-gl-js/v3.15.0/mapbox-gl.css'
const MAPBOX_TOKEN_ENDPOINT = '/api/secrets?key=MAPBOX_API_KEY'

let mapboxTokenPromise: Promise<string | null> | null = null
let mapboxReady: Promise<any | null> | null = null

type MonochromeMap = {
  getStyle: () => { layers?: { id: string; type: string }[] } | undefined
  setPaintProperty: (layerId: string, property: string, value: string | number) => void
}

async function fetchMapboxToken() {
  try {
    const res = await fetch(MAPBOX_TOKEN_ENDPOINT, {
      headers: { Accept: 'application/json' },
      credentials: 'same-origin',
    })
    if (!res.ok) return null
    const { value } = (await res.json()) as { value?: string }
    return value?.trim() || null
  } catch {
    return null
  }
}

export function getMapboxToken() {
  return (mapboxTokenPromise ??= fetchMapboxToken().then(token => {
    if (!token) mapboxTokenPromise = null
    return token
  }))
}

export async function loadMapbox() {
  const token = await getMapboxToken()
  if (!token) return null

  let link = document.querySelector<HTMLLinkElement>(`link[href="${MAPBOX_STYLESHEET_HREF}"]`)
  if (!link) {
    link = document.createElement('link')
    link.rel = 'stylesheet'
    link.href = MAPBOX_STYLESHEET_HREF
    document.head.appendChild(link)
  }
  link.dataset.persist = 'true'

  if (window.mapboxgl) {
    window.mapboxgl.accessToken = token
    return window.mapboxgl
  }

  if (!mapboxReady) {
    mapboxReady = new Promise<any | null>(resolve => {
      let script = document.querySelector<HTMLScriptElement>(`script[src="${MAPBOX_SCRIPT_SRC}"]`)
      if (!script) {
        script = document.createElement('script')
        script.src = MAPBOX_SCRIPT_SRC
        script.async = true
        script.defer = true
        document.head.appendChild(script)
      }
      script.dataset.persist = 'true'

      script.addEventListener('load', () => resolve(window.mapboxgl), { once: true })
      script.addEventListener('error', () => resolve(null), { once: true })
    }).then(mapbox => {
      if (!mapbox) {
        mapboxReady = null
        document.querySelector(`script[src="${MAPBOX_SCRIPT_SRC}"]`)?.remove()
      }
      return mapbox
    })
  }

  const mapbox = await mapboxReady
  if (mapbox) mapbox.accessToken = token
  return mapbox
}

export function applyMonochromeMapPalette(map: MonochromeMap, theme: 'light' | 'dark' = 'light') {
  const palette =
    theme === 'dark'
      ? {
          background: '#100f0f',
          water: '#20282d',
          fill: '#181715',
          line: '#57534d',
          symbol: '#878580',
        }
      : {
          background: '#fff9f3',
          water: '#e2e8ee',
          fill: '#fef6ee',
          line: '#cbbfb1',
          symbol: '#7c7468',
        }
  const layers = map.getStyle()?.layers ?? []
  for (const layer of layers) {
    const { id, type } = layer
    if (type === 'background') {
      map.setPaintProperty(id, 'background-color', palette.background)
    } else if (type === 'fill') {
      const isWater = id.includes('water')
      map.setPaintProperty(id, 'fill-color', isWater ? palette.water : palette.fill)
      map.setPaintProperty(id, 'fill-opacity', isWater ? 0.96 : 0.85)
    } else if (type === 'line') {
      map.setPaintProperty(id, 'line-color', palette.line)
      map.setPaintProperty(id, 'line-opacity', 0.35)
    } else if (type === 'symbol') {
      map.setPaintProperty(id, 'text-color', palette.symbol)
      map.setPaintProperty(id, 'icon-color', palette.symbol)
    } else if (type === 'circle') {
      map.setPaintProperty(id, 'circle-color', palette.symbol)
      map.setPaintProperty(id, 'circle-opacity', 0.4)
    }
  }
}
