export type MapboxStyle = 'mono' | 'streets' | 'satellite'
export type MapboxTheme = 'light' | 'dark'

export const mapboxStyleUrl = (style: MapboxStyle, theme: MapboxTheme): string => {
  if (style === 'satellite') return 'mapbox://styles/mapbox/satellite-streets-v12'
  if (theme === 'dark') return 'mapbox://styles/mapbox/dark-v11'
  return style === 'streets'
    ? 'mapbox://styles/mapbox/streets-v12'
    : 'mapbox://styles/mapbox/light-v11'
}
