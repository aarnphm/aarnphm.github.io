export interface SaunaLocation {
  name: string
  latitude: number | null
  longitude: number | null
}

// Coordinates from the map embeds on https://www.othership.us/adelaide and /yorkville.
export const SAUNA_LOCATIONS: readonly SaunaLocation[] = [
  { name: 'Othership Adelaide', latitude: 43.6460984, longitude: -79.3977406 },
  { name: 'Othership Yorkville', latitude: 43.6694504, longitude: -79.3922208 },
]
