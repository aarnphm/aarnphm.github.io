// Where the site pet last stood on screen, in client pixels, kept for the next page in the tab. The
// pet comes back near it after navigation; the 404 scene's pixel pet hops onto the clock from it,
// and leaves the clock's centre behind when it jumps through.
const key = 'garden:pet-position:v2'

export type Spot = { x: number; y: number }

export function readPetSpot(): Spot | null {
  const [x, y] = (sessionStorage.getItem(key) ?? '').split(',').map(Number)
  return Number.isFinite(x) && Number.isFinite(y) && sessionStorage.getItem(key) ? { x, y } : null
}

export function writePetSpot(p: Spot) {
  sessionStorage.setItem(key, `${Math.round(p.x)},${Math.round(p.y)}`)
}
