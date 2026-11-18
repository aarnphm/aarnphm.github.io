/**
 * The coordinates asset a latent embed's live bench turns (static/latent-embed/<name>.bin):
 * int16 coords over the largest reach, then one byte per point for the source kind, then the
 * Matryoshka level at the features level, then the picked feature's heat when the view has one.
 * The build writes it beside the SVG poster; the embed's runtime reads it on intent, so the
 * canvas needs no vectors, worker or dictionaries to draw the first frame.
 */
export interface EmbedCoords {
  coords: Float32Array
  kinds: Uint8Array
  lvl: Uint8Array | null
  heat: Uint8Array | null
}

export function encodeCoords(
  coords: Float32Array,
  kinds: Uint8Array,
  lvl: Uint8Array | null | undefined,
  heat: Uint8Array | null | undefined,
): { bytes: Uint8Array; reach: number } {
  const n = coords.length / 3
  let reach = 0
  for (let i = 0; i < coords.length; i++) reach = Math.max(reach, Math.abs(coords[i]))
  reach ||= 1
  const q = new Int16Array(coords.length)
  for (let i = 0; i < coords.length; i++) q[i] = Math.round((coords[i] / reach) * 32767)
  const bytes = new Uint8Array(q.byteLength + n * (1 + (lvl ? 1 : 0) + (heat ? 1 : 0)))
  bytes.set(new Uint8Array(q.buffer), 0)
  let at = q.byteLength
  bytes.set(kinds.subarray(0, n), at)
  at += n
  if (lvl) {
    bytes.set(lvl.subarray(0, n), at)
    at += n
  }
  if (heat) bytes.set(heat.subarray(0, n), at)
  return { bytes, reach }
}

export function decodeCoords(
  buffer: ArrayBuffer | Uint8Array,
  n: number,
  reach: number,
  has: { lvl: boolean; heat: boolean },
): EmbedCoords {
  const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer)
  const want = n * 6 + n * (1 + Number(has.lvl) + Number(has.heat))
  if (bytes.byteLength !== want) throw new Error('latent embed coordinates have the wrong size')
  // copy: the int16 view needs an even offset, and the caller may hold a sliced buffer
  const q = new Int16Array(bytes.slice(0, n * 6).buffer)
  const coords = new Float32Array(n * 3)
  const scale = reach / 32767
  for (let i = 0; i < q.length; i++) coords[i] = q[i] * scale
  let at = n * 6
  const kinds = bytes.slice(at, at + n)
  at += n
  let lvl: Uint8Array | null = null
  if (has.lvl) {
    lvl = bytes.slice(at, at + n)
    at += n
  }
  const heat = has.heat ? bytes.slice(at, at + n) : null
  return { coords, kinds, lvl, heat }
}
