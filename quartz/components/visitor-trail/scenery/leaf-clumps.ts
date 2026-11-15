// Splits a crown's foliage into the pieces the wind moves. A broadleaf crown breaks into leaf
// clumps around its highlights; a conifer breaks into tiers, cut where the crown is narrowest.

type LeafCell = { x: number; y: number; tone: string }
export type LeafClump = {
  // The clump's centre, in the tree's units.
  x: number
  y: number
  // How freely it moves: 0.35 at the middle of the crown, 1 at its edge.
  loose: number
}

const tierLoose = 0.3

function hashCell(x: number, y: number) {
  let hash = Math.imul(x, 374761393) + Math.imul(y, 668265263)
  hash = Math.imul(hash ^ (hash >>> 13), 1274126177)
  return (hash ^ (hash >>> 16)) >>> 0
}

// Returns the clumps in drawing order and, for each cell in the order given, the clump it is in.
export function leafClumps(cells: LeafCell[], pixel: number, tiers: boolean) {
  const count = cells.length
  const owner = new Int8Array(count)
  if (!count) return { clumps: [] as LeafClump[], owner }
  const gx = new Int32Array(count)
  const gy = new Int32Array(count)
  let top = Infinity
  let bottom = -Infinity
  let sumX = 0
  let sumY = 0
  for (let index = 0; index < count; index++) {
    gx[index] = Math.round(cells[index].x / pixel)
    gy[index] = Math.round(cells[index].y / pixel)
    top = Math.min(top, gy[index])
    bottom = Math.max(bottom, gy[index])
    sumX += gx[index]
    sumY += gy[index]
  }
  let groups = 1
  if (tiers) {
    // A conifer's rows widen down each whorl and narrow where the next one starts.
    const rows = bottom - top + 1
    const widths = new Int32Array(rows)
    for (let index = 0; index < count; index++) widths[gy[index] - top]++
    groups = Math.max(1, Math.min(5, Math.round(rows / 5)))
    const reach = Math.max(1, Math.floor(rows / (groups * 3)))
    const cuts: number[] = []
    for (let tier = 1; tier < groups; tier++) {
      const ideal = Math.round((rows * tier) / groups)
      let cut = ideal
      for (let row = ideal - reach; row <= ideal + reach; row++)
        if (row > 0 && row < rows && widths[row] < widths[cut]) cut = row
      cuts.push(cut)
    }
    for (let index = 0; index < count; index++) {
      let tier = 0
      for (const cut of cuts) if (gy[index] - top >= cut) tier++
      // The lowest tier is drawn first, so each tier overlaps the one below it.
      owner[index] = groups - 1 - tier
    }
  } else {
    const target = Math.max(1, Math.min(6, Math.round(Math.sqrt(count) / 3.2)))
    const middle = sumX / count
    // The first seed is the highlight nearest the top of the crown, and each one after it is the
    // cell farthest from the seeds already placed.
    let first = 0
    let best = Infinity
    for (let index = 0; index < count; index++) {
      const rank =
        gy[index] * 2 + Math.abs(gx[index] - middle) + (cells[index].tone === 'light' ? 0 : 1e6)
      if (rank < best) {
        best = rank
        first = index
      }
    }
    const seedX = [gx[first]]
    const seedY = [gy[first]]
    const span = (index: number, seed: number) =>
      Math.hypot(gx[index] - seedX[seed], (gy[index] - seedY[seed]) * 1.25)
    const near = new Float64Array(count).fill(Infinity)
    const apart = Math.max(3, Math.sqrt(count / target) * 0.8)
    while (seedX.length < target) {
      let pick = -1
      let far = apart
      for (let index = 0; index < count; index++) {
        near[index] = Math.min(near[index], span(index, seedX.length - 1))
        if (near[index] > far) {
          far = near[index]
          pick = index
        }
      }
      if (pick < 0) break
      seedX.push(gx[pick])
      seedY.push(gy[pick])
    }
    groups = seedX.length
    const dropped = new Set<number>()
    const assign = (ragged: boolean) => {
      for (let index = 0; index < count; index++) {
        // A cell of jitter keeps the seams between clumps ragged.
        const jitter = ragged ? ((hashCell(gx[index], gy[index]) % 5) - 2) * 0.3 : 0
        let own = 0
        let least = Infinity
        for (let seed = 0; seed < groups; seed++) {
          if (dropped.has(seed)) continue
          const distance = span(index, seed) + (seed % 2 ? jitter : -jitter)
          if (distance < least) {
            least = distance
            own = seed
          }
        }
        owner[index] = own
      }
    }
    // Seeds from the farthest cells sit on the crown's rim. A few rounds of moving each seed to
    // the middle of its cells pulls them in, which leaves compact clumps and one near the middle.
    const sizes = new Int32Array(groups)
    for (let round = 0; round < 4; round++) {
      assign(false)
      const sums = new Float64Array(groups * 2)
      sizes.fill(0)
      for (let index = 0; index < count; index++) {
        sums[owner[index] * 2] += gx[index]
        sums[owner[index] * 2 + 1] += gy[index]
        sizes[owner[index]]++
      }
      for (let seed = 0; seed < groups; seed++)
        if (sizes[seed]) {
          seedX[seed] = sums[seed * 2] / sizes[seed]
          seedY[seed] = sums[seed * 2 + 1] / sizes[seed]
        }
    }
    assign(true)
    sizes.fill(0)
    for (let index = 0; index < count; index++) sizes[owner[index]]++
    // A clump of a few cells reads as noise when it moves; its cells join their neighbours.
    const least = Math.max(4, count * 0.06)
    for (let seed = 0; seed < groups; seed++)
      if (sizes[seed] < least && dropped.size < groups - 1) dropped.add(seed)
    if (dropped.size) assign(true)
  }
  const centres = Array.from({ length: groups }, () => ({ x: 0, y: 0, cells: 0 }))
  for (let index = 0; index < count; index++) {
    const centre = centres[owner[index]]
    centre.x += gx[index]
    centre.y += gy[index]
    centre.cells++
  }
  const middleX = sumX / count
  const middleY = sumY / count
  let far = 0
  const drawn = centres
    .map((centre, key) => {
      const x = centre.cells ? centre.x / centre.cells : 0
      const y = centre.cells ? centre.y / centre.cells : 0
      const distance = Math.hypot(x - middleX, y - middleY)
      if (centre.cells) far = Math.max(far, distance)
      return { key, x, y, distance, cells: centre.cells }
    })
    .filter(centre => centre.cells)
  // Stiff inner clumps go down first and the loose outer ones over them. Tiers keep their order.
  if (!tiers) drawn.sort((a, b) => a.distance - b.distance)
  const place = new Int8Array(groups)
  const clumps: LeafClump[] = drawn.map((centre, index) => {
    place[centre.key] = index
    return {
      x: (centre.x + 0.5) * pixel,
      y: (centre.y + 0.5) * pixel,
      loose: tiers ? tierLoose : far > 0 ? 0.35 + (0.65 * centre.distance) / far : 0.7,
    }
  })
  for (let index = 0; index < count; index++) owner[index] = place[owner[index]]
  return { clumps, owner }
}
