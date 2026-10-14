export interface MasonrySize {
  width: number
  height: number
}

interface Rectangle extends MasonrySize {
  x: number
  y: number
}

export interface MasonryPlacement extends Rectangle {
  index: number
  clearance: number
}

function overlaps(a: Rectangle, b: Rectangle): boolean {
  return a.x < b.x + b.width && a.x + a.width > b.x && a.y < b.y + b.height && a.y + a.height > b.y
}

function contains(outer: Rectangle, inner: Rectangle): boolean {
  return (
    outer.x <= inner.x &&
    outer.y <= inner.y &&
    outer.x + outer.width >= inner.x + inner.width &&
    outer.y + outer.height >= inner.y + inner.height
  )
}

// Free rectangles can overlap. Subtract each placement from all of them before
// pruning contained regions, as in MaxRects: https://github.com/juj/RectangleBinPack
function subtract(rectangles: Rectangle[], used: Rectangle, minimumClearance: number): Rectangle[] {
  const unchanged: Rectangle[] = []
  const fragments: Rectangle[] = []
  for (const free of rectangles) {
    if (!overlaps(free, used)) {
      unchanged.push(free)
      continue
    }
    if (used.x > free.x) {
      fragments.push({ ...free, width: used.x - free.x })
    }
    if (used.x + used.width < free.x + free.width) {
      fragments.push({
        ...free,
        x: used.x + used.width,
        width: free.x + free.width - used.x - used.width,
      })
    }
    if (used.y > free.y) {
      fragments.push({ ...free, height: used.y - free.y })
    }
    if (used.y + used.height < free.y + free.height) {
      fragments.push({
        ...free,
        y: used.y + used.height,
        height: free.y + free.height - used.y - used.height,
      })
    }
  }

  const usable = fragments.filter(
    rect => rect.width >= minimumClearance + 1 && rect.height >= minimumClearance + 1,
  )
  // Unchanged regions are already maximal; only the split fragments need pruning.
  return [
    ...unchanged,
    ...usable.filter(
      (rect, index) =>
        !unchanged.some(other => contains(other, rect)) &&
        !usable.some(
          (other, otherIndex) =>
            otherIndex !== index &&
            contains(other, rect) &&
            (!contains(rect, other) || otherIndex < index),
        ),
    ),
  ]
}

export function createMasonryPacking(
  images: readonly MasonrySize[],
  width: number,
  viewportHeight: number,
  gap: number,
) {
  const breathingRoom = Math.min(32, width * 0.04)
  const minimumClearance = gap + breathingRoom * 0.6
  const pending = images.map((image, index) => {
    const sourceWidth = Number.isFinite(image.width) && image.width > 0 ? image.width : 800
    const sourceHeight = Number.isFinite(image.height) && image.height > 0 ? image.height : 600
    const ratio = Math.max(sourceWidth / sourceHeight, sourceHeight / sourceWidth)
    const area = Math.min(ratio > 8 ? 40000 : ratio > 4 ? 60000 : 150000, (width * width) / 5)
    const scale = Math.min(
      Math.sqrt((area * (0.65 + Math.random() * 0.75)) / (sourceWidth * sourceHeight)),
      width / sourceWidth,
      Math.max(320, viewportHeight * 0.85) / sourceHeight,
    )
    return {
      index,
      width: sourceWidth * scale,
      height: sourceHeight * scale,
      clearance: gap + breathingRoom * (0.6 + Math.random() * 0.8),
    }
  })
  let canvasHeight = Math.max(600, viewportHeight)
  let occupiedHeight = 0
  let placed = 0
  let backfilled = 0
  let free: Rectangle[] = [{ x: 0, y: 0, width: width + gap, height: canvasHeight + gap }]

  function findPlacement(scanAll = false): MasonryPlacement | undefined {
    const scatter = placed < 6
    const candidates = pending.slice(0, scatter ? 1 : scanAll ? pending.length : 48)
    const spaces = free.toSorted((a, b) =>
      scatter
        ? b.width * b.height - a.width * a.height
        : a.y - b.y || b.width * b.height - a.width * a.height,
    )
    for (const space of spaces) {
      let best: MasonryPlacement | undefined
      for (const tile of candidates) {
        const scale = Math.min(
          1,
          (space.width - tile.clearance) / tile.width,
          (space.height - tile.clearance) / tile.height,
        )
        if (scale < (scatter ? 0.75 : 0.35)) continue
        const tileWidth = tile.width * scale
        const tileHeight = tile.height * scale
        if (best && tileWidth * tileHeight <= best.width * best.height) continue
        const slackX = Math.max(0, space.width - tileWidth - tile.clearance)
        const slackY = Math.max(0, space.height - tileHeight - tile.clearance)
        best = {
          index: tile.index,
          clearance: tile.clearance,
          width: tileWidth,
          height: tileHeight,
          x: space.x + slackX * (scatter ? Math.random() : Math.random() < 0.5 ? 0 : 1),
          y: space.y + Math.random() * (scatter ? slackY : Math.min(slackY, breathingRoom * 0.5)),
        }
      }
      if (best) return best
    }
  }

  return {
    get height() {
      return occupiedHeight
    },
    get remaining() {
      return pending.length
    },
    get backfilled() {
      return backfilled
    },
    next(): MasonryPlacement | undefined {
      const first = pending[0]
      if (!first) return
      let placement = findPlacement()
      if (!placement && placed >= 6) placement = findPlacement(true)
      if (!placement) {
        const bandHeight = Math.max(320, first.height + first.clearance)
        free.push({ x: 0, y: canvasHeight + gap, width: width + gap, height: bandHeight })
        canvasHeight += bandHeight
        placement = { ...first, x: 0, y: canvasHeight + gap - bandHeight }
      }

      if (placement.y + placement.height <= occupiedHeight) backfilled++
      occupiedHeight = Math.max(occupiedHeight, placement.y + placement.height)
      free = subtract(
        free,
        {
          ...placement,
          width: placement.width + placement.clearance,
          height: placement.height + placement.clearance,
        },
        minimumClearance,
      )
      pending.splice(
        pending.findIndex(tile => tile.index === placement.index),
        1,
      )
      placed++
      return placement
    },
  }
}
