import type { CampSettings } from './scene-settings'
import { onDryLand, pondDistance, type Landscape, type Point } from './landscape'
import { trailRandom } from './random'

export type CampSite = Point & { id: number; tents: Point[] }

export function campSites(landscape: Landscape, seed: number, settings: CampSettings): CampSite[] {
  const { camp, width, meadowTop, height, pond } = landscape
  const random = trailRandom(seed ^ 0x724bd509)
  const sites: CampSite[] = []
  for (let id = 0; id < settings.sites; id++) {
    let point = camp
    if (id) {
      for (let attempt = 0; attempt < 80; attempt++) {
        point = onDryLand(
          {
            x: width * (id === 1 ? 0.3 : 0.78) + (random() - 0.5) * Math.min(90, width * 0.15),
            // A camp at the pond's latitude lands inside its dry-land margin and is pushed out to
            // the rim, at the scene's edge; the extra camps keep above that margin.
            y:
              Math.max(
                meadowTop + 92,
                Math.min(
                  camp.y - (width < 700 ? 130 * id : 90 + id * 50),
                  pond.y - pond.ry * 2.6 - 20,
                ),
              ) +
              (random() - 0.5) * 20,
          },
          landscape,
          2.6,
        )
        if (
          point.x > 55 &&
          point.x < width - 55 &&
          sites.every(site => Math.hypot(site.x - point.x, site.y - point.y) > 135) &&
          pondDistance(point, pond) > 2.6
        )
          break
      }
    }
    const tents = Array.from({ length: settings.tents }, (_, index) => {
      const side = index % 2 ? 1 : -1
      return onDryLand(
        {
          x: Math.max(28, Math.min(width - 28, point.x + side * (55 + Math.floor(index / 2) * 42))),
          y: Math.min(height - 24, point.y + (index > 1 ? -55 : -35)),
        },
        landscape,
        2,
      )
    })
    sites.push({ ...point, id, tents })
  }
  return sites
}
