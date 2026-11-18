import fs from 'node:fs/promises'
import { joinSegments } from './path'
import { syncMaintenanceDistances } from './triathlon-maintenance-sync'

const TRIATHLON_ROUTE_SOURCE = joinSegments('content', 'triathlon.md')

export async function refreshTriathlonRouteSource(now = new Date()): Promise<void> {
  await syncMaintenanceDistances(TRIATHLON_ROUTE_SOURCE)
  await fs.utimes(TRIATHLON_ROUTE_SOURCE, now, now)
}
