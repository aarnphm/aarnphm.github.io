import fs from 'node:fs/promises'
import { joinSegments } from './path'

const TRIATHLON_ROUTE_SOURCE = joinSegments('content', 'triathlon.md')

export async function refreshTriathlonRouteSource(now = new Date()): Promise<void> {
  await fs.utimes(TRIATHLON_ROUTE_SOURCE, now, now)
}
