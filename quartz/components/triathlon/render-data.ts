import type { Analytics } from '../../plugins/stores/analytics'
import type { WeatherSnapshot } from '../../plugins/stores/weather'
import type { HighParkLoops } from '../../util/high-park-loops'
import type { TriathlonCalendar } from '../../util/triathlon-calendar'
import type { TriathlonEquipmentUsage } from '../../util/triathlon-equipment'

export interface TriathlonRenderData {
  highParkLoops?: HighParkLoops
  analytics: Analytics
  weather: WeatherSnapshot | null
  equipment?: Record<string, TriathlonEquipmentUsage>
  calendar?: TriathlonCalendar | null
  calendars?: readonly TriathlonCalendar[]
}
