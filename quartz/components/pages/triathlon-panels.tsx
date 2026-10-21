import { ShortcutHint } from '../triathlon/shell/ShortcutHint'
export { OnTreePanel } from './triathlon-panels/activity-log'
export { AnalyticsPanel, CalendarOverlay, MapPanel } from './triathlon-panels/dialog-views'
export { CONVERSIONS, GEAR, GearPanel, ToolsPanel } from './triathlon-panels/equipment-tools'
export { CalcPanel } from './triathlon-panels/calculators'

const NAV = [
  ['tools', 'tools', 'g g'],
  ['calc', 'calculator', 'g c'],
  ['analytics', 'analytics', 'g a'],
  ['maps', 'maps', 'g m'],
  ['calendar', 'calendar', 'g e'],
  ['on', 'on', 'g o'],
] as const

export type TriView = (typeof NAV)[number][0]

export const TriathlonSubnav = ({ active, root }: { active?: TriView; root: string }) => (
  <nav class="tri-subnav" aria-label="triathlon sections">
    <a class="tri-subnav-home tri-key-anchor" href={`${root}/triathlon`}>
      ← triathlon
      <ShortcutHint>g h</ShortcutHint>
    </a>
    <span class="tri-subnav-links">
      {NAV.map(([slug, label, shortcut]) => (
        <a
          class="tri-subnav-link tri-key-anchor"
          href={`${root}/triathlon/${slug}`}
          aria-current={slug === active ? 'page' : undefined}
        >
          <span data-i18n={label}>{label}</span>
          <ShortcutHint>{shortcut}</ShortcutHint>
        </a>
      ))}
    </span>
  </nav>
)
