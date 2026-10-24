export type WorkspaceTab = 'source' | 'video' | 'scratchpad' | 'notes'

const shortcuts: Record<WorkspaceTab, string> = {
  source: 'S',
  video: 'V',
  scratchpad: 'P',
  notes: 'N',
}

export function workspaceTabs(hasSource: boolean, hasVideo: boolean): WorkspaceTab[] {
  return [
    ...(hasSource ? (['source'] as const) : []),
    ...(hasVideo ? (['video'] as const) : []),
    'scratchpad',
    'notes',
  ]
}

export function initialWorkspaceTab(hasSource: boolean, hasVideo: boolean): WorkspaceTab {
  return workspaceTabs(hasSource, hasVideo)[0]
}

interface HeaderProps {
  hasSource: boolean
  hasVideo: boolean
  tab: WorkspaceTab
  noteCount?: number
  onSelect?(tab: WorkspaceTab): void
  onExport?(): void
  onLoadSaved?(): void
}

export function WorkspaceHeader({
  hasSource,
  hasVideo,
  tab,
  noteCount = 0,
  onSelect,
  onExport,
  onLoadSaved,
}: HeaderProps) {
  const tabs = workspaceTabs(hasSource, hasVideo)
  return (
    <>
      <h2 id="course-workspace-heading" class="visually-hidden">
        lecture workspace
      </h2>
      {/* A div: the site header rules (base.scss) push the second-to-last child right. */}
      <div class="course-workspace-header">
        <div
          role="tablist"
          aria-label="Lecture workspace"
          class="tri-map-tablist course-workspace-tabs"
          onKeyDown={event => {
            if (!onSelect || event.ctrlKey || event.metaKey || event.altKey || event.isComposing)
              return
            const at = tabs.indexOf(tab)
            const key = event.key.toLowerCase()
            const next =
              event.key === 'ArrowRight'
                ? tabs[(at + 1) % tabs.length]
                : event.key === 'ArrowLeft'
                  ? tabs[(at + tabs.length - 1) % tabs.length]
                  : event.key === 'Home'
                    ? tabs[0]
                    : event.key === 'End'
                      ? tabs[tabs.length - 1]
                      : tabs.find(value => shortcuts[value].toLowerCase() === key)
            if (!next) return
            event.preventDefault()
            onSelect(next)
            document.getElementById(`course-workspace-tab-${next}`)?.focus()
          }}
        >
          {tabs.map(value => {
            const label = value === 'notes' ? `notes ${noteCount}` : value
            const short =
              value === 'notes' && noteCount > 0
                ? `${shortcuts[value]} ${noteCount}`
                : shortcuts[value]
            return (
              <button
                type="button"
                role="tab"
                class="tri-map-tab"
                id={`course-workspace-tab-${value}`}
                aria-selected={tab === value}
                aria-controls={`course-workspace-${value}`}
                aria-label={label}
                tabIndex={tab === value ? 0 : -1}
                disabled={!onSelect}
                title={`${value} (${shortcuts[value].toLowerCase()})`}
                style={{
                  '--tri-map-tab-shortcut-width': `${short.length}ch`,
                  '--tri-map-tab-label-width': `${label.length}ch`,
                }}
                onClick={() => onSelect?.(value)}
              >
                <span class="tri-map-tab-shortcut" aria-hidden="true">
                  {short}
                </span>
                <span class="tri-map-tab-label" aria-hidden="true">
                  {label}
                </span>
              </button>
            )
          })}
        </div>
        <div class="course-workspace-actions">
          {onLoadSaved && (
            <button type="button" onClick={onLoadSaved}>
              load saved
            </button>
          )}
          <button type="button" title="Export Markdown" disabled={!onExport} onClick={onExport}>
            export .md
          </button>
        </div>
      </div>
    </>
  )
}
