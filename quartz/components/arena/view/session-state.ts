import type { ArenaExternalEmbedMode } from '../../../util/arena-embed'
import { mountArenaChannel } from '../channel'
import { createArenaExternalEmbeds } from '../external-embed'

export const session: {
  currentBlockIndex: number
  totalBlocks: number
  scrollLockState: { x: number; y: number } | null
  searchInput: HTMLInputElement | null
  activeResultIndex: number | null
  registerCleanup: ((cleanup: () => void) => void) | null
  externalEmbeds: ReturnType<typeof createArenaExternalEmbeds> | null
  channelViews: ReturnType<typeof mountArenaChannel>
  modalRequest: AbortController | null
  searchIndex: SearchIndexItem[]
  searchDebounceTimer: number | undefined
  arenaSearchData: ArenaSearchIndex | null
  searchRequest: AbortController | null
} = {
  currentBlockIndex: 0,
  totalBlocks: 0,
  scrollLockState: null,
  searchInput: null,
  activeResultIndex: null,
  registerCleanup: null,
  externalEmbeds: null,
  channelViews: null,
  modalRequest: null,
  searchIndex: [],
  searchDebounceTimer: undefined,
  arenaSearchData: null,
  searchRequest: null,
}

export const addCleanup = (cleanup: () => void) => {
  if (session.registerCleanup) {
    session.registerCleanup(cleanup)
  }
}

// Search functionality - JSON-based
export interface ArenaBlockSearchable {
  id: string
  entryId?: string
  channelSlug: string
  channelName: string
  content: string
  title?: string
  titleHtml?: string
  blockHtml?: string
  url?: string
  highlighted: boolean
  embedHtml?: string
  metadata?: Record<string, string>
  coordinates?: { lon: number; lat: number }
  internalSlug?: string
  internalHref?: string
  internalHash?: string
  tags?: string[]
  subItems?: ArenaBlockSearchable[]
  hasModalInDom: boolean
  embedMode?: ArenaExternalEmbedMode
}

interface ArenaChannelSearchable {
  id: string
  name: string
  slug: string
  blockCount: number
}

export interface ArenaSearchIndex {
  version: string
  blocks: ArenaBlockSearchable[]
  channels: ArenaChannelSearchable[]
}

export interface SearchIndexItem {
  blockId: string
  channelSlug?: string
  channelName?: string
  title: string
  content: string
  highlighted: boolean
  hasModalInDom: boolean
  tags?: string[]
  matchedTags?: string[]
}
