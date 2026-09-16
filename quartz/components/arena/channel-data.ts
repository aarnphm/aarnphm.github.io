export const ARENA_CARD_PAGE_SIZE = 24

export type ArenaSectionName = 'pinned' | 'later' | 'blocks'

export interface ArenaCardPage {
  html: string
  offset: number
  total: number
}

export function arenaChannelAssets(channelSlug: string): string {
  return `/static/arena-channels/${channelSlug.split('/').map(encodeURIComponent).join('/')}`
}

export function arenaCardPageSource(base: string, section: ArenaSectionName, offset: number) {
  return `${base}/cards/${section}-${offset}.json`
}

export function arenaModalSource(base: string, blockId: string) {
  return `${base}/modals/${encodeURIComponent(blockId)}.html`
}

export function parseArenaCardPage(value: unknown): ArenaCardPage {
  if (
    !value ||
    typeof value !== 'object' ||
    !('html' in value) ||
    typeof value.html !== 'string' ||
    !('offset' in value) ||
    typeof value.offset !== 'number' ||
    !Number.isInteger(value.offset) ||
    !('total' in value) ||
    typeof value.total !== 'number' ||
    !Number.isInteger(value.total) ||
    value.offset < 0 ||
    value.offset % ARENA_CARD_PAGE_SIZE !== 0 ||
    value.offset >= value.total
  ) {
    throw new Error('Invalid Arena card page')
  }
  return { html: value.html, offset: value.offset, total: value.total }
}

export function parseArenaBlockOrder(value: string): string[] {
  const parsed: unknown = JSON.parse(value)
  if (!Array.isArray(parsed) || !parsed.every(id => typeof id === 'string')) {
    throw new Error('Invalid Arena block order')
  }
  return parsed
}
