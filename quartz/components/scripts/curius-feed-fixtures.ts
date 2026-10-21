import type { Link, Trail } from '../../types/curius'

export const trail: Trail = {
  id: 1,
  trailName: 'Fixture trail',
  ownerId: 1,
  description: 'Saved articles',
  colorHex: '',
  emojiUnicode: '',
  flipped: null,
  hash: 'fixture-trail',
  slug: 'fixture-trail',
  createdDate: '2026-09-01T00:00:00Z',
  users: [],
}

export function savedLink(id: number, trails: Trail[] = []): Link {
  return {
    id,
    link: `https://example.org/article/${id}`,
    title: `Saved article ${id}`,
    favorite: false,
    snippet: `Saved text ${id}`,
    toRead: false,
    createdBy: 1,
    metadata: { full_text: '', author: '', page_type: '' },
    lastCrawled: null,
    createdDate: '2026-09-01T00:00:00Z',
    modifiedDate: '2026-09-01T00:00:00Z',
    trails,
    comments: [],
    mentions: [],
    topics: [],
    highlights: [],
  }
}
