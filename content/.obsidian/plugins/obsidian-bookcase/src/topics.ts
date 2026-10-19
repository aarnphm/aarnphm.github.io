import type { BasesPropertyId, BasesViewConfig } from 'obsidian'
import {
  type TopicShelf,
  hasTopicTag,
  topicKeys,
  topicLabel,
  topicShelf,
  topicShortcuts,
} from './scene-model'

export const ALL = 'all'
export const OTHER = 'other'

export interface Topic {
  id: string
  label: string
  count: number
}

export interface TopicTab extends Topic {
  /** Shown on the collapsed tab; its first letter jumps to it. */
  shortcut: string
}

export interface TopicConfig {
  /** Property that groups a topic's books into shelves. */
  shelvesBy: BasesPropertyId
  /** Raw `topicShelves` map from the view; the shared helpers validate it. */
  shelves: unknown
}

/** A property as the .base file writes it (`category`, `note.category`, `formula.x`) as an id. */
export function propertyId(raw: unknown): BasesPropertyId | null {
  const name = typeof raw === 'string' ? raw.trim() : ''
  if (!name) return null
  return (/^(note|file|formula)\./.test(name) ? name : `note.${name}`) as BasesPropertyId
}

export function readTopicConfig(config: BasesViewConfig): TopicConfig {
  return {
    shelvesBy: propertyId(config.get('topicGroupBy')) ?? ('note.subcategory' as BasesPropertyId),
    shelves: config.get('topicShelves'),
  }
}

/**
 * Label and shelf layout of a topic, by the garden's shared rule. What the shelf leaves out, the
 * view's own style, material and labels supply; "all" and "other" have no shelf.
 */
export function lookupTopic(
  topic: string,
  config: TopicConfig,
): { label: string; shelf?: TopicShelf } {
  if (topic === ALL || topic === OTHER) return { label: topic }
  return { label: topicLabel(topic, config.shelves), shelf: topicShelf(topic, config.shelves) }
}

/**
 * Books of each topic, by file path, as the garden counts them. A topic is a group (`groups`
 * gives each book's group value, null when it has none) or a `topicShelves` key that the book
 * carries as a tag. Each book counts once per topic. Topics of one book share "other".
 */
export function topicMembers(
  groups: Iterable<[string | null, string]>,
  tags: (path: string) => string[],
  config: TopicConfig,
): Map<string, Set<string>> {
  const declared = topicKeys(config.shelves)
  const members = new Map<string, Set<string>>()
  const join = (topic: string, path: string) => {
    const set = members.get(topic)
    if (set) set.add(path)
    else members.set(topic, new Set([path]))
  }
  for (const [topic, path] of groups) {
    if (topic !== null) join(topic, path)
    for (const key of declared) if (key !== topic && hasTopicTag(tags(path), key)) join(key, path)
  }
  const small = [...members].filter(([, set]) => set.size <= 1)
  for (const [topic, set] of small) {
    members.delete(topic)
    for (const path of set) join(OTHER, path)
  }
  return members
}

/** Tabs: "all", then topics by book count, most first, then "other". Empty with one topic or none. */
export function topicTabs(
  members: Map<string, Set<string>>,
  total: number,
  config: TopicConfig,
): TopicTab[] {
  if (members.size <= 1) return []
  const tabs: Topic[] = [
    { id: ALL, label: ALL, count: total },
    ...[...members]
      .filter(([topic]) => topic !== OTHER)
      .sort(([, a], [, b]) => b.size - a.size)
      .map(([id, set]) => ({ id, label: lookupTopic(id, config).label, count: set.size })),
  ]
  const other = members.get(OTHER)
  if (other) tabs.push({ id: OTHER, label: OTHER, count: other.size })
  const shortcuts = topicShortcuts(tabs.map(tab => tab.label))
  shortcuts[0] = 'ALL'
  return tabs.map((tab, i) => ({ ...tab, shortcut: shortcuts[i] }))
}
