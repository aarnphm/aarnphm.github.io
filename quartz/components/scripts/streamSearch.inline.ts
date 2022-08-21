import FlexSearch from "flexsearch"
import { tokenizeTerm, encode } from "./util"

interface StreamEntry {
  id: string
  html: string
  metadata: unknown
  isoDate: string | null
  displayDate: string | null
}

interface StreamGroup {
  groupId: string
  timestamp: number | null
  isoDate: string | null
  groupSize: number
  path: string | null
  entries: StreamEntry[]
}

interface IndexedEntry {
  id: number
  entryId: string
  groupId: string
  content: string
  metadata: string
  isoDate: string
  displayDate: string
  tags: string[]
}

function extractMetadata(raw: unknown): { tags: string[]; metadataString: string } {
  let metadataObj: Record<string, unknown> = {}
  let metadataString = "{}"

  if (typeof raw === "string") {
    metadataString = raw
    try {
      const parsed = JSON.parse(raw)
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        metadataObj = parsed as Record<string, unknown>
      }
    } catch {
      metadataObj = {}
    }
  } else if (raw && typeof raw === "object" && !Array.isArray(raw)) {
    metadataObj = raw as Record<string, unknown>
    try {
      metadataString = JSON.stringify(metadataObj)
    } catch {
      metadataString = "{}"
    }
  }

  const rawTags = Array.isArray(metadataObj.tags) ? metadataObj.tags : []
  const tags = rawTags.map((tag) => String(tag).trim()).filter((tag) => tag.length > 0)

  return { tags, metadataString }
}

let searchIndex: any | null = null
let indexedEntries: IndexedEntry[] = []
let isIndexBuilt = false
let searchTimeout: number | null = null

/**
 * Strip HTML tags from content for indexing
 */
function stripHtml(html: string): string {
  const tmp = document.createElement("div")
  tmp.innerHTML = html
  return tmp.textContent || tmp.innerText || ""
}

/**
 * Build search index from stream data
 */
function isStreamHost(): boolean {
  return typeof window !== "undefined" && window.location.hostname === "stream.aarnphm.xyz"
}

async function buildSearchIndex() {
  if (isIndexBuilt) return

  try {
    const endpoint = isStreamHost() ? `${window.location.origin}/streams.jsonl` : "/streams.jsonl"
    const response = await fetch(endpoint)
    if (!response.ok) {
      console.error("[StreamSearch] Failed to load stream data:", response.statusText)
      return
    }

    const text = await response.text()
    const lines = text.trim().split("\n")

    let entryIndex = 0
    for (const line of lines) {
      if (!line.trim()) continue

      try {
        const group: StreamGroup = JSON.parse(line)

        for (const entry of group.entries) {
          const { tags, metadataString } = extractMetadata(entry.metadata)

          const indexedEntry: IndexedEntry = {
            id: entryIndex++,
            entryId: entry.id,
            groupId: group.groupId,
            content: stripHtml(entry.html),
            metadata: metadataString,
            isoDate: entry.isoDate || group.isoDate || "",
            displayDate: entry.displayDate || group.isoDate || "",
            tags,
          }
          indexedEntries.push(indexedEntry)
        }
      } catch (err) {
        console.warn("[StreamSearch] Failed to parse line:", err)
      }
    }

    // Build FlexSearch index
    searchIndex = new FlexSearch.Document({
      tokenize: "forward",
      encode,
      document: {
        id: "id",
        index: ["content", "metadata", "isoDate", "displayDate", "tags"],
      },
    })

    // Add all entries to index
    for (const entry of indexedEntries) {
      const tagsField = entry.tags
        .flatMap((tag) => [tag, `#${tag}`])
        .join(" ")
        .trim()

      await searchIndex.addAsync({
        ...entry,
        tags: tagsField,
      })
    }

    isIndexBuilt = true
  } catch (err) {
    console.error("[StreamSearch] Failed to build search index:", err)
  }
}

/**
 * Highlight matches in text nodes recursively
 */
function highlightTextNodes(element: HTMLElement, searchTerm: string) {
  const tokens = tokenizeTerm(searchTerm)
  const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT, null)

  const nodesToReplace: { node: Text; parent: Node }[] = []
  let currentNode: Node | null

  while ((currentNode = walker.nextNode())) {
    const textNode = currentNode as Text
    const text = textNode.nodeValue || ""

    // Check if any token matches
    let hasMatch = false
    for (const token of tokens) {
      if (text.toLowerCase().includes(token.toLowerCase())) {
        hasMatch = true
        break
      }
    }

    if (hasMatch && textNode.parentNode) {
      nodesToReplace.push({ node: textNode, parent: textNode.parentNode })
    }
  }

  // Replace text nodes with highlighted versions
  for (const { node, parent } of nodesToReplace) {
    const text = node.nodeValue || ""
    const fragment = document.createDocumentFragment()
    let lastIndex = 0
    let modified = false

    for (const token of tokens) {
      const regex = new RegExp(token, "gi")
      let match: RegExpExecArray | null

      while ((match = regex.exec(text))) {
        modified = true
        // Add text before match
        if (match.index > lastIndex) {
          fragment.appendChild(document.createTextNode(text.slice(lastIndex, match.index)))
        }

        // Add highlighted match
        const mark = document.createElement("mark")
        mark.className = "search-highlight"
        mark.textContent = match[0]
        fragment.appendChild(mark)

        lastIndex = match.index + match[0].length
      }
    }

    if (modified) {
      // Add remaining text
      if (lastIndex < text.length) {
        fragment.appendChild(document.createTextNode(text.slice(lastIndex)))
      }
      parent.replaceChild(fragment, node)
    }
  }
}

/**
 * Clear all highlights
 */
function clearHighlights() {
  const highlights = document.querySelectorAll(".stream-entry .search-highlight")
  highlights.forEach((mark) => {
    const text = mark.textContent || ""
    const textNode = document.createTextNode(text)
    mark.parentNode?.replaceChild(textNode, mark)
  })
}

function parseTagTokens(query: string): string[] {
  return Array.from(
    new Set(
      query
        .trim()
        .split(/\s+/)
        .map((token) => (token.startsWith("#") ? token.slice(1) : ""))
        .filter((token) => token.length > 0),
    ),
  )
}

/**
 * Filter stream entries based on search query
 */
async function filterStreamEntries(query: string) {
  if (!searchIndex || !isIndexBuilt) {
    await buildSearchIndex()
  }

  if (!searchIndex) {
    console.error("[StreamSearch] Search index not available")
    return
  }

  const trimmedQuery = query.trim()
  const streamEntries = document.querySelectorAll<HTMLElement>(".stream-entry")

  // Clear existing highlights
  clearHighlights()

  // If no query, show all entries
  if (!trimmedQuery) {
    streamEntries.forEach((entry) => {
      entry.style.display = ""
    })
    updateSearchStatus("")
    return
  }

  const lowerQuery = trimmedQuery.toLowerCase()
  const isTagQuery = lowerQuery.startsWith("#")
  const tagTokens = isTagQuery ? parseTagTokens(lowerQuery) : []
  const matchedEntryIds = new Set<string>()
  let highlightTerm = trimmedQuery

  if (isTagQuery) {
    if (tagTokens.length === 0) {
      streamEntries.forEach((entry) => {
        entry.style.display = ""
      })
      updateSearchStatus("type a tag name after '#'")
      return
    }

    for (const entry of indexedEntries) {
      const normalizedTags = entry.tags.map((tag) => tag.toLowerCase())
      const matchesAll = tagTokens.every((token) =>
        normalizedTags.some((entryTag) => entryTag.startsWith(token)),
      )
      if (matchesAll) {
        matchedEntryIds.add(entry.entryId)
      }
    }

    highlightTerm = tagTokens.join(" ")
  } else {
    try {
      // Search across all indexed fields
      const results = await searchIndex.searchAsync({
        query: trimmedQuery,
        limit: 500,
        index: ["content", "metadata", "isoDate", "displayDate", "tags"],
      })

      for (const fieldResult of Object.values(results)) {
        if (fieldResult && (fieldResult as any).result) {
          for (const id of (fieldResult as any).result) {
            const entry = indexedEntries[Number(id)]
            if (entry) {
              matchedEntryIds.add(entry.entryId)
            }
          }
        }
      }
    } catch (err) {
      console.error("[StreamSearch] Search failed:", err)
      updateSearchStatus("search error")
      return
    }
  }

  try {
    // Show/hide entries based on matches
    let visibleCount = 0
    streamEntries.forEach((entry) => {
      const entryId = (entry as HTMLElement).dataset.entryId
      if (entryId && matchedEntryIds.has(entryId)) {
        entry.style.display = ""

        // Highlight matches in content
        const contentEl = entry.querySelector(".stream-entry-content") as HTMLElement
        if (contentEl && highlightTerm) {
          highlightTextNodes(contentEl, highlightTerm)
        }

        if (isTagQuery && highlightTerm) {
          const tagElements = entry.querySelectorAll(".stream-entry-tag")
          tagElements.forEach((tagEl) => {
            highlightTextNodes(tagEl as HTMLElement, highlightTerm)
          })
        }
        visibleCount++
      } else {
        entry.style.display = "none"
      }
    })

    if (isTagQuery) {
      const readableTags = tagTokens.map((tag) => `#${tag}`).join(" ")
      updateSearchStatus(
        visibleCount > 0
          ? `showing ${visibleCount} ${visibleCount === 1 ? "entry" : "entries"} tagged ${readableTags}`
          : `no entries tagged ${readableTags}`,
      )
    } else {
      updateSearchStatus(
        visibleCount > 0
          ? `showing ${visibleCount} ${visibleCount === 1 ? "entry" : "entries"}`
          : `no results for "${trimmedQuery}"`,
      )
    }
  } catch (err) {
    console.error("[StreamSearch] Failed to update stream entries:", err)
    updateSearchStatus("search error")
  }
}

/**
 * Update search status message
 */
function updateSearchStatus(message: string) {
  let statusEl = document.querySelector(".stream-search-status") as HTMLElement
  if (!statusEl && message) {
    statusEl = document.createElement("div")
    statusEl.className = "stream-search-status"
    const form = document.querySelector(".stream-search-form")
    if (form) {
      form.after(statusEl)
    }
  }

  if (statusEl) {
    if (message) {
      statusEl.textContent = message
      statusEl.style.display = "block"
    } else {
      statusEl.style.display = "none"
    }
  }
}

/**
 * Initialize stream search
 */
async function initStreamSearch() {
  const currentPath = window.location.pathname
  const isStreamPage =
    currentPath === "/stream" || currentPath.startsWith("/stream/") || isStreamHost()
  if (!isStreamPage) return

  // Pre-build index for instant search
  await buildSearchIndex()

  const form = document.querySelector(".stream-search-form") as HTMLFormElement
  const searchInput = document.querySelector(".stream-search-input") as HTMLInputElement

  if (!form || !searchInput) return

  const focusShortcutHandler = (event: KeyboardEvent) => {
    const key = event.key.toLowerCase()
    const isMetaDot = event.metaKey && key === "."
    const isCommandK = (event.metaKey || event.ctrlKey) && key === "k"
    if (!isMetaDot && !isCommandK) return

    const target = event.target as Element | null
    if (target) {
      const tag = target.tagName?.toLowerCase()
      if (tag === "input" || tag === "textarea" || (target as HTMLElement).isContentEditable) {
        return
      }
    }

    event.preventDefault()
    event.stopPropagation()
    event.stopImmediatePropagation()

    searchInput.focus()
    searchInput.select()
  }

  const focusListenerOptions: AddEventListenerOptions = { capture: true }
  document.addEventListener("keydown", focusShortcutHandler, focusListenerOptions)

  // Handle input changes with debounce
  const handleInput = () => {
    if (searchTimeout !== null) {
      window.clearTimeout(searchTimeout)
    }

    searchTimeout = window.setTimeout(async () => {
      const query = searchInput.value
      await filterStreamEntries(query)
    }, 300)
  }

  // Prevent form submission
  const handleSubmit = (e: Event) => {
    e.preventDefault()
  }

  searchInput.addEventListener("input", handleInput)
  form.addEventListener("submit", handleSubmit)

  window.addCleanup(() => {
    searchInput.removeEventListener("input", handleInput)
    form.removeEventListener("submit", handleSubmit)
    document.removeEventListener("keydown", focusShortcutHandler, focusListenerOptions)
    if (searchTimeout !== null) {
      window.clearTimeout(searchTimeout)
      searchTimeout = null
    }
  })
}

// Initialize on page load and navigation
document.addEventListener("nav", async () => {
  await initStreamSearch()
})
