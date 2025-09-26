import FlexSearch, { DefaultDocumentSearchResults, DocumentData, Id } from "flexsearch"
import { SemanticClient } from "./semantic.inline"
import type { ContentDetails } from "../../plugins"
import {
  registerEscapeHandler,
  removeAllChildren,
  highlight,
  tokenizeTerm,
  encode,
  fetchCanonical,
} from "./util"
import { FullSlug, normalizeRelativeURLs, resolveRelative } from "../../util/path"
import { escapeHTML } from "../../util/escape"

interface Item extends DocumentData {
  id: number
  slug: FullSlug
  title: string
  content: string
  tags: string[]
  aliases: string[]
  target: string
  [key: string]: any
}

// Can be expanded with things like "term" in the future
type SearchType = "basic" | "tags"
type SearchMode = "lexical" | "semantic"
let searchMode: SearchMode = "semantic"
let currentSearchTerm: string = ""
let rawSearchTerm: string = ""
let semantic: SemanticClient | null = null
let semanticReady = false
let semanticInitFailed = false
type SimilarityResult = { item: Item; similarity: number }

// Initialize the FlexSearch Document instance with the appropriate configuration
const index = new FlexSearch.Document<Item>({
  tokenize: "forward",
  encode,
  document: {
    id: "id",
    index: [
      {
        field: "title",
        tokenize: "forward",
      },
      {
        field: "content",
        tokenize: "forward",
      },
      {
        field: "tags",
        tokenize: "forward",
      },
      {
        field: "aliases",
        tokenize: "forward",
      },
    ],
  },
})

const p = new DOMParser()
const fetchContentCache: Map<FullSlug, Element[]> = new Map()
const numSearchResults = 10
const numTagResults = 10
function highlightHTML(searchTerm: string, el: HTMLElement) {
  const p = new DOMParser()
  const tokenizedTerms = tokenizeTerm(searchTerm)
  const html = p.parseFromString(el.innerHTML, "text/html")

  const createHighlightSpan = (text: string) => {
    const span = document.createElement("span")
    span.className = "highlight"
    span.textContent = text
    return span
  }

  const highlightTextNodes = (node: Node, term: string) => {
    if (node.nodeType === Node.TEXT_NODE) {
      const nodeText = node.nodeValue ?? ""
      const regex = new RegExp(term.toLowerCase(), "gi")
      const matches = nodeText.match(regex)
      if (!matches || matches.length === 0) return
      const spanContainer = document.createElement("span")
      let lastIndex = 0
      for (const match of matches) {
        const matchIndex = nodeText.indexOf(match, lastIndex)
        spanContainer.appendChild(document.createTextNode(nodeText.slice(lastIndex, matchIndex)))
        spanContainer.appendChild(createHighlightSpan(match))
        lastIndex = matchIndex + match.length
      }
      spanContainer.appendChild(document.createTextNode(nodeText.slice(lastIndex)))
      node.parentNode?.replaceChild(spanContainer, node)
    } else if (node.nodeType === Node.ELEMENT_NODE) {
      if ((node as HTMLElement).classList.contains("highlight")) return
      Array.from(node.childNodes).forEach((child) => highlightTextNodes(child, term))
    }
  }

  for (const term of tokenizedTerms) {
    highlightTextNodes(html.body, term)
  }

  return html.body
}

async function setupSearch(
  searchElement: HTMLDivElement,
  currentSlug: FullSlug,
  data: ContentIndex,
) {
  const container = searchElement.querySelector(".search-container") as HTMLElement
  if (!container) return

  const searchButton = searchElement.querySelector(".search-button") as HTMLButtonElement
  if (!searchButton) return

  const searchBar = searchElement.querySelector(".search-bar") as HTMLInputElement
  if (!searchBar) return

  const searchLayout = searchElement?.querySelector(".search-layout") as HTMLOutputElement
  if (!searchLayout) return

  const searchSpace = searchElement?.querySelector(".search-space") as HTMLFormElement
  if (!searchSpace) return

  const idDataMap = Object.keys(data) as FullSlug[]
  const slugToIndex = new Map<FullSlug, number>()
  idDataMap.forEach((slug, idx) => slugToIndex.set(slug, idx))
  const el = searchSpace?.querySelector("ul#helper")
  const modeToggle = searchSpace.querySelector(".search-mode-toggle") as HTMLDivElement | null
  const modeButtons = modeToggle
    ? Array.from(modeToggle.querySelectorAll<HTMLButtonElement>(".mode-option"))
    : []

  const appendLayout = (el: HTMLElement) => {
    searchLayout.appendChild(el)
  }

  if (!el) {
    const keys = [
      { kbd: "↑↓", description: "pour naviguer" },
      { kbd: "↵", description: "pour ouvrir" },
      { kbd: "esc", description: "pour rejeter" },
    ]
    const helper = document.createElement("ul")
    helper.id = "helper"
    for (const { kbd, description } of keys) {
      const liEl = document.createElement("li")
      liEl.innerHTML = `<kbd>${escapeHTML(kbd)}</kbd>${description}`
      helper.appendChild(liEl)
    }
    searchSpace.appendChild(helper)
  }

  const enablePreview = searchLayout.dataset.preview === "true"
  if (!semantic && !semanticInitFailed) {
    const client = new SemanticClient(semanticCfg)
    try {
      await client.ensureReady()
      semantic = client
      semanticReady = true
    } catch (err) {
      console.warn("[SemanticClient] initialization failed:", err)
      client.dispose()
      semantic = null
      semanticReady = false
      semanticInitFailed = true
    }
  } else if (semantic && !semanticReady) {
    try {
      await semantic.ensureReady()
      semanticReady = true
    } catch (err) {
      console.warn("[SemanticClient] became unavailable:", err)
      semantic.dispose()
      semantic = null
      semanticReady = false
      semanticInitFailed = true
    }
  }
  if (!semanticReady && searchMode === "semantic") {
    searchMode = "lexical"
  }
  let searchSeq = 0
  let runSearchTimer: number | null = null
  searchLayout.dataset.mode = searchMode

  const updateModeUI = (mode: SearchMode) => {
    modeButtons.forEach((button) => {
      const btnMode = (button.dataset.mode as SearchMode) ?? "lexical"
      const isActive = btnMode === mode
      button.classList.toggle("active", isActive)
      button.setAttribute("aria-pressed", String(isActive))
    })
    if (modeToggle) {
      modeToggle.dataset.mode = mode
    }
    searchLayout.dataset.mode = mode
  }

  const triggerSearchWithMode = (mode: SearchMode) => {
    if (mode === "semantic" && !semanticReady) {
      return
    }
    if (searchMode === mode) return
    searchMode = mode
    updateModeUI(mode)
    if (rawSearchTerm.trim() !== "") {
      searchLayout.classList.add("display-results")
      const token = ++searchSeq
      void runSearch(rawSearchTerm, token)
    }
  }

  updateModeUI(searchMode)

  modeButtons.forEach((button) => {
    const btnMode = (button.dataset.mode as SearchMode) ?? "lexical"
    if (btnMode === "semantic") {
      button.disabled = !semanticReady
      button.setAttribute("aria-disabled", String(!semanticReady))
    }
    const handler = () => triggerSearchWithMode(btnMode)
    button.addEventListener("click", handler)
    window.addCleanup(() => button.removeEventListener("click", handler))
  })
  let preview: HTMLDivElement | undefined = undefined
  let previewInner: HTMLDivElement | undefined = undefined
  const results = document.createElement("div")
  results.className = "results-container"
  appendLayout(results)

  if (enablePreview) {
    preview = document.createElement("div")
    preview.className = "preview-container"
    appendLayout(preview)
  }

  function hideSearch() {
    container.classList.remove("active")
    searchBar.value = "" // clear the input when we dismiss the search
    rawSearchTerm = ""
    removeAllChildren(results)
    if (preview) {
      removeAllChildren(preview)
    }
    searchLayout.classList.remove("display-results")
    searchButton.focus()
  }

  function showSearch(type: SearchType) {
    container.classList.add("active")
    if (type === "tags") {
      searchBar.value = "#"
      rawSearchTerm = "#"
    }
    searchBar.focus()
  }

  let currentHover: HTMLInputElement | null = null

  async function shortcutHandler(e: HTMLElementEventMap["keydown"]) {
    const paletteOpen = document.querySelector("search#palette-container") as HTMLDivElement
    if (paletteOpen && paletteOpen.classList.contains("active")) return

    if ((e.key === "/" || e.key === "k") && (e.ctrlKey || e.metaKey) && !e.shiftKey) {
      e.preventDefault()
      const searchBarOpen = container.classList.contains("active")
      searchBarOpen ? hideSearch() : showSearch("basic")
      return
    } else if (e.shiftKey && (e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
      // Hotkey to open tag search
      e.preventDefault()
      const searchBarOpen = container.classList.contains("active")
      searchBarOpen ? hideSearch() : showSearch("tags")
      return
    }

    if (currentHover) {
      currentHover.classList.remove("focus")
    }

    // If search is active, then we will render the first result and display accordingly
    if (!container.classList.contains("active")) return
    if (e.key === "Enter") {
      // If result has focus, navigate to that one, otherwise pick first result
      let anchor: HTMLAnchorElement | undefined
      if (results.contains(document.activeElement)) {
        anchor = document.activeElement as HTMLAnchorElement
        if (anchor.classList.contains("no-match")) return
        await displayPreview(anchor)
        e.preventDefault()
        anchor.click()
      } else {
        anchor = document.getElementsByClassName("result-card")[0] as HTMLAnchorElement
        if (!anchor || anchor.classList.contains("no-match")) return
        await displayPreview(anchor)
        e.preventDefault()
        anchor.click()
      }
      if (anchor !== undefined)
        window.spaNavigate(new URL(new URL(anchor.href).pathname, window.location.toString()))
    } else if (
      e.key === "ArrowUp" ||
      (e.shiftKey && e.key === "Tab") ||
      (e.ctrlKey && e.key === "p")
    ) {
      e.preventDefault()
      if (results.contains(document.activeElement)) {
        // If an element in results-container already has focus, focus previous one
        const currentResult = currentHover
          ? currentHover
          : (document.activeElement as HTMLInputElement | null)
        const prevResult = currentResult?.previousElementSibling as HTMLInputElement | null
        currentResult?.classList.remove("focus")
        prevResult?.focus()
        if (prevResult) currentHover = prevResult
        await displayPreview(prevResult)
      }
    } else if (e.key === "ArrowDown" || e.key === "Tab" || (e.ctrlKey && e.key === "n")) {
      e.preventDefault()
      // The results should already been focused, so we need to find the next one.
      // The activeElement is the search bar, so we need to find the first result and focus it.
      if (document.activeElement === searchBar || currentHover !== null) {
        const firstResult = currentHover
          ? currentHover
          : (document.getElementsByClassName("result-card")[0] as HTMLInputElement | null)
        const secondResult = firstResult?.nextElementSibling as HTMLInputElement | null
        firstResult?.classList.remove("focus")
        secondResult?.focus()
        if (secondResult) currentHover = secondResult
        await displayPreview(secondResult)
      }
    }
  }

  const formatForDisplay = (term: string, id: number, renderType: SearchType) => {
    const slug = idDataMap[id]
    if (data[slug].layout === "letter") {
      return null
    }
    const aliases: string[] = data[slug].aliases
    const target = aliases.find((alias) => alias.toLowerCase().includes(term.toLowerCase())) ?? ""

    return {
      id,
      slug,
      title:
        renderType === "tags" || target
          ? data[slug].title
          : highlight(term, data[slug].title ?? ""),
      target,
      content: highlight(term, data[slug].content ?? "", true),
      tags: highlightTags(term, data[slug].tags, renderType),
      aliases: aliases,
    }
  }

  function highlightTags(term: string, tags: string[], renderType: SearchType) {
    if (!tags || renderType !== "tags") {
      return []
    }

    const tagTerm = term.toLowerCase()
    return tags
      .map((tag) => {
        if (tag.toLowerCase().includes(tagTerm)) {
          return `<li><p class="match-tag">#${tag}</p></li>`
        } else {
          return `<li><p>#${tag}</p></li>`
        }
      })
      .slice(0, numTagResults)
  }

  function resolveUrl(slug: FullSlug): URL {
    return new URL(resolveRelative(currentSlug, slug), location.toString())
  }

  const resultToHTML = ({ item, percent }: { item: Item; percent: number | null }) => {
    const { slug, title, content, tags, target } = item
    const htmlTags = tags.length > 0 ? `<ul class="tags">${tags.join("")}</ul>` : ``
    const itemTile = document.createElement("a")
    const titleContent = target ? highlight(currentSearchTerm, target) : title
    const subscript = target ? `<b>${slug}</b>` : ``
    let percentLabel = "—"
    let percentAttr = ""
    if (percent !== null && Number.isFinite(percent)) {
      const bounded = Math.max(0, Math.min(100, percent))
      percentLabel = `${bounded.toFixed(1)}%`
      percentAttr = bounded.toFixed(3)
    }
    itemTile.classList.add("result-card")
    itemTile.id = slug
    itemTile.href = resolveUrl(slug).toString()
    itemTile.innerHTML = `<hgroup>
      <h3>${titleContent}</h3>
      ${subscript}${htmlTags}
      ${searchMode === "semantic" ? `<span class="result-likelihood" title="match likelihood">&nbsp;${percentLabel}</span>` : ""}
      ${enablePreview && window.innerWidth > 600 ? "" : `<p>${content}</p>`}
    </hgroup>`
    if (percentAttr) itemTile.dataset.scorePercent = percentAttr
    else delete itemTile.dataset.scorePercent

    const handler = (evt: MouseEvent) => {
      if (evt.altKey || evt.ctrlKey || evt.metaKey || evt.shiftKey) return
      window.spaNavigate(new URL((evt.target as HTMLAnchorElement).href))
      hideSearch()
    }

    async function onMouseEnter(ev: MouseEvent) {
      if (!ev.target) return
      const target = ev.target as HTMLInputElement
      await displayPreview(target)
    }

    itemTile.addEventListener("mouseenter", onMouseEnter)
    window.addCleanup(() => itemTile.removeEventListener("mouseenter", onMouseEnter))
    itemTile.addEventListener("click", handler)
    window.addCleanup(() => itemTile.removeEventListener("click", handler))

    return itemTile
  }

  async function displayResults(finalResults: SimilarityResult[]) {
    removeAllChildren(results)
    if (finalResults.length === 0) {
      results.innerHTML = `<a class="result-card no-match">
          <h3>No results.</h3>
          <p>Try another search term?</p>
      </a>`
      currentHover = null
    } else {
      const decorated = finalResults.map(({ item, similarity }) => {
        if (!Number.isFinite(similarity)) return { item, percent: null }
        const bounded = Math.max(-1, Math.min(1, similarity))
        const percent = ((bounded + 1) / 2) * 100
        return { item, percent }
      })
      results.append(...decorated.map(resultToHTML))
    }

    if (finalResults.length === 0 && preview) {
      // no results, clear previous preview
      removeAllChildren(preview)
    } else {
      // focus on first result, then also dispatch preview immediately
      const firstChild = results.firstElementChild as HTMLElement
      firstChild.classList.add("focus")
      currentHover = firstChild as HTMLInputElement
      await displayPreview(firstChild)
    }
  }

  async function fetchContent(slug: FullSlug): Promise<Element[]> {
    if (fetchContentCache.has(slug)) {
      return fetchContentCache.get(slug) as Element[]
    }

    const targetUrl = resolveUrl(slug)
    const contents = await fetchCanonical(targetUrl)
      .then((res) => res.text())
      .then((contents) => {
        if (contents === undefined) {
          throw new Error(`Could not fetch ${targetUrl}`)
        }
        const html = p.parseFromString(contents ?? "", "text/html")
        normalizeRelativeURLs(html, targetUrl)
        return [...html.getElementsByClassName("popover-hint")]
      })

    fetchContentCache.set(slug, contents)
    return contents
  }

  async function displayPreview(el: HTMLElement | null) {
    if (!searchLayout || !enablePreview || !el || !preview) return
    const slug = el.id as FullSlug
    const innerDiv = await fetchContent(slug).then((contents) =>
      contents.flatMap((el) => [...highlightHTML(currentSearchTerm, el as HTMLElement).children]),
    )
    previewInner = document.createElement("div")
    previewInner.classList.add("preview-inner")
    previewInner.append(...innerDiv)
    preview.replaceChildren(previewInner)

    // scroll to longest
    const highlights = [...preview.getElementsByClassName("highlight")].sort(
      (a, b) => b.innerHTML.length - a.innerHTML.length,
    )
    if (highlights.length > 0) {
      const highlight = highlights[0]
      const container = preview
      if (container && highlight) {
        // Get the relative positions
        const containerRect = container.getBoundingClientRect()
        const highlightRect = highlight.getBoundingClientRect()
        // Calculate the scroll position relative to the container
        const relativeTop = highlightRect.top - containerRect.top + container.scrollTop - 20 // 20px buffer
        // Smoothly scroll the container
        container.scrollTo({
          top: relativeTop,
          behavior: "smooth",
        })
      }
    }
  }

  async function runSearch(rawTerm: string, token: number) {
    if (!searchLayout || !index) return
    const trimmed = rawTerm.trim()
    if (trimmed === "") {
      removeAllChildren(results)
      if (preview) {
        removeAllChildren(preview)
      }
      currentHover = null
      searchLayout.classList.remove("display-results")
      return
    }

    const modeForRanking: SearchMode = searchMode
    const initialType: SearchType = trimmed.startsWith("#") ? "tags" : "basic"
    let workingType: SearchType = initialType
    let highlightTerm = trimmed
    let tagTerm = ""
    let searchResults: DefaultDocumentSearchResults<Item> = []

    if (initialType === "tags") {
      tagTerm = trimmed.substring(1).trim()
      const separatorIndex = tagTerm.indexOf(" ")
      if (separatorIndex !== -1) {
        const tag = tagTerm.substring(0, separatorIndex).trim()
        const query = tagTerm.substring(separatorIndex + 1).trim()
        const results = await index.searchAsync({
          query,
          limit: Math.max(numSearchResults, 10000),
          index: ["title", "content", "aliases"],
          tag: { tags: tag },
        })
        if (token !== searchSeq) return
        searchResults = Object.values(results)
        workingType = "basic"
        highlightTerm = query
      } else {
        const results = await index.searchAsync({
          query: tagTerm,
          limit: numSearchResults,
          index: ["tags"],
        })
        if (token !== searchSeq) return
        searchResults = Object.values(results)
        highlightTerm = tagTerm
      }
    } else {
      const results = await index.searchAsync({
        query: highlightTerm,
        limit: numSearchResults,
        index: ["title", "content", "aliases"],
      })
      if (token !== searchSeq) return
      searchResults = Object.values(results)
    }

    const coerceIds = (hit?: DefaultDocumentSearchResults<Item>[number]): number[] => {
      if (!hit) return []
      return hit.result
        .map((value: Id) => {
          if (typeof value === "number") {
            return value
          }
          const parsed = Number.parseInt(String(value), 10)
          return Number.isNaN(parsed) ? null : parsed
        })
        .filter((value): value is number => value !== null)
    }

    const getByField = (field: string): number[] => {
      const hit = searchResults.find((x) => x.field === field)
      return coerceIds(hit)
    }

    const allIds: Set<number> = new Set([
      ...getByField("aliases"),
      ...getByField("title"),
      ...getByField("content"),
      ...getByField("tags"),
    ])

    currentSearchTerm = highlightTerm

    const candidateItems = new Map<string, Item>()
    const ensureItem = (id: number): Item | null => {
      const slug = idDataMap[id]
      if (!slug) return null
      const cached = candidateItems.get(slug)
      if (cached) return cached
      const item = formatForDisplay(highlightTerm, id, workingType)
      if (item) {
        candidateItems.set(slug, item)
        return item
      }
      return null
    }

    const baseIndices: number[] = []
    for (const id of allIds) {
      const item = ensureItem(id)
      if (!item) continue
      const idx = slugToIndex.get(item.slug)
      if (typeof idx === "number") {
        baseIndices.push(idx)
      }
    }

    let semanticIds: number[] = []
    let bmIds: number[] = []
    const semanticSimilarity = new Map<number, number>()
    const bmSimilarity = new Map<number, number>()

    const integrateIds = (ids: number[]) => {
      ids.forEach((docId) => {
        ensureItem(docId)
      })
    }

    const orchestrator = semanticReady && semantic ? semantic : null

    const resolveSimilarity = (item: Item): number => {
      const semanticHit = semanticSimilarity.get(item.id)
      if (semanticHit !== undefined) return semanticHit
      const lexicalHit = bmSimilarity.get(item.id)
      return lexicalHit ?? Number.NaN
    }

    const render = async () => {
      if (token !== searchSeq) return
      const useSemantic = semanticReady && semanticIds.length > 0
      const weights =
        modeForRanking === "semantic" && useSemantic
          ? { base: 0.3, semantic: 1, bm: 0.45 }
          : { base: 1, semantic: useSemantic ? 0.15 : 0, bm: 0.9 }
      const rrf = new Map<string, number>()
      const push = (ids: number[], weight: number) => {
        if (!ids.length || weight <= 0) return
        ids.forEach((docId, rank) => {
          const slug = idDataMap[docId]
          if (!slug) return
          const item = ensureItem(docId)
          if (!item) return
          const prev = rrf.get(slug) ?? 0
          rrf.set(slug, prev + weight / (1 + rank))
        })
      }

      push(baseIndices, weights.base)
      push(semanticIds, weights.semantic)
      push(bmIds, weights.bm)

      const rankedEntries = Array.from(candidateItems.values())
        .map((item) => ({ item, score: rrf.get(item.slug) ?? 0 }))
        .sort((a, b) => b.score - a.score)
        .slice(0, numSearchResults)

      const displayEntries: SimilarityResult[] = []
      for (const entry of rankedEntries) {
        const similarity = resolveSimilarity(entry.item)
        displayEntries.push({ item: entry.item, similarity })
      }

      await displayResults(displayEntries)
    }

    await render()

    if (workingType === "tags" || !orchestrator || !semanticReady || highlightTerm.length < 2) {
      return
    }

    try {
      const { semantic: semRes, bm25: bmRes } = await orchestrator.search(
        highlightTerm,
        numSearchResults,
      )
      if (token !== searchSeq) return
      semanticIds = semRes.map((x) => x.id)
      semanticSimilarity.clear()
      semRes.forEach(({ id, score }) => semanticSimilarity.set(id, score))
      bmIds = bmRes.map((x) => x.id)
      bmSimilarity.clear()
      bmRes.forEach(({ id, score }) => bmSimilarity.set(id, score))
      integrateIds(semanticIds)
      integrateIds(bmIds)
    } catch (err) {
      console.warn("[SemanticClient] search failed:", err)
      orchestrator.dispose()
      semantic = null
      semanticReady = false
      semanticInitFailed = true
      if (searchMode === "semantic") {
        searchMode = "lexical"
        updateModeUI(searchMode)
      }
      modeButtons.forEach((button) => {
        if ((button.dataset.mode as SearchMode) === "semantic") {
          button.disabled = true
          button.setAttribute("aria-disabled", "true")
        }
      })
    }

    await render()
  }

  function onType(e: HTMLElementEventMap["input"]) {
    if (!searchLayout || !index) return
    rawSearchTerm = (e.target as HTMLInputElement).value
    const hasQuery = rawSearchTerm.trim() !== ""
    searchLayout.classList.toggle("display-results", hasQuery)
    const term = rawSearchTerm
    const token = ++searchSeq
    if (runSearchTimer !== null) {
      window.clearTimeout(runSearchTimer)
      runSearchTimer = null
    }
    if (!hasQuery) {
      void runSearch("", token)
      return
    }
    const delay = searchMode === "semantic" ? 160 : 60
    runSearchTimer = window.setTimeout(() => {
      runSearchTimer = null
      void runSearch(term, token)
    }, delay)
  }

  document.addEventListener("keydown", shortcutHandler)
  window.addCleanup(() => document.removeEventListener("keydown", shortcutHandler))
  const openHandler = () => showSearch("basic")
  searchButton.addEventListener("click", openHandler)
  window.addCleanup(() => searchButton.removeEventListener("click", openHandler))
  searchBar.addEventListener("input", onType)
  window.addCleanup(() => searchBar.removeEventListener("input", onType))
  window.addCleanup(() => {
    if (runSearchTimer !== null) {
      window.clearTimeout(runSearchTimer)
      runSearchTimer = null
    }
  })

  registerEscapeHandler(container, hideSearch)
  await fillDocument(data)
}

/**
 * Fills flexsearch document with data
 * @param data data to fill index with
 */
let indexPopulated = false
async function fillDocument(data: ContentIndex) {
  if (indexPopulated) return
  let id = 0
  const promises = []
  for (const [slug, fileData] of Object.entries<ContentDetails>(data)) {
    promises.push(
      //@ts-ignore
      index.addAsync({
        id,
        slug: slug as FullSlug,
        title: fileData.title,
        content: fileData.content,
        tags: fileData.tags,
        aliases: fileData.aliases,
      }),
    )
    id++
  }

  await Promise.all(promises)
  indexPopulated = true
}

document.addEventListener("nav", async (e: CustomEventMap["nav"]) => {
  const currentSlug = e.detail.url
  const data = await fetchData
  const searchElement = document.getElementsByClassName(
    "search",
  ) as HTMLCollectionOf<HTMLDivElement>
  for (const element of searchElement) {
    await setupSearch(element, currentSlug, data)
  }
})
