import FlexSearch, { DefaultDocumentSearchResults, DocumentData, Id } from "flexsearch"
import { SemanticClient, type SemanticResult } from "./semantic.inline"
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

type SearchType = "basic" | "tags"
type SearchMode = "lexical" | "semantic"
const SEARCH_MODE_STORAGE_KEY = "quartz:search:mode"

const loadStoredSearchMode = (): SearchMode | null => {
  if (typeof window === "undefined") {
    return null
  }

  try {
    const stored = window.localStorage.getItem(SEARCH_MODE_STORAGE_KEY)
    return stored === "lexical" || stored === "semantic" ? stored : null
  } catch (err) {
    console.warn("[Search] failed to read stored search mode:", err)
    return null
  }
}

const persistSearchMode = (mode: SearchMode) => {
  if (typeof window === "undefined") {
    return
  }

  try {
    window.localStorage.setItem(SEARCH_MODE_STORAGE_KEY, mode)
  } catch (err) {
    console.warn("[Search] failed to persist search mode:", err)
  }
}

let searchMode: SearchMode = "lexical"
let currentSearchTerm: string = ""
let rawSearchTerm: string = ""
let semantic: SemanticClient | null = null
let semanticReady = false
let semanticInitFailed = false
type SimilarityResult = { item: Item; similarity: number }
let chunkMetadata: Record<string, { parentSlug: string; chunkId: number }> = {}
let manifestIds: string[] = []

function getParentSlug(slug: string): string {
  const meta = chunkMetadata[slug]
  return meta ? meta.parentSlug : slug
}

function aggregateChunkResults(
  results: SemanticResult[],
  slugToDocIndex: Map<FullSlug, number>,
): { rrfScores: Map<number, number>; maxScores: Map<number, number> } {
  const docChunks = new Map<string, Array<{ score: number }>>()

  results.forEach(({ id, score }) => {
    const chunkSlug = manifestIds[id]
    if (!chunkSlug) return

    const parentSlug = getParentSlug(chunkSlug)

    if (!docChunks.has(parentSlug)) {
      docChunks.set(parentSlug, [])
    }

    docChunks.get(parentSlug)!.push({ score })
  })

  const rrfScores = new Map<number, number>()
  const maxScores = new Map<number, number>()
  // This can probably be tuned a bit better, i.e from the range 30-58 would be nice.
  // depending on the distribution  of chunks per documents, but for 20 i found this works decently well...
  const RRF_K = 36
  // now, some files, such as content/are.na.md that has 423 chunks (in case they are all match) RRF = sum(1/(60+i) for i in range(423)) ~ 2.8
  // comparing to a docs with 7 chunks (which is the average, currently.) with RRF = 1/60 + 1/61 + ... + 1/66 ~ 0.111
  // essentially, we will limit the MAX_CHUNKS_PER_DOCS = 20 such that we should cap the distribution a bit.
  // another strategy is to avoid/filter out outliers, and display it separately.
  const MAX_CHUNKS_PER_DOC = 20

  for (const [parentSlug, chunks] of docChunks) {
    const docIdx = slugToDocIndex.get(parentSlug as FullSlug)
    if (typeof docIdx !== "number") continue

    chunks.sort((a, b) => b.score - a.score)
    // TODO: we might want to find out the distribution based on docs that has a lot of chunks, to see which part is relevant
    const topChunks = chunks.slice(0, MAX_CHUNKS_PER_DOC)
    const rrfScore = topChunks.reduce((sum, _, rank) => sum + 1.0 / (RRF_K + rank), 0)
    const maxScore = chunks[0].score

    rrfScores.set(docIdx, rrfScore)
    maxScores.set(docIdx, maxScore)
  }

  return { rrfScores, maxScores }
}

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
const numSearchResultsLexical = 10
const numSearchResultsSemantic = 60
const numTagResults = 10

function getNumSearchResults(mode: SearchMode): number {
  return mode === "semantic" ? numSearchResultsSemantic : numSearchResultsLexical
}
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
  const container = searchElement.querySelector<HTMLDivElement>(".search-container")
  if (!container) return

  const searchButton = searchElement.querySelector<HTMLButtonElement>(".search-button")
  if (!searchButton) return

  const searchBar = searchElement.querySelector<HTMLInputElement>(".search-bar")
  if (!searchBar) return

  const searchLayout = searchElement.querySelector<HTMLOutputElement>(".search-layout")
  if (!searchLayout) return

  const searchSpace = searchElement.querySelector<HTMLFormElement>(".search-space")
  if (!searchSpace) return

  const progressBar = document.createElement("div")
  progressBar.className = "semantic-search-progress"
  searchBar.parentElement?.appendChild(progressBar)

  const startSemanticProgress = () => {
    progressBar.style.opacity = "1"
    progressBar.style.width = "0"
    setTimeout(() => {
      progressBar.style.width = "100%"
      progressBar.style.animation = "semantic-progress-sweep 2.5s linear infinite"
    }, 10)
  }

  const completeSemanticProgress = () => {
    progressBar.style.animation = "none"
    progressBar.style.opacity = "0"
    setTimeout(() => {
      progressBar.style.width = "0"
      progressBar.style.backgroundPosition = "-100% 0"
    }, 300)
  }

  const resetProgressBar = () => {
    progressBar.style.animation = "none"
    progressBar.style.opacity = "0"
    progressBar.style.width = "0"
    progressBar.style.backgroundPosition = "-100% 0"
  }

  const idDataMap = Object.keys(data) as FullSlug[]
  const slugToIndex = new Map<FullSlug, number>()
  idDataMap.forEach((slug, idx) => slugToIndex.set(slug, idx))
  const el = searchSpace.querySelector("ul#helper")
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

      try {
        const manifestUrl = "/embeddings/manifest.json"
        const res = await fetch(manifestUrl)
        if (res.ok) {
          const manifest = await res.json()
          chunkMetadata = manifest.chunkMetadata || {}
          manifestIds = manifest.ids || []
        }
      } catch (err) {
        console.warn("[Search] failed to load chunk metadata:", err)
        chunkMetadata = {}
        manifestIds = []
      }
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
  const storedMode = loadStoredSearchMode()
  if (storedMode === "semantic") {
    if (semanticReady) {
      searchMode = storedMode
    }
  } else if (storedMode === "lexical") {
    searchMode = storedMode
  }
  if (!semanticReady && searchMode === "semantic") {
    searchMode = "lexical"
  }
  let searchSeq = 0
  let runSearchTimer: number | null = null
  let lastInputAt = 0
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

  const computeDebounceDelay = (term: string): number => {
    const trimmed = term.trim()
    const lastTerm = currentSearchTerm
    const isExtension =
      lastTerm.length > 0 && trimmed.length > lastTerm.length && trimmed.startsWith(lastTerm)
    const isRetraction = lastTerm.length > trimmed.length
    const isReplacement =
      lastTerm.length > 0 && !trimmed.startsWith(lastTerm) && !lastTerm.startsWith(trimmed)
    const baseFullQueryDelay = 200
    const semanticPenalty = searchMode === "semantic" ? 30 : 0

    if (isExtension && trimmed.length > 2) {
      return baseFullQueryDelay + semanticPenalty
    }

    if (isReplacement && trimmed.length > 3) {
      return Math.max(90, baseFullQueryDelay - 80)
    }

    if (isRetraction) {
      return 90
    }

    return baseFullQueryDelay + (searchMode === "semantic" ? 40 : 0)
  }

  const triggerSearchWithMode = (mode: SearchMode) => {
    if (mode === "semantic" && !semanticReady) {
      return
    }
    if (searchMode === mode) return
    searchMode = mode
    updateModeUI(mode)
    persistSearchMode(searchMode)
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

  let results = searchLayout.querySelector(".results-container") as HTMLDivElement
  if (!results) {
    results = document.createElement("div")
    results.className = "results-container"
    appendLayout(results)
  }

  if (enablePreview) {
    preview = searchLayout.querySelector(".preview-container") as HTMLDivElement
    if (!preview) {
      preview = document.createElement("div")
      preview.className = "preview-container"
      appendLayout(preview)
    }
  }

  function hideSearch() {
    container!.classList.remove("active")
    searchBar!.value = ""
    rawSearchTerm = ""
    removeAllChildren(results)
    if (preview) {
      removeAllChildren(preview)
    }
    searchLayout!.classList.remove("display-results")
    searchButton!.focus()
    resetProgressBar()
  }

  function showSearch(type: SearchType) {
    container!.classList.add("active")
    if (type === "tags") {
      searchBar!.value = "#"
      rawSearchTerm = "#"
    }
    searchBar!.focus()
  }

  let currentHover: HTMLInputElement | null = null

  async function shortcutHandler(e: HTMLElementEventMap["keydown"]) {
    const paletteOpen = document.querySelector("search#palette-container") as HTMLDivElement
    if (paletteOpen && paletteOpen.classList.contains("active")) return

    const isBasePage = document.body?.dataset?.isBase === "true"

    if ((e.key === "/" || e.key === "k") && (e.ctrlKey || e.metaKey) && !e.shiftKey) {
      if (isBasePage && e.key.toLowerCase() === "k") {
        return
      }
      const hasInlineSearch = document.querySelector(".page-list-search-container")
      if (hasInlineSearch && e.key.toLowerCase() === "k") {
        return
      }
      e.preventDefault()
      const searchBarOpen = container!.classList.contains("active")
      searchBarOpen ? hideSearch() : showSearch("basic")
      return
    } else if (e.shiftKey && (e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
      e.preventDefault()
      const searchBarOpen = container!.classList.contains("active")
      searchBarOpen ? hideSearch() : showSearch("tags")
      return
    }

    if (currentHover) {
      currentHover.classList.remove("focus")
    }

    if (!container!.classList.contains("active")) return
    if (e.key === "Enter") {
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
    if (data[slug].layout === "letter" || (searchMode === "semantic" && slug.includes("arena")))
      return null
    const aliases: string[] = data[slug].aliases
    const target = aliases.find((alias) => alias.toLowerCase().includes(term.toLowerCase())) ?? ""

    const queryTokens = tokenizeTerm(term)
    const titleTokens = tokenizeTerm(data[slug].title ?? "")
    const titleMatch = titleTokens.some((t) => queryTokens.includes(t))

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
      titleMatch,
      protected: data[slug].protected,
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
    const isProtected = item.protected === true
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

    const fileData = data[slug]
    const fileName = (fileData?.fileName || slug || "").toLowerCase()
    const isCanvas = fileName.includes(".canvas")
    const isBases = fileName.includes(".bases")
    if (isCanvas) itemTile.dataset.canvas = "true"
    if (isBases) itemTile.dataset.bases = "true"

    if (isProtected) {
      itemTile.dataset.protected = "true"
      itemTile.innerHTML = `<hgroup>
        <h3>${titleContent}</h3>
        ${subscript}${htmlTags}
        <span class="search-protected-badge" style="
          font-size: 0.75rem;
          color: var(--gray);
          font-style: italic;
          margin-left: 0.5rem;
        ">🔒 protected content</span>
        ${searchMode === "semantic" ? `<span class="result-likelihood" title="match likelihood">&nbsp;${percentLabel}</span>` : ""}
      </hgroup>`
    } else {
      delete itemTile.dataset.protected
      itemTile.innerHTML = `<hgroup>
        <h3>${titleContent}</h3>
        ${subscript}${htmlTags}
        ${searchMode === "semantic" ? `<span class="result-likelihood" title="match likelihood">&nbsp;${percentLabel}</span>` : ""}
        ${enablePreview && window.innerWidth > 600 ? "" : `<p>${content}</p>`}
      </hgroup>`
    }

    if (percentAttr) itemTile.dataset.scorePercent = percentAttr
    else delete itemTile.dataset.scorePercent

    const handler = (evt: MouseEvent) => {
      if (evt.altKey || evt.ctrlKey || evt.metaKey || evt.shiftKey) return
      const anchor = evt.currentTarget as HTMLAnchorElement | null
      if (!anchor) return
      evt.preventDefault()
      const href = anchor.getAttribute("href")
      if (!href) return
      const url = new URL(href, window.location.toString())
      window.spaNavigate(url)
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
      removeAllChildren(preview)
    } else {
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
    if (!enablePreview || !el || !preview) return
    const slug = el.id as FullSlug
    if (el.dataset.canvas === "true" || el.dataset.bases === "true") {
      const fileData = data[slug]
      previewInner = document.createElement("div")
      previewInner.classList.add("preview-inner")

      const metaContainer = document.createElement("div")
      metaContainer.style.padding = "1rem"
      metaContainer.style.color = "var(--gray)"
      metaContainer.style.fontSize = "0.9rem"

      const filePath = document.createElement("p")
      filePath.innerHTML = `<strong>File:</strong> ${fileData?.fileName || slug}`
      metaContainer.appendChild(filePath)

      if (fileData?.description) {
        const desc = document.createElement("p")
        desc.innerHTML = `<strong>Description:</strong> ${fileData.description}`
        metaContainer.appendChild(desc)
      }

      if (fileData?.tags && fileData.tags.length > 0) {
        const tags = document.createElement("p")
        tags.innerHTML = `<strong>Tags:</strong> ${fileData.tags.join(", ")}`
        metaContainer.appendChild(tags)
      }

      previewInner.appendChild(metaContainer)
      preview.replaceChildren(previewInner)
      return
    }

    const isProtected = el.dataset.protected === "true" || data[slug]?.protected === true
    if (isProtected) {
      previewInner = document.createElement("div")
      previewInner.classList.add("preview-inner", "preview-redacted")

      const blurLayer = document.createElement("div")
      blurLayer.className = "preview-redacted-blur"
      const label = document.createElement("span")
      label.className = "preview-redacted-label"
      label.textContent = "redacted"

      previewInner.append(blurLayer, label)
      preview.replaceChildren(previewInner)
      return
    }

    const innerDiv = await fetchContent(slug).then((contents) =>
      contents.flatMap((el) => [...highlightHTML(currentSearchTerm, el as HTMLElement).children]),
    )
    previewInner = document.createElement("div")
    previewInner.classList.add("preview-inner")
    previewInner.append(...innerDiv)
    preview.replaceChildren(previewInner)

    const highlights = [...preview.getElementsByClassName("highlight")].sort(
      (a, b) => b.innerHTML.length - a.innerHTML.length,
    )
    if (highlights.length > 0) {
      const highlight = highlights[0]
      const containerRect = preview.getBoundingClientRect()
      const highlightRect = highlight.getBoundingClientRect()
      const relativeTop = highlightRect.top - containerRect.top + preview.scrollTop - 20
      preview.scrollTo({
        top: relativeTop,
        behavior: "smooth",
      })
    }
  }

  async function runSearch(rawTerm: string, token: number) {
    const trimmed = rawTerm.trim()
    if (trimmed === "") {
      removeAllChildren(results)
      if (preview) {
        removeAllChildren(preview)
      }
      currentHover = null
      searchLayout!.classList.remove("display-results")
      resetProgressBar()
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
          limit: Math.max(getNumSearchResults(modeForRanking), 10000),
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
          limit: getNumSearchResults(modeForRanking),
          index: ["tags"],
        })
        if (token !== searchSeq) return
        searchResults = Object.values(results)
        highlightTerm = tagTerm
      }
    } else {
      const results = await index.searchAsync({
        query: highlightTerm,
        limit: getNumSearchResults(modeForRanking),
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
    const semanticSimilarity = new Map<number, number>()

    const orchestrator = semanticReady && semantic ? semantic : null

    const render = async () => {
      if (token !== searchSeq) return
      const useSemantic = semanticReady && semanticIds.length > 0
      const weights =
        modeForRanking === "semantic" && useSemantic
          ? { base: 0.3, semantic: 1.0 }
          : { base: 1.0, semantic: useSemantic ? 0.3 : 0 }
      const rrf = new Map<string, number>()
      const push = (ids: number[], weight: number, applyTitleBoost: boolean = false) => {
        if (!ids.length || weight <= 0) return
        ids.forEach((docId, rank) => {
          const slug = idDataMap[docId]
          if (!slug) return
          const item = ensureItem(docId)
          if (!item) return

          let effectiveWeight = weight
          if (applyTitleBoost && item.titleMatch) {
            effectiveWeight *= 1.5
          }

          const prev = rrf.get(slug) ?? 0
          rrf.set(slug, prev + effectiveWeight / (1 + rank))
        })
      }

      push(baseIndices, weights.base, true)
      push(semanticIds, weights.semantic, false)

      const entries = Array.from(candidateItems.values()).map((item) => ({
        item,
        score: rrf.get(item.slug) ?? 0,
        similarity: semanticSimilarity.get(item.id) ?? Number.NaN,
      }))

      const rankedEntries =
        modeForRanking === "semantic" && useSemantic
          ? entries
              .sort((a, b) => {
                const aHas = Number.isFinite(a.similarity)
                const bHas = Number.isFinite(b.similarity)
                if (aHas && bHas) return b.similarity - a.similarity
                if (aHas) return -1
                if (bHas) return 1
                return b.score - a.score
              })
              .slice(0, getNumSearchResults(modeForRanking))
          : entries.sort((a, b) => b.score - a.score).slice(0, getNumSearchResults(modeForRanking))

      const displayEntries: SimilarityResult[] = []
      for (const entry of rankedEntries) {
        displayEntries.push({ item: entry.item, similarity: entry.similarity })
      }

      await displayResults(displayEntries)
    }

    await render()

    if (workingType === "tags" || !orchestrator || !semanticReady || highlightTerm.length < 2) {
      return
    }

    const showProgress = modeForRanking === "semantic"
    if (showProgress) {
      startSemanticProgress()
    }

    try {
      const { semantic: semRes } = await orchestrator.search(
        highlightTerm,
        getNumSearchResults(modeForRanking) * 8,
      )
      if (token !== searchSeq) {
        if (showProgress) completeSemanticProgress()
        return
      }

      const { rrfScores: semRrfScores, maxScores: semMaxScores } = aggregateChunkResults(
        semRes,
        slugToIndex,
      )

      semanticIds = Array.from(semRrfScores.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, getNumSearchResults(modeForRanking))
        .map(([docIdx]) => docIdx)

      semanticSimilarity.clear()
      semMaxScores.forEach((score, docIdx) => {
        semanticSimilarity.set(docIdx, score)
      })

      semanticIds.forEach((docId) => {
        ensureItem(docId)
      })
      if (showProgress) completeSemanticProgress()
    } catch (err) {
      console.warn("[SemanticClient] search failed:", err)
      if (showProgress) completeSemanticProgress()
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
    rawSearchTerm = (e.target as HTMLInputElement).value
    const hasQuery = rawSearchTerm.trim() !== ""
    searchLayout!.classList.toggle("display-results", hasQuery)
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
    lastInputAt = performance.now()
    const delay = computeDebounceDelay(term)
    const scheduledAt = lastInputAt
    runSearchTimer = window.setTimeout(() => {
      if (scheduledAt !== lastInputAt) {
        return
      }
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
    resetProgressBar()
  })

  registerEscapeHandler(container, hideSearch)
  await fillDocument(data)
}

let indexPopulated = false
async function fillDocument(data: ContentIndex) {
  if (indexPopulated) return
  let id = 0
  const promises = []
  for (const [slug, fileData] of Object.entries<ContentDetails>(data)) {
    promises.push(
      index.addAsync({
        id,
        slug: slug as FullSlug,
        title: fileData.title,
        content: fileData.content,
        tags: fileData.tags,
        aliases: fileData.aliases,
        target: "",
        protected: fileData.protected,
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
