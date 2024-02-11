import FlexSearch, { IndexOptions } from "flexsearch"
import { sample } from "../../util/helpers"
import { fetchLinks } from "./curius.inline"
import { Link } from "../types"
import { registerEscapeHandler, removeAllChildren } from "./util"

const _SENTINEL: Link = {
  id: 0,
  link: "",
  title: "",
  favorite: false,
  snippet: "",
  toRead: false,
  createdBy: 0,
  metadata: {
    full_text: "",
    author: "",
    page_type: "",
  },
  createdDate: "",
  modifiedDate: "",
  lastCrawled: null,
  trails: [],
  comments: [],
  mentions: [],
  topics: [],
  highlights: [],
  userIds: [],
}

let index: FlexSearch.Document<Link> = new FlexSearch.Document({
  charset: "latin:advanced",
  document: {
    id: "id",
    index: [
      ...Object.keys(_SENTINEL).map(
        (key) =>
          ({ field: key, tokenize: "forward" }) as IndexOptions<Link, false> & {
            field: string
          },
      ),
    ],
  },
})

const numSearchResults = 20
const contextWindowWords = 30

const tokenizeTerm = (term: string) => {
  const tokens = term.split(/\s+/).filter((t) => t.trim() !== "")
  const tokenLen = tokens.length
  if (tokenLen > 1) {
    for (let i = 1; i < tokenLen; i++) {
      tokens.push(tokens.slice(0, i + 1).join(" "))
    }
  }

  return tokens.sort((a, b) => b.length - a.length) // always highlight longest terms first
}

function highlight(searchTerm: string, text: string, trim?: boolean) {
  const tokenizedTerms = tokenizeTerm(searchTerm)
  let tokenizedText = text.split(/\s+/).filter((t) => t !== "")

  let startIndex = 0
  let endIndex = tokenizedText.length - 1
  if (trim) {
    const includesCheck = (tok: string) =>
      tokenizedTerms.some((term) => tok.toLowerCase().startsWith(term.toLowerCase()))
    const occurrencesIndices = tokenizedText.map(includesCheck)

    let bestSum = 0
    let bestIndex = 0
    for (let i = 0; i < Math.max(tokenizedText.length - contextWindowWords, 0); i++) {
      const window = occurrencesIndices.slice(i, i + contextWindowWords)
      const windowSum = window.reduce((total, cur) => total + (cur ? 1 : 0), 0)
      if (windowSum >= bestSum) {
        bestSum = windowSum
        bestIndex = i
      }
    }

    startIndex = Math.max(bestIndex - contextWindowWords, 0)
    endIndex = Math.min(startIndex + 2 * contextWindowWords, tokenizedText.length - 1)
    tokenizedText = tokenizedText.slice(startIndex, endIndex)
  }

  const slice = tokenizedText
    .map((tok) => {
      // see if this tok is prefixed by any search terms
      for (const searchTok of tokenizedTerms) {
        if (tok.toLowerCase().includes(searchTok.toLowerCase())) {
          const regex = new RegExp(searchTok.toLowerCase(), "gi")
          return tok.replace(regex, `<span class="highlight">$&</span>`)
        }
      }
      return tok
    })
    .join(" ")

  return `${startIndex === 0 ? "" : "..."}${slice}${
    endIndex === tokenizedText.length - 1 ? "" : "..."
  }`
}

document.addEventListener("nav", async (e: CustomEventMap["nav"]) => {
  const bar = document.getElementById("curius-bar") as HTMLInputElement | null
  const container = document.getElementById("curius-search-container") as HTMLDivElement | null

  const resp = await fetchLinks()
  const linksData = resp.links ?? []
  const sampleLinks = sample(linksData, 20)

  // Search functionality
  async function onType(e: HTMLElementEventMap["input"]) {
    let term = (e.target as HTMLInputElement).value
    container?.classList.toggle("active", term !== "")
    let searchResults =
      (await index?.searchAsync({
        query: term,
        limit: numSearchResults,
        index: ["title", "snippet", "topics"],
      })) ?? []

    const getByField = (field: string): number[] => {
      const results = searchResults.filter((x) => x.field === field)
      return results.length === 0 ? [] : ([...results[0].result] as number[])
    }

    const allIds: Set<number> = new Set([
      ...getByField("title"),
      ...getByField("snippet"),
      ...getByField("topics"),
    ])

    const finalResults = [...allIds].map((id) => formatLinks(term, id))
    displayLinks(finalResults)
  }

  const formatLinks = (term: string, id: number): Link => {
    const L = linksData[id]
    return {
      ...L,
      title: highlight(term, L.title),
      snippet: highlight(term, L.snippet, true),
    }
  }

  const notes = document.getElementsByClassName("curius-notes")[0] as HTMLDivElement | null

  function displayLinks(links: Link[]) {
    if (!container) return
    removeAllChildren(container)

    if (links.length === 0) {
      container.innerHTML = `<a class="curius-search-link"><span class="curius-search-title">No results found.</span><p class="curius-search-snippet">Try another search term?</p></a>`
    } else {
      container?.append(...links.map(createSearchLinks))
    }
  }

  function shortcutHandler(e: HTMLElementEventMap["keydown"]) {
    if (e.key === "k" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault()
      if (notes?.classList.contains("active")) notes.classList.remove("active")
      const searchBarOpen = container?.classList.contains("active")
      searchBarOpen ? hideLinks() : showLinks(sampleLinks)
      return
    }

    if (!container?.classList.contains("active")) return
    if (e.key === "Enter") {
      if (container?.contains(document.activeElement)) {
        const active = document.activeElement as HTMLInputElement
        active.click()
      } else {
        const anchor = document.getElementsByClassName(
          "curius-search-link",
        )[0] as HTMLInputElement | null
        anchor?.click()
      }
    } else if (e.key === "ArrowUp" || (e.shiftKey && e.key === "Tab")) {
      e.preventDefault()
      // When first pressing ArrowDown, results wont contain the active element, so focus first element
      if (container?.contains(document.activeElement)) {
        const prevResult = document.activeElement?.previousElementSibling as HTMLInputElement | null
        prevResult?.focus()
      }
    } else if (e.key === "ArrowDown" || e.key === "Tab") {
      e.preventDefault()

      // When first pressing ArrowDown, results wont contain the active element, so focus first element
      if (!container?.contains(document.activeElement)) {
        const firstResult = document.getElementsByClassName(
          "curius-search-link",
        )[0] as HTMLInputElement | null
        firstResult?.focus()
      } else {
        // If an element in results-container already has focus, focus next one
        const nextResult = document.activeElement?.nextElementSibling as HTMLInputElement | null
        nextResult?.focus()
      }
    }
  }

  function onClick(e: HTMLElementEventMap["click"]) {
    if (bar?.classList.contains("active")) return
    if (notes?.classList.contains("active")) notes.classList.remove("active")
    const searchBarOpen = container?.classList.contains("active")
    searchBarOpen ? hideLinks() : showLinks(sampleLinks)
  }

  function showLinks(links: Link[]) {
    if (!container) return
    container?.classList.add("active")
    bar?.focus()
    bar?.scrollIntoView({ behavior: "smooth" })
    displayLinks(links)
  }

  function hideLinks() {
    if (container) container.classList.remove("active")
    if (bar) bar.value = ""
  }

  function createSearchLinks(link: Link): HTMLAnchorElement {
    const curiusLink = document.createElement("a")
    curiusLink.classList.add("curius-search-link")
    curiusLink.target = "_blank"
    curiusLink.href = link.link
    curiusLink.innerHTML = `<span class="curius-search-title">${link.title}</span><p class="curius-search-snippet">${link.snippet}</div>`

    const onClick = (e: MouseEvent) => {
      if (e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return
      hideLinks()
    }

    curiusLink.addEventListener("click", onClick)
    window.addCleanup(() => curiusLink.removeEventListener("click", onClick))

    return curiusLink
  }

  document.addEventListener("keydown", shortcutHandler)
  window.addCleanup(() => document.removeEventListener("keydown", shortcutHandler))
  bar?.addEventListener("input", onType)
  window.addCleanup(() => bar?.removeEventListener("input", onType))
  bar?.addEventListener("click", onClick)
  window.addCleanup(() => bar?.removeEventListener("click", onClick))

  registerEscapeHandler(container, hideLinks)

  await fillIndex(linksData)
})

async function fillIndex(links: Link[]) {
  let id = 0
  const promises: Array<Promise<unknown>> = []
  for (const link of links) {
    promises.push(index.addAsync(id++, { ...link }))
  }
  return await Promise.all(promises)
}
