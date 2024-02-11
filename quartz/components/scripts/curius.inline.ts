import FlexSearch, { IndexOptions } from "flexsearch"
import { registerEscapeHandler, removeAllChildren, registerEvents } from "./util"
import { computePosition, arrow as arrowFloating, inline, offset } from "@floating-ui/dom"
import type { Coords } from "@floating-ui/dom"
import { sample } from "../../util/helpers"

interface Entity {
  id: number
  createdDate: string
  modifiedDate: string
}

interface Highlight extends Entity {
  userId: number
  linkId: number
  highlight: string
  leftContext: string
  rightContext: string
  rawHighlight: string
  comment_ids: string[]
  comment: string
}

interface Topic extends Entity {
  userId: number
  topic: string
  slug: string
  public: boolean
}

interface User extends Entity {
  firstName: string
  lastName: string
  major?: string
  interests?: string
  expertise?: string
  school: string
  github?: string
  twitter: string
  website: string
  lastOnline: string
  lastCheckedNotifications: string
  views: number
  numFollowers: number
  followed?: boolean
  followingMe?: boolean
  recentUsers: any[]
  followingUsers: Following[]
}

interface Following {
  id: number
  firstName: string
  lastName: string
  userLink: string
  lastOnline: string
}

interface Response {
  links?: Link[]
  user?: User
}

interface Trail {
  id: number
  trailName: string
  ownerId: number
  description: string
  colorHex: string
  emojiUnicode: string
  flipped: any
  hash: string
  slug: string
  createdDate: string
}

interface Link extends Entity {
  link: string
  title: string
  favorite: boolean
  snippet: string
  toRead: any
  createdBy: number
  metadata: {
    full_text: string
    author: string
    page_type: string
  }
  lastCrawled: any
  trails: Trail[]
  comments: string[]
  mentions: string[]
  topics: Topic[]
  highlights: Highlight[]
  userIds?: number[]
}

const _SENTINEL: Link = {
  id: 0,
  link: "",
  title: "",
  favorite: false,
  snippet: "",
  toRead: null,
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

const localFetchKey = "curiusLinks"
const localTimeKey = "curiusLastFetch"
const numSearchResults = 20
const refetchTimeout = 2 * 60 * 1000 // 2 minutes
const fetchLinksHeaders: RequestInit = {
  method: "POST",
  headers: { "Content-Type": "application/json" },
}
const externalLinkRegex = /^(?:https?:\/\/)?(?:www\.)?([^\/]+)/

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

const _IconMapping = {
  favourite: `<svg fill="currentColor" preserveAspectRatio="xMidYMid meet" height="1em" width="1em" viewBox="0 0 40 40" data-tip="Unfavorite" data-for="links" style="vertical-align: unset;"><g><path d="m5.2 18.8l6 5.5-1.7 7.7c-0.2 1 0.2 2 1 2.5 0.3 0.3 0.8 0.5 1.3 0.5 0.4 0 0.7 0 1-0.2 0 0 0.2 0 0.2-0.1l6.8-3.9 6.9 3.9s0.1 0 0.1 0.1c0.9 0.4 1.9 0.4 2.5-0.1 0.9-0.5 1.2-1.5 1-2.5l-1.6-7.7c0.6-0.5 1.6-1.5 2.6-2.5l3.2-2.8 0.2-0.2c0.6-0.7 0.8-1.7 0.5-2.5s-1-1.5-2-1.7h-0.2l-7.8-0.8-3.2-7.2s0-0.1-0.2-0.1c-0.1-1.2-1-1.7-1.8-1.7s-1.7 0.5-2.2 1.3c0 0 0 0.2-0.1 0.2l-3.2 7.2-7.8 0.8h-0.2c-0.8 0.2-1.7 0.8-2 1.7-0.2 1 0 2 0.7 2.6z"></path></g></svg>`,
}

type Keys = keyof typeof _IconMapping

const Icons = (name: Keys) => _IconMapping[name] ?? null

function timeSince(date: Date | string) {
  const now = new Date()
  const dateObject = date instanceof Date ? date : new Date(date)
  const diff = Math.floor((now.getTime() - dateObject.getTime()) / 1000)
  const days = Math.floor(diff / (3600 * 24))
  const hours = Math.floor((diff % (3600 * 24)) / 3600)
  const minutes = Math.floor((diff % 3600) / 60)

  if (days > 1) {
    return `${days} days ago`
  } else if (days === 1) {
    return `1 day ago`
  } else if (hours > 1) {
    return `${hours} hours ago`
  } else if (hours === 1) {
    return `1 hour ago`
  } else if (minutes > 1) {
    return `${minutes} minutes ago`
  } else if (minutes === 1) {
    return `1 minute ago`
  } else {
    return `just now`
  }
}

function getLocalItem(key: "curiusLinks" | "curiusLastFetch", value: any) {
  return localStorage.getItem(key) ?? value
}

function extractApexDomain(url: string) {
  const match = url.match(externalLinkRegex)
  return match ? match[1] : ""
}

async function fetchLinks(refetch: boolean = false): Promise<Response> {
  // user metadata
  const user = await fetch("https://raw.aarnphm.xyz/api/curius?query=user", fetchLinksHeaders)
    .then((res): Promise<Response> => res.json())
    .then((data) => {
      if (data === undefined || data.user === undefined) {
        throw new Error("Failed to fetch user")
      }
      return data.user
    })

  const currentTime = new Date()
  const lastFetched = new Date(getLocalItem(localTimeKey, 0))
  // set fetched period to 5 minutes
  const periods = 5 * 60 * 1000

  const getCachedLinks = () => JSON.parse(getLocalItem(localFetchKey, "[]"))

  if (!refetch && currentTime.getTime() - lastFetched.getTime() < periods) {
    return { links: getCachedLinks(), user }
  }

  localStorage.setItem(localTimeKey, currentTime.toString())

  // fetch new links
  const newLinks: Link[] = await fetch(
    "https://raw.aarnphm.xyz/api/curius?query=links",
    fetchLinksHeaders,
  )
    .then((res) => res.json())
    .then((data: Response) => {
      if (data === undefined || data.links === undefined) {
        throw new Error("Failed to fetch links")
      }
      return data.links
    })

  if (JSON.stringify(getCachedLinks()) !== JSON.stringify(newLinks)) {
    localStorage.setItem(localFetchKey, JSON.stringify(newLinks))
  }
  return { links: newLinks, user }
}

let currentActive: HTMLLIElement | null = null

function createLinkEl(Link: Link): HTMLLIElement {
  const curiusItem = document.createElement("li")
  curiusItem.id = `curius-item-${Link.id}`

  const createTitle = (Link: Link): HTMLDivElement => {
    const item = document.createElement("div")
    item.classList.add("curius-item-title")

    const header = document.createElement("div")
    header.classList.add("curius-item-link")

    const link = document.createElement("a")
    Object.assign(link, {
      href: Link.link,
      target: "_blank",
      rel: "noopener noreferrer",
      innerHTML: `<span class="curius-item-span">${Link.title}</span>`,
    })
    header.appendChild(link)

    const address = document.createElement("div")
    address.classList.add("curius-item-address")
    address.textContent = extractApexDomain(Link.link)

    const icons = document.createElement("div")
    icons.classList.add("curius-item-icons")
    if (Link.favorite) {
      const icon = document.createElement("span")
      icon.classList.add("curius-favourite")
      icon.innerHTML = Icons("favourite")
      icons.appendChild(icon)
    }

    item.append(header, address, icons)
    return item
  }

  const createMetadata = (Link: Link): HTMLDivElement => {
    const item = document.createElement("div")
    item.classList.add("curius-item-metadata")

    const tags = document.createElement("ul")
    tags.classList.add("curius-item-tags")
    tags.innerHTML =
      Link.topics.length > 0
        ? `${Link.topics
            .map((topic) =>
              topic.public
                ? `<li><a href="https://curius.app/aaron-pham/${topic.slug}" target="_blank">${topic.topic}</a></li>`
                : ``,
            )
            .join("")}`
        : ``

    const misc = document.createElement("div")
    misc.id = `curius-misc-${Link.id}`
    const time = document.createElement("span")
    time.id = `curius-span-${Link.id}`
    const modifiedDate = new Date(Link.modifiedDate)
    time.innerHTML = `<time datetime=${
      Link.modifiedDate
    } title="${modifiedDate.toUTCString()}">${timeSince(Link.createdDate)}</time>`
    misc.appendChild(time)

    if (Link.highlights.length > 0) {
      const highlights = document.createElement("div")
      highlights.id = `curius-highlights-${Link.id}`
      highlights.innerHTML = `${Link.highlights.length} highlight`
      misc.appendChild(highlights)

      const modal = document.getElementById("highlight-modal")
      const modalList = document.getElementById("highlight-modal-list")

      const onMouseEnter = () => {
        const highlightsData = Link.highlights

        if (!modal || !modalList) return
        // clear the previous modal
        modalList.innerHTML = ""
        curiusItem.classList.remove("focus")

        highlightsData.forEach((highlight) => {
          let hiItem = document.createElement("li")
          hiItem.textContent = highlight.highlight
          modalList.appendChild(hiItem)
        })
        modal.style.visibility = "visible"
        modal.classList.add("active")
      }

      const onMouseLeave = () => {
        curiusItem.classList.add("focus")

        if (!modal) return
        modal.style.visibility = "hidden"
        modal.classList.remove("active")
      }

      const onMouseMove = ({ pageX, pageY }: MouseEvent) => {
        curiusItem.classList.remove("focus")

        if (!modal) return
        modal.classList.add("active")
        Object.assign(modal.style, {
          left: `${pageX + 10}px`,
          top: `${pageY + 10}px`,
        })
      }

      const events = [
        ["mouseenter", onMouseEnter],
        ["mouseleave", onMouseLeave],
        ["mousemove", onMouseMove],
      ] as [keyof HTMLElementEventMap, (this: HTMLElement) => void][]
      registerEvents(highlights, ...events)
    }

    item.append(tags, misc)
    return item
  }

  curiusItem.append(createTitle(Link), createMetadata(Link))

  const onMouseEnter = () => curiusItem.classList.add("focus")

  const onMouseLeave = () => curiusItem.classList.remove("focus")

  const onClick = (e: HTMLElementEventMap["click"]) => {
    if (e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return
    const note = document.querySelector("#curius-notes") as HTMLDivElement | null
    if (!note) return

    if (currentActive) {
      currentActive.classList.remove("active")
    }
    note.classList.add("active")
    curiusItem.classList.add("active")
    currentActive = curiusItem
    updateNotePanel(Link, note, curiusItem)
  }

  function onKeydown(e: HTMLElementEventMap["keydown"]) {
    const note = document.querySelector("#curius-notes") as HTMLDivElement | null
    if (!note) return

    if (e.key === "Escape") {
      e.preventDefault()
      note.classList.remove("active")
      if (currentActive) currentActive.classList.remove("active")
      else curiusItem.classList.remove("active")
      return
    }
  }

  document.addEventListener("keydown", onKeydown)
  window.addCleanup(() => document.removeEventListener("keydown", onKeydown))

  const events = [
    ["mouseenter", onMouseEnter],
    ["mouseleave", onMouseLeave],
    ["click", onClick],
  ] as [keyof HTMLElementEventMap, (this: HTMLElement) => void][]

  registerEvents(curiusItem, ...events)

  return curiusItem
}

function updateNotePanel(Link: Link, note: HTMLDivElement, parent: HTMLLIElement) {
  const titleNode = note.querySelector("#note-link") as HTMLAnchorElement
  const snippetNode = note.querySelector(".curius-note-snippet") as HTMLDivElement
  const highlightsNode = note.querySelector(".curius-note-highlights") as HTMLDivElement

  titleNode.innerHTML = `<span class="curius-item-span">${Link.title}</span>`
  titleNode.href = Link.link
  titleNode.target = "_blank"
  titleNode.rel = "noopener noreferrer"

  const close = document.querySelector(".icon-container")
  const cleanUp = () => {
    note.style.visibility = "hidden"
    note.classList.remove("active")
    parent.classList.remove("active")
  }
  close?.addEventListener("click", cleanUp)
  window.addCleanup(() => close?.removeEventListener("click", cleanUp))

  removeAllChildren(snippetNode)
  snippetNode.textContent = Link.snippet

  removeAllChildren(highlightsNode)
  if (Link.highlights.length === 0) return
  for (const hl of Link.highlights) {
    const highlightItem = document.createElement("li")
    const hlLink = document.createElement("a")
    hlLink.dataset.highlight = hl.id.toString()
    hlLink.href = `${Link.link}?curius=${hl.userId}`
    hlLink.target = "_blank"
    hlLink.rel = "noopener noreferrer"
    hlLink.textContent = hl.highlight
    highlightItem.appendChild(hlLink)
    highlightsNode.appendChild(highlightItem)
  }
}

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
  const elements = [
    "#curius-container",
    "#curius-fetching-text",
    "#curius-fragments",
    ".navigation-container",
    ".total-links",
  ].map((id) => document.querySelector(id))

  if (elements.some((el) => el === null)) return

  const [container, fetchText, fragment, nav, total] = elements as HTMLElement[]

  fetchText.textContent = "Récupération des liens curius"
  fetchText.classList.toggle("active", true)
  const resp = await fetchLinks()
  fetchText.classList.toggle("active", false)

  const linksData = resp.links ?? []

  const callIfEmpty = (data: Link[]) => {
    if (data.length === 0) {
      container.innerHTML = `<p>Échec de la récupération des liens.</p>`
      return
    }
  }

  callIfEmpty(linksData)
  fragment.append(...linksData.map(createLinkEl))
  total.textContent = `${linksData.length} éléments`
  nav.classList.toggle("active", true)

  const refetchIcon = document.getElementById("curius-refetch")

  // Ensure refetchIcon exists before adding event listener
  if (refetchIcon) {
    const preventRefreshDefault = (e: HTMLElementEventMap["keydown"]) => {
      const icon = document.getElementById("curius-refetch")
      if (!icon || !icon.classList.contains("disabled")) return
      if ((e.key === "r" || e.key === "R") && (e.ctrlKey || e.metaKey)) {
        e.preventDefault()
        e.stopPropagation()
      }
    }

    document.addEventListener("keydown", preventRefreshDefault)
    window.addCleanup(() => document.removeEventListener("keydown", preventRefreshDefault))

    let isTimeout = false

    const onClick = async () => {
      if (isTimeout) return

      const searchContainer = document.getElementById(
        "curius-search-container",
      ) as HTMLDivElement | null

      if (searchContainer?.classList.contains("active")) {
        searchContainer.classList.remove("active")
      }

      refetchIcon.classList.add("disabled")
      refetchIcon.style.opacity = "0.5"

      removeAllChildren(fragment)
      nav.classList.toggle("active", false)

      fetchText.classList.toggle("active", true)
      fetchText.textContent = "Rafraîchissement des liens curius"
      const refetched = await fetchLinks(true)
      fetchText.classList.toggle("active", false)

      const newData = refetched.links ?? []
      callIfEmpty(newData)
      fragment.append(...newData.map(createLinkEl))
      total.textContent = `${newData.length} éléments`
      nav.classList.toggle("active", true)

      isTimeout = true
      setTimeout(() => {
        refetchIcon.classList.remove("disabled")
        refetchIcon.style.opacity = "0"
        isTimeout = false
      }, refetchTimeout)
    }

    refetchIcon.addEventListener("click", onClick)
    window.addCleanup(() => refetchIcon.removeEventListener("click", onClick))

    const events = [
      [
        "mouseenter",
        () =>
          (refetchIcon.style.opacity = refetchIcon.classList.contains("disabled") ? "0.5" : "1"),
      ],
      [
        "mouseleave",
        () =>
          (refetchIcon.style.opacity = refetchIcon.classList.contains("disabled") ? "0.5" : "0"),
      ],
    ] as [keyof HTMLElementEventMap, EventListenerOrEventListenerObject][]
    registerEvents(refetchIcon, ...events)
  }
})

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
    const searchBarOpen = container?.classList.contains("active")
    searchBarOpen ? hideLinks() : showLinks(sampleLinks)
  }

  function showLinks(links: Link[]) {
    if (!container) return
    container?.classList.add("active")
    bar?.focus()
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
