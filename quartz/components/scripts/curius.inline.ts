import FlexSearch, { IndexOptions } from "flexsearch"
import { registerEscapeHandler, removeAllChildren, registerEvents } from "./util"
import { computePosition, arrow as arrowFloating, inline, offset } from "@floating-ui/dom"
import type { Coords } from "@floating-ui/dom"

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

interface Link extends Entity {
  link: string
  title: string
  favorite: boolean
  snippet: string
  toRead: any
  lastCrawled: any
  topics: Topic[]
  highlights: Highlight[]
  userIds?: number[]
}

const _Link: Link = {
  id: 0,
  link: "",
  title: "",
  favorite: false,
  snippet: "",
  toRead: null,
  createdDate: "",
  modifiedDate: "",
  lastCrawled: null,
  topics: [],
  highlights: [],
  userIds: [],
}

const localFetchKey = "curiusLinks"
const localTimeKey = "curiusLastFetch"
const numSearchResults = 20
const fetchTimeout = 2 * 60 * 1000 // 2 minutes
const fetchLinksHeaders: RequestInit = {
  method: "POST",
  headers: { "Content-Type": "application/json" },
}
const externalLinkRegex = /^(?:https?:\/\/)?(?:www\.)?([^\/]+)/

let disableEndTime: number | null = null
let countdownIntervalId: NodeJS.Timeout | null = null
let index: FlexSearch.Document<Link> = new FlexSearch.Document({
  charset: "latin:advanced",
  document: {
    id: "id",
    index: [
      ...Object.keys(_Link).map(
        (key) =>
          ({ field: key, tokenize: "forward" }) as IndexOptions<Link, false> & {
            field: string
          },
      ),
    ],
  },
})

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

function random(min: number, max: number) {
  if (max == null) {
    max = min
    min = 0
  }
  return min + Math.floor(Math.random() * (max - min + 1))
}

function sample(object: any[], n: number) {
  const sample = [...object]
  var length = sample.length
  n = Math.max(Math.min(n, length), 0)
  var last = length - 1
  for (var index = 0; index < n; index++) {
    var rand = random(index, last)
    var temp = object[index]
    sample[index] = sample[rand]
    sample[rand] = temp
  }
  return sample.slice(0, n)
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

    item.append(header, address)
    return item
  }

  const createMetadata = (Link: Link): HTMLDivElement => {
    const item = document.createElement("div")
    item.classList.add("curius-item-metadata")

    const tags = document.createElement("div")
    tags.classList.add("curius-item-tags")
    tags.innerHTML =
      Link.topics.length > 0
        ? `${Link.topics
            .map((topic) =>
              topic.public
                ? `<ul><a href="https://curius.app/aaron-pham/${topic.slug}" target="_blank">${topic.topic}</a></ul>`
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

  const onMouseEnter = (e: HTMLElementEventMap["mouseenter"]) => curiusItem.classList.add("focus")

  const onMouseLeave = (e: HTMLElementEventMap["mouseleave"]) =>
    curiusItem.classList.remove("focus")

  const onClick = (e: HTMLElementEventMap["click"]) => {
    if (e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return
    if (currentActive) {
      currentActive.classList.remove("active")
    }
    curiusItem.classList.add("active")
    currentActive = curiusItem
  }

  function onKeydown(e: HTMLElementEventMap["keydown"]) {
    if (e.key === "Escape") {
      e.preventDefault()
      curiusItem.classList.remove("active")
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

const toggleVisibility = (el: HTMLElement, state: boolean) => {
  Object.assign(el.style, {
    opacity: state ? "1" : "0",
    visibility: state ? "visible" : "hidden",
  })
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
  ].map((id) => document.querySelector(id))

  if (elements.some((el) => el === null)) return

  const [container, fetchText, fragment, nav] = elements as HTMLElement[]

  fetchText.textContent = "Fetching curius links"
  toggleVisibility(fetchText, true)
  const resp = await fetchLinks()
  toggleVisibility(fetchText, false)

  const linksData = resp.links ?? []

  if (linksData.length === 0) {
    container.innerHTML = `<p>Failed to fetch links.</p>`
    return
  }
  fragment.append(...linksData.map(createLinkEl))
  toggleVisibility(nav, true)

  const refetchIcon = document.getElementById("curius-refetch")

  // Ensure refetchIcon exists before adding event listener
  if (refetchIcon) {
    const refetchContent = refetchIcon.dataset.tooltip as string

    const preventRefreshDefault = (e: HTMLElementEventMap["keydown"]) => {
      if (!refetchIcon.classList.contains("disabled")) return
      if ((e.key === "r" || e.key === "R") && (e.ctrlKey || e.metaKey)) {
        e.preventDefault()
        e.stopPropagation()
      }
    }

    document.addEventListener("keydown", preventRefreshDefault)
    window.addCleanup(() => document.removeEventListener("keydown", preventRefreshDefault))

    const onClick = async () => {
      refetchIcon.classList.add("disabled")
      refetchIcon.style.opacity = "0.5"

      removeAllChildren(fragment)
      toggleVisibility(nav, false)

      toggleVisibility(fetchText, true)
      fetchText.textContent = "Refreshing curius links"
      const refetched = await fetchLinks(true)
      toggleVisibility(fetchText, false)

      const newData = refetched.links ?? []
      if (newData.length === 0) {
        container.innerHTML = `<p>Failed to fetch links.</p>`
        return
      }
      fragment.append(...newData.map(createLinkEl))
      toggleVisibility(nav, true)
    }

    refetchIcon.addEventListener("click", onClick)
    window.addCleanup(() => refetchIcon.removeEventListener("click", onClick))

    const events = [
      ["mouseenter", () => (refetchIcon.style.opacity = "1")],
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

    if (currentActive) {
      currentActive.classList.remove("active")
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
