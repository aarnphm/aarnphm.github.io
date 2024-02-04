import FlexSearch, { IndexOptions } from "flexsearch"
import { pluralize } from "../../util/lang"
import { registerEscapeHandler, removeAllChildren, registerEvents } from "./util"
import { computePosition, arrow as arrowFloating, inline, offset } from "@floating-ui/dom"
import type { Coords } from "@floating-ui/dom"

interface Highlight {
  id: number
  userId: number
  linkId: number
  highlight: string
  createdDate: string
  leftContext: string
  rightContext: string
  rawHighlight: string
  comment_ids: string[]
  comment: string
}

interface Topic {
  id: number
  userId: number
  topic: string
  slug: string
  public: boolean
  createdDate: string
  modifiedDate: string
}

interface User {
  id: number
  firstName: string
  lastName: string
  major?: string
  interests?: string
  expertise?: string
  school: string
  github?: string
  twitter: string
  website: string
  createdDate: string
  modifiedDate: string
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

interface Link {
  id: number
  link: string
  title: string
  favorite: boolean
  snippet: string
  toRead: any
  createdDate: string
  modifiedDate: string
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
  charset: "latin:extra",
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

function createNotePanel(Link: Link, note: HTMLDivElement) {
  removeAllChildren(note)

  function createNoteTitle() {
    const titleDiv = document.createElement("div")
    titleDiv.classList.add("curius-note-title")
    Object.assign(titleDiv.style, {
      display: "flex",
      alignItems: "flex-start",
      fontSize: "1.2rem",
    })
    const title = document.createElement("a")
    Object.assign(title, {
      href: Link.link,
      target: "_blank",
      rel: "noopener noreferrer",
      innerHTML: `<span class="curius-item-span">${Link.title}</span>`,
    })
    const icon = document.createElement("div")
    const svg = document.createElement("svg")
    const attrs = {
      viewBox: "0 0 24 24",
      focusable: "false",
      fill: "currentColor",
      width: "15px",
      height: "15px",
    }
    for (const [key, value] of Object.entries(attrs)) {
      svg.setAttribute(key, value)
    }
    Object.assign(svg.style, {
      color: "rgba(170, 170, 170)",
      display: "inline-block",
      fontSize: "1.5rem",
      transition: "fill 200ms cubic-bezier(0.4, 0, 0.2, 1) 0ms",
      flexShrink: "0",
      userSelect: "none",
    })
    const path = document.createElement("path")
    path.setAttribute(
      "d",
      "M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z",
    )
    svg.appendChild(path)
    icon.appendChild(svg)

    titleDiv.append(title, icon)
    return titleDiv
  }

  function createSnippet() {
    const linkSnippet = document.createElement("div")
    linkSnippet.classList.add("curius-note-snippet")
    linkSnippet.textContent = Link.snippet
    return linkSnippet
  }

  function createHighlights() {
    const highlight = document.createElement("div")
    highlight.classList.add("curius-note-highlights")
    if (Link.highlights.length === 0) return highlight
    for (const hl of Link.highlights) {
      const highlightItem = document.createElement("li")
      const hlLink = document.createElement("a")
      hlLink.dataset.highlight = hl.id.toString()
      hlLink.href = `${Link.link}?curius=${hl.userId}`
      hlLink.textContent = hl.highlight
      highlightItem.appendChild(hlLink)
      highlight.appendChild(highlightItem)
    }
    return highlight
  }

  note.append(createNoteTitle(), createSnippet(), createHighlights())
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
      highlights.innerHTML = `${pluralize(Link.highlights.length, "highlight")}`
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
    const note = document.querySelector("#curius-notes") as HTMLDivElement | null
    if (!note) return

    if (currentActive) {
      currentActive.classList.remove("active")
    }
    note.classList.add("active")
    note.style.visibility = "visible"
    curiusItem.classList.add("active")
    currentActive = curiusItem
    createNotePanel(Link, note)
  }

  const events = [
    ["mouseenter", onMouseEnter],
    ["mouseleave", onMouseLeave],
    ["click", onClick],
  ] as [keyof HTMLElementEventMap, (this: HTMLElement) => void][]

  registerEvents(curiusItem, ...events)

  return curiusItem
}

async function createTooltip(item: HTMLElement) {
  if (!item) return

  const parentNode = item.parentNode
  if (!parentNode) return

  const tool = document.createElement("div")
  tool.classList.add("tooltip")
  tool.id = `curius-tooltip-${item.dataset.id ?? "helper"}`
  const content = document.createElement("span")
  Object.assign(content, {
    id: "curius-tooltip-content",
    textContent: item.dataset.tooltip ?? "helper",
  })
  const tip = document.createElement("div")
  tip.id = "arrow"
  tool.append(content, tip)

  if (parentNode.querySelector("#curius-tooltip")) return

  const hide = () => {
    item.style.opacity = "0"
    tool.style.visibility = "hidden"
  }

  function show(this: HTMLElement, { clientX, clientY }: { clientX: number; clientY: number }) {
    async function setPosition(popoverElement: HTMLElement) {
      await computePosition(item, popoverElement, {
        placement: "top",
        middleware: [
          inline({ x: clientX, y: clientY }),
          arrowFloating({ element: tip, padding: 10 }),
          offset(5),
        ],
      }).then(({ x, y, middlewareData }) => {
        Object.assign(popoverElement.style, {
          left: `${x}px`,
          top: `${y}px`,
        })
        const { x: arrowX, y: arrowY } = middlewareData.arrow as Partial<Coords>
        Object.assign(tip.style, {
          left: arrowX != null ? `${arrowX}px` : "",
          top: arrowY != null ? `${arrowY}px` : "",
        })
      })
    }

    item.style.opacity = "1"
    tool.style.visibility = "show"
    setPosition(tool)
  }

  const events = [
    ["mouseleave", hide],
    ["mouseenter", show],
  ] as [keyof HTMLElementEventMap, (this: HTMLElement) => void][]
  for (const [event, listener] of events) {
    item.addEventListener(event, listener)
  }

  parentNode?.appendChild(tool)
  return tool
}

const toggleVisibility = (el: HTMLElement, state: boolean) => {
  Object.assign(el.style, {
    opacity: state ? "1" : "0",
    visibility: state ? "visible" : "hidden",
  })
}

document.addEventListener("nav", async (e: CustomEventMap["nav"]) => {
  const elements = [
    "#curius-search-container",
    "#curius-container",
    "#curius-fetching-text",
    "#curius-fragments",
    ".navigation-container",
    ".curius-outer",
  ].map((id) => document.querySelector(id))
  const searchBar = document.getElementById("curius-bar") as HTMLInputElement | null
  const resultCards = document.getElementsByClassName("curius-search-link")

  if (elements.some((el) => el === null)) return

  const [searchContainer, container, fetchText, fragment, nav, outer] = elements as HTMLElement[]

  fetchText.textContent = "Fetching curius links"
  toggleVisibility(fetchText, true)
  const resp = await fetchLinks()
  toggleVisibility(fetchText, false)

  const userData = resp.user ?? {}
  const linksData = resp.links ?? []
  const sampleLinks = sample(linksData, 20)

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

    refetchIcon.addEventListener("click", async () => {
      refetchIcon.classList.add("disabled")
      refetchIcon.style.opacity = "0.5"

      const notes = document.getElementById("curius-notes") as HTMLDivElement | null
      if (notes) removeAllChildren(notes)

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
    })

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

  // Search functionality
  async function onType(e: HTMLElementEventMap["input"]) {
    let term = (e.target as HTMLInputElement).value
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

    const finalResults = [...allIds].map((id) => linksData[id])
    showLinks(finalResults)
  }

  function createSearchLinks(link: Link): HTMLAnchorElement {
    const curiusLink = document.createElement("a")
    curiusLink.classList.add("curius-search-link")
    curiusLink.target = "_blank"
    curiusLink.href = link.link
    const linkTitle = document.createElement("div")
    linkTitle.classList.add("curius-search-title")
    linkTitle.textContent = link.title
    const linkSnippet = document.createElement("div")
    linkSnippet.classList.add("curius-search-snippet")
    linkSnippet.textContent = link.snippet
    curiusLink.append(linkTitle, linkSnippet)
    curiusLink.onclick = (event) => {
      if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return
      hideLinks()
    }
    return curiusLink
  }

  function shortcutHandler(e: HTMLElementEventMap["keydown"]) {
    if (e.key === "k" && (e.ctrlKey || e.metaKey) && !e.shiftKey) {
      e.preventDefault()
      if (!searchBar) return
      const rect = searchBar.getBoundingClientRect()
      // check if the search bar is in the viewport
      if (rect.top >= 0 && rect.bottom <= window.innerHeight) {
        const searchBarOpen = searchBar.classList.contains("active")
        searchBarOpen ? hideLinks() : showLinks(sampleLinks)
        searchBar?.focus()
      }
    }

    if (!searchBar?.classList.contains("active")) return
    else if (e.key.startsWith("Esc")) {
      e.preventDefault()
      hideLinks()
    } else if (e.key === "Enter") {
      if (searchContainer?.contains(document.activeElement)) {
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
      if (searchContainer?.contains(document.activeElement)) {
        const prevResult = document.activeElement?.previousElementSibling as HTMLInputElement | null
        prevResult?.focus()
      }
    } else if (e.key === "ArrowDown" || e.key === "Tab") {
      e.preventDefault()

      // When first pressing ArrowDown, results wont contain the active element, so focus first element
      if (!searchContainer?.contains(document.activeElement)) {
        const firstResult = resultCards[0] as HTMLInputElement | null
        firstResult?.focus()
      } else {
        // If an element in results-container already has focus, focus next one
        const nextResult = document.activeElement?.nextElementSibling as HTMLInputElement | null
        nextResult?.focus()
      }
    }
  }

  function showLinks(links: Link[]) {
    const notes = document.getElementById("curius-notes") as HTMLDivElement | null
    if (notes) removeAllChildren(notes)
    if (!searchContainer) return
    searchBar?.classList.add("active")
    removeAllChildren(searchContainer)
    searchContainer?.append(...links.map(createSearchLinks))
  }

  function hideLinks() {
    if (searchBar) {
      searchBar.value = ""
    }
    searchBar?.classList.remove("active")
    removeAllChildren(searchContainer)
  }

  function onClick(e: HTMLElementEventMap["click"]) {
    if (searchBar?.classList.contains("active")) return
    showLinks(sampleLinks)
  }

  document.addEventListener("keydown", shortcutHandler)
  window.addCleanup(() => document.removeEventListener("keydown", shortcutHandler))

  const events = [
    ["click", onClick],
    ["input", onType],
  ] as [keyof HTMLElementEventMap, EventListenerOrEventListenerObject][]
  registerEvents(searchBar, ...events)
  registerEscapeHandler(outer, hideLinks)

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
