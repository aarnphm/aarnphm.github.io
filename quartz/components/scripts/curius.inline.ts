import FlexSearch, { IndexOptions } from "flexsearch"
import { pluralize } from "../../util/lang"
import { registerEscapeHandler, removeAllChildren } from "./util"
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

const LinkKeys: Array<keyof Link> = [
  "id",
  "link",
  "title",
  "favorite",
  "snippet",
  "toRead",
  "createdDate",
  "modifiedDate",
  "lastCrawled",
  "topics",
  "highlights",
  "userIds",
]

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

const timeSince = (date: Date | string) => {
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

const formatTimeLeft = (timeLeft: number) => {
  const totalSeconds = Math.ceil(timeLeft / 1000)
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return `${minutes}m ${seconds}s`
}

function random(min: number, max: number): number {
  if (max == null) {
    max = min
    min = 0
  }
  return min + Math.floor(Math.random() * (max - min + 1))
}

function sample(object: any[], n: number): any[] {
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

const localFetchKey = "curiusLinks"
const localTimeKey = "curiusLastFetch"

let index: FlexSearch.Document<Link> | undefined = undefined
const numSearchResults = 20
const fetchTimeout = 2 * 60 * 1000 // 2 minutes
let disableEndTime: number | null = null
let countdownIntervalId: NodeJS.Timeout | null = null
let prevCuriusShortcutHandler: ((e: HTMLElementEventMap["keydown"]) => void) | undefined = undefined

const getLocalItem = (key: "curiusLinks" | "curiusLastFetch", value: any): any =>
  localStorage.getItem(key) ?? value

const fetchLinksHeaders: RequestInit = {
  method: "POST",
  headers: { "Content-Type": "application/json" },
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

const externalLinkRegex = /^(?:https?:\/\/)?(?:www\.)?([^\/]+)/
const extractApexDomain = (url: string) => {
  const match = url.match(externalLinkRegex)
  return match ? match[1] : ""
}

const createLinkEl = (Link: Link): HTMLLIElement => {
  const curiusItem = document.createElement("li")
  curiusItem.id = `curius-item-${Link.id}`
  curiusItem.onmouseenter = () => curiusItem.classList.add("focus")
  curiusItem.onmouseleave = () => curiusItem.classList.remove("focus")

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

      function onMouseEnter(event: MouseEvent) {
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
        modal.style.display = "block"
        modal.classList.add("active")
      }

      function onMouseLeave(event: MouseEvent) {
        curiusItem.classList.add("focus")

        if (!modal) return
        modal.style.display = "none"
        modal.classList.remove("active")
      }

      function onMouseMove(event: MouseEvent) {
        curiusItem.classList.remove("focus")

        if (!modal) return
        modal.classList.add("active")
        modal.style.left = `${event.pageX + 10}px`
        modal.style.top = `${event.pageY + 10}px`
      }

      const events = [
        ["mouseenter", onMouseEnter],
        ["mouseleave", onMouseLeave],
        ["mousemove", onMouseMove],
      ] as [keyof HTMLElementEventMap, (this: HTMLElement) => void][]
      events.forEach(([event, listener]) => {
        highlights.removeEventListener(event, listener)
        highlights.addEventListener(event, listener)
      })
    }

    item.append(tags, misc)
    return item
  }

  curiusItem.append(
    ...([createTitle, createMetadata] as ((Link: Link) => Node)[]).map((fn) => fn(Link)),
  )

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
    tool.style.display = "none"
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
    tool.style.display = "block"
    setPosition(tool)
  }

  const caller = [
    ["mouseleave", hide],
    ["mouseenter", show],
  ] as [keyof HTMLElementEventMap, (this: HTMLElement) => void][]
  caller.forEach(([event, listener]) => {
    item.removeEventListener(event, listener)
    item.addEventListener(event, listener)
  })

  parentNode?.appendChild(tool)
  return tool
}

const currentVisibility = (el: HTMLElement) => localStorage.getItem(el.id) ?? "true"

const toggleVisibility = (el: HTMLElement, override?: boolean) => {
  let visible = currentVisibility(el)
  const isVisibile = override !== undefined ? override : visible === "true"
  if (override !== undefined) {
    visible = override ? "true" : "false"
  } else {
    visible = visible === "true" ? "false" : "true"
  }
  Object.assign(el.style, {
    opacity: isVisibile ? "1" : "0",
    display: isVisibile ? "block" : "none",
  })
  localStorage.setItem(el.id, visible)
}

let prevGatedShortcutHandler: ((e: HTMLElementEventMap["keydown"]) => void) | undefined = undefined
document.addEventListener("nav", async (e: CustomEventMap["nav"]) => {
  const elements = [
    "#curius-search-container",
    "#curius-container",
    "#curius-fetching-text",
    "#curius-fragments",
    ".navigation-container",
  ].map((id) => document.querySelector(id))
  const searchBar = document.getElementById("curius-bar") as HTMLInputElement | null
  const resultCards = document.getElementsByClassName("curius-search-link")

  if (elements.some((el) => el === null)) return

  const [searchContainer, container, fetchText, fragment, nav] = elements as HTMLElement[]

  fetchText.textContent = "Fetching curius links"
  toggleVisibility(fetchText)
  const resp = await fetchLinks()
  toggleVisibility(fetchText)

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
    const popover = (await createTooltip(refetchIcon)) as HTMLElement

    function updatePopoverText(popover: HTMLElement, endTime: number) {
      const timeLeft = Math.max(endTime - Date.now(), 0)
      const span = popover.querySelector("#curius-tooltip-content") as HTMLSpanElement
      span.textContent = timeLeft <= 0 ? refetchContent : `Wait time: ${formatTimeLeft(timeLeft)}`
    }

    const gatedRefresh = async (e: HTMLElementEventMap["keydown"]) => {
      if ((e.key === "R" || e.key === "r") && (e.ctrlKey || e.metaKey)) {
        e.preventDefault()
        e.stopPropagation()
      }
    }

    if (refetchIcon.classList.contains("disabled")) {
      if (prevGatedShortcutHandler) {
        refetchIcon.removeEventListener("keydown", prevGatedShortcutHandler)
      }
      refetchIcon.addEventListener("keydown", gatedRefresh)
      prevGatedShortcutHandler = gatedRefresh
    } else {
      refetchIcon.removeEventListener("keydown", gatedRefresh)
    }

    refetchIcon.addEventListener("click", async () => {
      if (refetchIcon.classList.contains("disabled")) return
      refetchIcon.classList.add("disabled")
      refetchIcon.style.opacity = "0.5"

      // Set the disable end time
      disableEndTime = Date.now() + fetchTimeout // fetchTimeout should be defined somewhere in your code

      // Update the tooltip immediately
      updatePopoverText(popover as HTMLElement, disableEndTime)

      // Set an interval to update the tooltip every second
      countdownIntervalId = setInterval(() => {
        updatePopoverText(popover as HTMLElement, disableEndTime!)
      }, 1000)

      // Remove disabled state after the timeout
      setTimeout(() => {
        refetchIcon.classList.remove("disabled")
        refetchIcon.style.opacity = ""
        disableEndTime = null
        clearInterval(countdownIntervalId!)
        countdownIntervalId = null
      }, fetchTimeout)

      removeAllChildren(fragment)
      toggleVisibility(nav, false)

      toggleVisibility(fetchText)
      fetchText.textContent = "Refreshing curius links"
      const refetched = await fetchLinks(true)
      toggleVisibility(fetchText)

      const newData = refetched.links ?? []
      if (newData.length === 0) {
        container.innerHTML = `<p>Failed to fetch links.</p>`
        return
      }
      fragment.append(...newData.map(createLinkEl))
      toggleVisibility(nav, true)
    })

    const events = [
      [
        "mouseenter",
        () => {
          if (refetchIcon.classList.contains("disabled")) {
            refetchIcon.style.opacity = "0.5"
            updatePopoverText(popover, disableEndTime!)
            if (countdownIntervalId === null) {
              countdownIntervalId = setInterval(() => {
                updatePopoverText(popover, disableEndTime!)
              }, 1000)
            }
          }
        },
      ],
      [
        "mouseleave",
        () => {
          // Clear the countdown when the mouse leaves
          const popover = document.getElementById("curius-tooltip")
          if (countdownIntervalId !== null) {
            clearInterval(countdownIntervalId)
            countdownIntervalId = null
          }

          if (refetchIcon.classList.contains("disabled")) {
            refetchIcon.style.opacity = "0.5"
          }
        },
      ],
    ] as [keyof HTMLElementEventMap, (this: HTMLElement) => void][]
    events.forEach(([event, listener]) => {
      refetchIcon.removeEventListener(event, listener)
      refetchIcon.addEventListener(event, listener)
    })
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

  if (prevCuriusShortcutHandler) {
    document.removeEventListener("keydown", prevCuriusShortcutHandler)
  }

  document.addEventListener("keydown", shortcutHandler)
  prevCuriusShortcutHandler = shortcutHandler
  searchBar?.removeEventListener("click", onClick)
  searchBar?.addEventListener("click", onClick)
  searchBar?.removeEventListener("input", onType)
  searchBar?.addEventListener("input", onType)

  if (!index) {
    const documentIndex = [
      ...LinkKeys.map(
        (key) =>
          ({ field: key, tokenize: "forward" }) as IndexOptions<Link, false> & { field: string },
      ),
    ]
    index = new FlexSearch.Document({
      charset: "latin:extra",
      document: { id: "id", index: documentIndex },
    })

    let id = 0
    for (const link of linksData) {
      await index.addAsync(id, { ...link })
      id++
    }
  }

  for (const el of [
    document.getElementById("quartz-body"),
    document.getElementsByClassName("center")[0],
  ] as (HTMLElement | null)[]) {
    registerEscapeHandler(el, hideLinks)
  }
})
