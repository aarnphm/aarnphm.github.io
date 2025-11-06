import { Link, CuriusResponse, Trail, TrailInfo, Following } from "../types"
import { registerEscapeHandler, registerEvents, removeAllChildren } from "./util"
import { ValidLocale, i18n } from "../../i18n"
import FlexSearch, { Id } from "flexsearch"
import { LCG } from "../../util/helpers"
import { encode } from "./util"

const CURIUS_HOST = "https://curius.app"
export const CURIUS = `${CURIUS_HOST}/aaron-pham`
const externalLinkRegex = /^(?:https?:\/\/)?(?:www\.)?([^\/]+)/
export const PINNED_FOLLOWING_IDS = [3971, 1296] as const

export const fetchLinksHeaders: RequestInit = {
  method: "GET",
  headers: { "Content-Type": "application/json" },
}

export const _SENTINEL: Link = {
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

const iconMapping = {
  default: `<svg width="11" height="11" fill="none" viewBox="0 0 46 46" preserveAspectRatio="none"><path stroke="#000" stroke-width="4" d="M2.828 22.627L22.627 2.828l19.799 19.8-19.8 19.798z"></path><path fill="#000" d="M17 22.657L22.657 17l5.657 5.657-5.657 5.657z"></path></svg>`,
  youtube: `<svg width="11" height="11" fill="none" viewBox="0 0 46 46" preserveAspectRatio="none"><path d="M32.076 24.233L4.998 39.816C3.664 40.584 2 39.621 2 38.084V6.917c0-1.538 1.664-2.5 2.998-1.734l27.078 15.584c1.337.769 1.337 2.697 0 3.466z" stroke="#000" stroke-width="4"></path></svg>`,
  github: `<svg viewBox="64 64 896 896" focusable="false" data-icon="github" width="1em" height="1em" fill="currentColor" aria-hidden="true"><path d="M511.6 76.3C264.3 76.2 64 276.4 64 523.5 64 718.9 189.3 885 363.8 946c23.5 5.9 19.9-10.8 19.9-22.2v-77.5c-135.7 15.9-141.2-73.9-150.3-88.9C215 726 171.5 718 184.5 703c30.9-15.9 62.4 4 98.9 57.9 26.4 39.1 77.9 32.5 104 26 5.7-23.5 17.9-44.5 34.7-60.8-140.6-25.2-199.2-111-199.2-213 0-49.5 16.3-95 48.3-131.7-20.4-60.5 1.9-112.3 4.9-120 58.1-5.2 118.5 41.6 123.2 45.3 33-8.9 70.7-13.6 112.9-13.6 42.4 0 80.2 4.9 113.5 13.9 11.3-8.6 67.3-48.8 121.3-43.9 2.9 7.7 24.7 58.3 5.5 118 32.4 36.8 48.9 82.7 48.9 132.3 0 102.2-59 188.1-200 212.9a127.5 127.5 0 0138.1 91v112.5c.8 9 0 17.9 15 17.9 177.1-59.7 304.6-227 304.6-424.1 0-247.2-200.4-447.3-447.5-447.3z"></path></svg>`,
  twitter: `<svg viewBox="64 64 896 896" focusable="false" data-icon="twitter" width="1em" height="1em" fill="currentColor" aria-hidden="true"><path d="M928 254.3c-30.6 13.2-63.9 22.7-98.2 26.4a170.1 170.1 0 0075-94 336.64 336.64 0 01-108.2 41.2A170.1 170.1 0 00672 174c-94.5 0-170.5 76.6-170.5 170.6 0 13.2 1.6 26.4 4.2 39.1-141.5-7.4-267.7-75-351.6-178.5a169.32 169.32 0 00-23.2 86.1c0 59.2 30.1 111.4 76 142.1a172 172 0 01-77.1-21.7v2.1c0 82.9 58.6 151.6 136.7 167.4a180.6 180.6 0 01-44.9 5.8c-11.1 0-21.6-1.1-32.2-2.6C211 652 273.9 701.1 348.8 702.7c-58.6 45.9-132 72.9-211.7 72.9-14.3 0-27.5-.5-41.2-2.1C171.5 822 261.2 850 357.8 850 671.4 850 843 590.2 843 364.7c0-7.4 0-14.8-.5-22.2 33.2-24.3 62.3-54.4 85.5-88.2z"></path></svg>`,
  arxiv: `<svg width="10" height="10" fill="none" viewBox="0 0 45 45" data-icon="arxiv" preserveAspectRatio="none"><path d="M25.468 7.836C24.993 3.384 24.4 0 20.244 0c-5.64 0-9.499 4.037-9.499 5.7 0 .415.357.415.653.415.654 0 3.266-.772 4.334-2.79l.119-.06c3.265 0 4.037 1.84 4.63 7.955l.832 8.727c-2.137 1.188-10.152 5.996-14.426 8.965C.119 33.72 0 35.502 0 37.223c0 2.018 1.069 3.325 2.85 3.325 1.78 0 4.511-1.96 4.511-2.85 0-.297-.178-.356-.415-.416-.475-.059-1.9-.296-1.9-2.612 0-1.603 0-2.315 6.293-6.53 3.621-2.375 7.065-4.275 10.152-6.234 0 .12.89 9.024 1.009 10.033.475 4.69.89 8.608 5.284 8.608 5.64 0 9.498-4.036 9.498-5.699 0-.415-.296-.415-.653-.415-.593 0-3.206.772-4.334 2.79-.178.06-.474.06-.593.06-3.384 0-3.74-3.74-4.215-8.728-.178-1.9-.356-3.443-.95-9.617C45 7.896 45 5.818 45 3.325 45 1.306 43.931 0 42.21 0c-1.84 0-4.572 2.018-4.572 2.85 0 .356.357.415.416.415.475.06 1.9.297 1.9 2.612 0 1.484-.178 2.375-5.224 5.818-1.781 1.188-4.275 2.731-8.371 5.224l-.89-9.083z" fill="#000"></path></svg>`,
}

type iconKeys = keyof typeof iconMapping

const getIconSvg = (name: iconKeys) => iconMapping[name] ?? null

function extractApexDomain(url: string) {
  const match = url.match(externalLinkRegex)
  return match ? match[1] : ""
}

let currentActive: HTMLLIElement | null = null

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
    note.classList.remove("active")
    parent.classList.remove("active")
  }

  close?.addEventListener("click", cleanUp)
  window.addCleanup(() => close?.removeEventListener("click", cleanUp))
  registerEscapeHandler(note, cleanUp)

  removeAllChildren(snippetNode)
  snippetNode.textContent = Link.metadata ? Link.metadata.full_text : Link.snippet

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

export function createLinkEl(Link: Link): HTMLLIElement {
  const curiusItem = document.createElement("li")
  curiusItem.id = `curius-item-${Link.id}`
  curiusItem.classList.add("curius-item")

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
    const modifiedDate = new Date(Link.modifiedDate as string)
    time.innerHTML = `<time datetime=${
      Link.modifiedDate
    } title="${modifiedDate.toUTCString()}">${timeSince(Link.createdDate as string)}</time>`
    misc.appendChild(time)

    if (Link.highlights.length > 0) {
      const highlights = document.createElement("div")
      highlights.id = `curius-highlights-${Link.id}`
      highlights.innerHTML = `${Link.highlights.length} ${Link.highlights.length > 0 ? "highlights" : "highlight"}`
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
        modal.style.left = `${pageX + 10}px`
        modal.style.top = `${pageY + 10}px`
      }

      registerEvents(
        highlights,
        ["mouseenter", onMouseEnter],
        ["mouseleave", onMouseLeave],
        ["mousemove", onMouseMove],
      )
    }

    item.append(tags, misc)
    return item
  }

  curiusItem.append(createTitle({ Link, addFaIcon: true }), createMetadata(Link))
  curiusItem.dataset.items = JSON.stringify(true)

  const onClick = (e: HTMLElementEventMap["click"]) => {
    const note = document.getElementsByClassName("curius-notes")[0] as HTMLDivElement | null

    if (e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return
    if (currentActive) currentActive.classList.remove("active")
    if (note) note.classList.remove("active")

    currentActive = curiusItem
    currentActive.classList.add("active")

    if (Link.highlights.length > 0) {
      if (!note) return
      note.classList.add("active")
      updateNotePanel(Link, note, currentActive)
    }

    if (e.target instanceof HTMLAnchorElement || note?.classList.contains("active")) return
    window.open(Link.link, "_blank")
  }

  registerEscapeHandler(curiusItem, () => curiusItem.classList.remove("active"))

  const onMouseEnter = () => {
    const favoriteDiv = curiusItem.querySelector("svg.favorite-icon") as HTMLDivElement | null

    if (favoriteDiv) favoriteDiv.classList.add("focus")
    curiusItem.classList.add("focus")
  }

  const onMouseLeave = () => {
    const favoriteDiv = curiusItem.querySelector("svg.favorite-icon") as HTMLDivElement | null

    if (favoriteDiv) favoriteDiv.classList.remove("focus")
    curiusItem.classList.remove("focus")
  }

  registerEvents(
    curiusItem,
    //@ts-ignore
    ["click", onClick],
    ["mouseenter", onMouseEnter],
    ["mouseleave", onMouseLeave],
  )

  return curiusItem
}

export function timeSince(date: Date | string) {
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

interface Title {
  Link: Link
  elementType?: "div" | "li"
  addFaIcon?: boolean
}

const defaultTitle: Title = {
  Link: _SENTINEL,
  elementType: "div",
  addFaIcon: false,
}

let p: DOMParser
export const createTitle = (userOpts: Title): HTMLDivElement | HTMLLIElement => {
  const { Link, elementType, addFaIcon } = { ...defaultTitle, ...userOpts }

  if (elementType === undefined) throw new Error("Element type is undefined")

  p = p || new DOMParser()
  const item = document.createElement(elementType)
  item.classList.add("curius-item-title")

  const header = document.createElement("div")
  header.classList.add("curius-item-link")

  const link = document.createElement("a")
  Object.assign(link, {
    href: Link.link,
    target: "_blank",
    rel: "noopener noreferrer",
    innerHTML: Link.title,
  })
  header.appendChild(link)

  const address = document.createElement("div")
  address.classList.add("curius-item-address")
  address.textContent = extractApexDomain(Link.link)

  if (addFaIcon) {
    const faIcon = document.createElement("div")
    faIcon.classList.add("curius-item-fa")
    switch (true) {
      case /twitter/.test(Link.link):
        faIcon.innerHTML = getIconSvg("twitter")
        break
      case /youtube/.test(Link.link):
        faIcon.innerHTML = getIconSvg("youtube")
        break
      case /github/.test(Link.link):
        faIcon.innerHTML = getIconSvg("github")
        break
      case /arxiv/.test(Link.link):
        faIcon.innerHTML = getIconSvg("arxiv")
        break
      default:
        faIcon.innerHTML = getIconSvg("default")
        break
    }
    item.appendChild(faIcon)
  }

  item.append(header, address)
  const itemIcon = document.createElement("div")
  itemIcon.classList.add("curius-item-icons")

  if (Link.favorite) {
    const icon = document.createElementNS("http://www.w3.org/2000/svg", "svg")
    icon.setAttribute("fill", "currentColor")
    icon.setAttribute("preserveAspectRatio", "xMidYMid meet")
    icon.setAttribute("height", "1rem")
    icon.setAttribute("width", "1rem")
    icon.setAttribute("viewBox", "0 0 40 40")
    icon.setAttribute("class", "favorite-icon")
    icon.style.verticalAlign = "unset"

    const g = document.createElementNS("http://www.w3.org/2000/svg", "g")
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path")
    path.setAttribute("stroke-width", "5")
    path.setAttribute("fill", "rgb(236, 180, 19)")
    path.setAttribute(
      "d",
      "m5.2 18.8l6 5.5-1.7 7.7c-0.2 1 0.2 2 1 2.5 0.3 0.3 0.8 0.5 1.3 0.5 0.4 0 0.7 0 1-0.2 0 0 0.2 0 0.2-0.1l6.8-3.9 6.9 3.9s0.1 0 0.1 0.1c0.9 0.4 1.9 0.4 2.5-0.1 0.9-0.5 1.2-1.5 1-2.5l-1.6-7.7c0.6-0.5 1.6-1.5 2.6-2.5l3.2-2.8 0.2-0.2c0.6-0.7 0.8-1.7 0.5-2.5s-1-1.5-2-1.7h-0.2l-7.8-0.8-3.2-7.2s0-0.1-0.2-0.1c-0.1-1.2-1-1.7-1.8-1.7s-1.7 0.5-2.2 1.3c0 0 0 0.2-0.1 0.2l-3.2 7.2-7.8 0.8h-0.2c-0.8 0.2-1.7 0.8-2 1.7-0.2 1 0 2 0.7 2.6z",
    )

    g.appendChild(path)
    icon.appendChild(g)
    itemIcon.appendChild(icon)
  }

  item.appendChild(itemIcon)

  return item
}

export async function fetchFollowing() {
  try {
    const resp = await fetch("/api/curius?query=following", fetchLinksHeaders)
    if (!resp.ok) {
      throw new Error("Failed to get followings from curius")
    }
    const data: CuriusResponse = await resp.json()
    if (data === undefined || data.following === undefined) {
      throw new Error("No following data")
    }
    const prioritizedSet = new Set<number>(PINNED_FOLLOWING_IDS)
    const prioritizedBuckets = new Map<number, Following[]>()
    PINNED_FOLLOWING_IDS.forEach((id) => prioritizedBuckets.set(id, []))
    const remaining: Following[] = []

    data.following.forEach((entry) => {
      const userId = entry.user.id
      if (prioritizedSet.has(userId)) {
        prioritizedBuckets.get(userId)!.push(entry)
      } else {
        remaining.push(entry)
      }
    })

    const prioritized: Following[] = []
    PINNED_FOLLOWING_IDS.forEach((id) => {
      const entries = prioritizedBuckets.get(id)
      if (entries && entries.length > 0) {
        prioritized.push(...entries)
      }
    })

    return [...prioritized, ...remaining]
  } catch (err) {
    console.error(err)
  }
}

export async function fetchUsers() {
  try {
    const resp = await fetch("/api/curius?query=user", fetchLinksHeaders)
    if (!resp.ok) {
      throw new Error("Failed to get users from curius")
    }
    const data: CuriusResponse = await resp.json()
    if (data === undefined || data.user === undefined) {
      throw new Error("Failed to fetch user")
    }
    return data.user
  } catch (error) {
    console.error(error)
  }
}

async function fetchLinks(): Promise<CuriusResponse | undefined> {
  try {
    const resp = await fetch("/api/curius?query=links", fetchLinksHeaders)
    if (!resp.ok) {
      throw new Error("Failed to get links from curius")
    }
    const data: CuriusResponse = await resp.json()
    if (data === undefined || data.links === undefined) {
      throw new Error("Failed to fetch links")
    }
    return data
  } catch (error) {
    console.error(error)
  }
}

export async function fetchCuriusLinks(): Promise<CuriusResponse> {
  const [user, following, linkResp, trails] = await Promise.all([
    fetchUsers(),
    fetchFollowing(),
    fetchLinks(),
    fetchTrails(),
  ])

  const links = linkResp?.links ?? []
  const hasMore = typeof linkResp?.hasMore === "boolean" ? linkResp.hasMore : links.length > 0
  const page = linkResp?.page ?? 0

  return { links, user, following, trails, hasMore, page }
}

async function fetchTrailPages(trail: Trail): Promise<Link[]> {
  const collected: Link[] = []
  const seenIds = new Set<number>()
  let page = 0
  let hasMore = true

  while (hasMore) {
    try {
      const resp = await fetch(
        `/api/curius?query=trails&name=${encodeURIComponent(trail.hash)}&page=${page}`,
        fetchLinksHeaders,
      )
      if (!resp.ok) {
        throw new Error(`Failed to get trail ${trail.trailName} page ${page}`)
      }

      const data: CuriusResponse = await resp.json()
      const pageLinks = data.links ?? []

      pageLinks.forEach((rawLink) => {
        const linkId = typeof rawLink.id === "number" ? rawLink.id : NaN
        if (Number.isNaN(linkId) || seenIds.has(linkId)) return
        seenIds.add(linkId)

        const normalized: Link = {
          ...rawLink,
          trails: rawLink.trails && rawLink.trails.length > 0 ? rawLink.trails : [trail],
        }

        collected.push(normalized)
      })

      hasMore = data.hasMore ?? false
      page += 1

      if (!hasMore) break
    } catch (error) {
      console.error(`Failed to fetch trail ${trail.trailName} page ${page}`, error)
      break
    }
  }

  return collected
}

export async function createTrailMetadata(res: CuriusResponse): Promise<Map<string, TrailInfo>> {
  const trailMetadata: Map<string, TrailInfo> = new Map()
  const trails = res.trails ?? []

  if (trails.length > 0) {
    const trailCollections = await Promise.all(
      trails.map(async (trail) => ({ trail, links: await fetchTrailPages(trail) })),
    )

    trailCollections.forEach(({ trail, links }) => {
      if (links.length === 0) return
      const trailName = trail.trailName

      if (!trailMetadata.has(trailName)) {
        trailMetadata.set(trailName, { trail, links: new Map() })
      }

      const trailBucket = trailMetadata.get(trailName)
      if (!trailBucket) return

      links.forEach((link) => {
        const linkId = typeof link.id === "number" ? link.id : NaN
        if (Number.isNaN(linkId)) return
        if (!trailBucket.links.has(linkId)) {
          trailBucket.links.set(linkId, link)
        }
      })
    })
  }

  if (trailMetadata.size === 0) {
    const links = res.links ?? []
    links
      .filter((link) => link.trails.length > 0)
      .forEach((link) => {
        link.trails.forEach((trail) => {
          if (!trailMetadata.has(trail.trailName)) {
            trailMetadata.set(trail.trailName, { trail, links: new Map() })
          }
          const trailBucket = trailMetadata.get(trail.trailName)
          if (!trailBucket) return

          const linkId = link.id
          if (typeof linkId === "number") {
            trailBucket.links.set(linkId, link)
          }
        })
      })
  }

  return trailMetadata
}

export const createTrailList = (trails: Map<string, TrailInfo>) => {
  const trail = document.getElementById("trail-list") as HTMLUListElement | null
  const total = document.getElementsByClassName("curius-trail")[0] as HTMLDivElement | null
  if (!trail || !total) return

  const limits = parseInt(total.dataset.limits!) ?? 5
  const numTrails = parseInt(total.dataset.numTrails!) ?? 4
  const locale = total.dataset.locale! as ValidLocale

  removeAllChildren(trail)
  for (const [trail_name, { trail: Trail, links: linksMap }] of Array.from(trails.entries()).slice(
    0,
    numTrails,
  )) {
    const links = Array.from(linksMap.values())
    const remaining = Math.max(0, links.length - limits)
    trail.appendChild(createTrailEl(trail_name, links.slice(0, limits), Trail, remaining, locale))
  }
}

function createTrailEl(
  trail_name: string,
  trails: Link[],
  info: Trail,
  remaining: number,
  locale: ValidLocale,
): HTMLLIElement {
  const container = document.createElement("li")
  container.classList.add("trails-li")

  const headers = document.createElement("div")
  headers.classList.add("curius-trail-header")
  headers.innerHTML = `<span class="trail-title"><em>${trail_name}</em></span><span class="trail-description">${info.description!}</span>`

  const trailLink = `${CURIUS_HOST}/trail/${info.slug}`

  const links = document.createElement("ul")
  links.classList.add("trail-ul")
  links.append(
    ...trails.map((link) => {
      const el = createTitle({ Link: link, elementType: "li" })

      const onMouseEnter = () => {
        const favoriteDiv = el.querySelector("svg.favorite-icon") as HTMLDivElement | null

        if (favoriteDiv) favoriteDiv.classList.add("focus")
        el.classList.add("focus")
      }

      const onMouseLeave = () => {
        const favoriteDiv = el.querySelector("svg.favorite-icon") as HTMLDivElement | null

        if (favoriteDiv) favoriteDiv.classList.remove("focus")
        el.classList.remove("focus")
      }

      const openLink = (evt: HTMLElementEventMap["click"]) => {
        if (evt.target instanceof HTMLAnchorElement) return
        window.open(trailLink, "_blank")
      }

      registerEvents(
        el,
        ["mouseenter", onMouseEnter],
        ["mouseleave", onMouseLeave],
        //@ts-ignore
        ["click", openLink],
      )

      return el
    }),
  )

  const seeMore = document.createElement("div")
  seeMore.classList.add("see-more")
  seeMore.innerHTML = `<span><a href=${trailLink} target="_blank">${remaining > 0 ? i18n(locale).components.recentNotes.seeRemainingMore({ remaining }) : "Void de plus →"}</a></span>`

  container.append(headers, links, seeMore)

  return container
}

let index = new FlexSearch.Document({
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
        field: "link",
        tokenize: "forward",
      },
      {
        field: "snippet",
        tokenize: "forward",
      },
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

  return tokens.sort((a, b) => b.length - a.length)
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

// Use current date as seed to get same equation for whole day
const now = new Date()
const seed = now.getFullYear() * 10000 + (now.getMonth() + 1) * 100 + now.getDate()
const rng = new LCG(seed)

export async function curiusSearch(searchData: Link[]) {
  const sampleLinks = rng.shuffle(searchData).splice(0, 20)

  const bar = document.getElementById("curius-bar") as HTMLInputElement | null
  const container = document.getElementById("curius-search-container") as HTMLDivElement | null

  async function onType(e: HTMLElementEventMap["input"]) {
    let term = (e.target as HTMLInputElement).value
    container?.classList.toggle("active", term !== "")
    let searchResults =
      (await index?.searchAsync({
        query: term,
        limit: numSearchResults,
        index: ["title", "snippet", "link"],
      })) ?? []

    const getByField = (field: string): number[] => {
      const results = searchResults.filter((x) => x.field === field)
      return results.length === 0 ? [] : ([...results[0].result] as number[])
    }

    const allIds: Set<number> = new Set([
      ...getByField("title"),
      ...getByField("snippet"),
      ...getByField("link"),
    ])

    const finalResults = [...allIds].map((id) => formatLinks(term, id))
    displayLinks(finalResults)
  }

  const formatLinks = (term: string, id: number): Link => {
    const L = searchData[id]
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

  function onClick() {
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

  await fillIndex(searchData)
}

export async function fetchSearchLinks() {
  try {
    const resp = await fetch("/api/curius?query=searchLinks", fetchLinksHeaders)
    if (!resp.ok) {
      throw new Error("Failed to get searchLinks from curius")
    }
    const data: CuriusResponse = await resp.json()
    if (data === undefined || data.links === undefined) {
      throw new Error("No searchLinks data")
    }
    return data.links
  } catch (err) {
    console.error(err)
    return []
  }
}

export async function fetchTrails() {
  try {
    const resp = await fetch("/api/curius?query=trails", fetchLinksHeaders)
    if (!resp.ok) {
      throw new Error("Failed to get trails from curius")
    }
    const data: CuriusResponse = await resp.json()
    if (data === undefined || data.trails === undefined) {
      throw new Error("No trails data")
    }
    return data.trails
  } catch (err) {
    console.error(err)
    return []
  }
}

let indexPopulated = false
async function fillIndex(links: Link[]) {
  if (indexPopulated) return
  let id: Id = 0
  const promises = []
  for (const link of links) {
    //@ts-ignore
    promises.push(index.addAsync(id, { ...link }))
    id++
  }

  await Promise.all(promises)
  indexPopulated = true
}
