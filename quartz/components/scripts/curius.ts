import { Link, User, CuriusResponse, Trail } from "../types"
import { registerMouseHover, removeAllChildren } from "./util"
import { joinSegments } from "../../util/path"
import { ValidLocale, i18n } from "../../i18n"

const curiusBase = "https://curius.app"
export const CURIUS = joinSegments(curiusBase, "aaron-pham")
const externalLinkRegex = /^(?:https?:\/\/)?(?:www\.)?([^\/]+)/
const localFetchKey = "curiusLinks"
const localTimeKey = "curiusLastFetch"

const fetchLinksHeaders: RequestInit = {
  method: "POST",
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
  favourite: `<path d="m5.2 18.8l6 5.5-1.7 7.7c-0.2 1 0.2 2 1 2.5 0.3 0.3 0.8 0.5 1.3 0.5 0.4 0 0.7 0 1-0.2 0 0 0.2 0 0.2-0.1l6.8-3.9 6.9 3.9s0.1 0 0.1 0.1c0.9 0.4 1.9 0.4 2.5-0.1 0.9-0.5 1.2-1.5 1-2.5l-1.6-7.7c0.6-0.5 1.6-1.5 2.6-2.5l3.2-2.8 0.2-0.2c0.6-0.7 0.8-1.7 0.5-2.5s-1-1.5-2-1.7h-0.2l-7.8-0.8-3.2-7.2s0-0.1-0.2-0.1c-0.1-1.2-1-1.7-1.8-1.7s-1.7 0.5-2.2 1.3c0 0 0 0.2-0.1 0.2l-3.2 7.2-7.8 0.8h-0.2c-0.8 0.2-1.7 0.8-2 1.7-0.2 1 0 2 0.7 2.6z"></path>`,
  default: `<svg width="11" height="11" fill="none" viewBox="0 0 46 46" preserveAspectRatio="none"><path stroke="#000" stroke-width="4" d="M2.828 22.627L22.627 2.828l19.799 19.8-19.8 19.798z"></path><path fill="#000" d="M17 22.657L22.657 17l5.657 5.657-5.657 5.657z"></path></svg>`,
  youtube: `<svg width="11" height="11" fill="none" viewBox="0 0 46 46" preserveAspectRatio="none"><path d="M32.076 24.233L4.998 39.816C3.664 40.584 2 39.621 2 38.084V6.917c0-1.538 1.664-2.5 2.998-1.734l27.078 15.584c1.337.769 1.337 2.697 0 3.466z" stroke="#000" stroke-width="4"></path></svg>`,
  github: `<svg viewBox="64 64 896 896" focusable="false" data-icon="github" width="1em" height="1em" fill="currentColor" aria-hidden="true"><path d="M511.6 76.3C264.3 76.2 64 276.4 64 523.5 64 718.9 189.3 885 363.8 946c23.5 5.9 19.9-10.8 19.9-22.2v-77.5c-135.7 15.9-141.2-73.9-150.3-88.9C215 726 171.5 718 184.5 703c30.9-15.9 62.4 4 98.9 57.9 26.4 39.1 77.9 32.5 104 26 5.7-23.5 17.9-44.5 34.7-60.8-140.6-25.2-199.2-111-199.2-213 0-49.5 16.3-95 48.3-131.7-20.4-60.5 1.9-112.3 4.9-120 58.1-5.2 118.5 41.6 123.2 45.3 33-8.9 70.7-13.6 112.9-13.6 42.4 0 80.2 4.9 113.5 13.9 11.3-8.6 67.3-48.8 121.3-43.9 2.9 7.7 24.7 58.3 5.5 118 32.4 36.8 48.9 82.7 48.9 132.3 0 102.2-59 188.1-200 212.9a127.5 127.5 0 0138.1 91v112.5c.8 9 0 17.9 15 17.9 177.1-59.7 304.6-227 304.6-424.1 0-247.2-200.4-447.3-447.5-447.3z"></path></svg>`,
  twitter: `<svg viewBox="64 64 896 896" focusable="false" data-icon="twitter" width="1em" height="1em" fill="currentColor" aria-hidden="true"><path d="M928 254.3c-30.6 13.2-63.9 22.7-98.2 26.4a170.1 170.1 0 0075-94 336.64 336.64 0 01-108.2 41.2A170.1 170.1 0 00672 174c-94.5 0-170.5 76.6-170.5 170.6 0 13.2 1.6 26.4 4.2 39.1-141.5-7.4-267.7-75-351.6-178.5a169.32 169.32 0 00-23.2 86.1c0 59.2 30.1 111.4 76 142.1a172 172 0 01-77.1-21.7v2.1c0 82.9 58.6 151.6 136.7 167.4a180.6 180.6 0 01-44.9 5.8c-11.1 0-21.6-1.1-32.2-2.6C211 652 273.9 701.1 348.8 702.7c-58.6 45.9-132 72.9-211.7 72.9-14.3 0-27.5-.5-41.2-2.1C171.5 822 261.2 850 357.8 850 671.4 850 843 590.2 843 364.7c0-7.4 0-14.8-.5-22.2 33.2-24.3 62.3-54.4 85.5-88.2z"></path></svg>`,
  arxiv: `<svg width="10" height="10" fill="none" viewBox="0 0 45 45" preserveAspectRatio="none"><path d="M25.468 7.836C24.993 3.384 24.4 0 20.244 0c-5.64 0-9.499 4.037-9.499 5.7 0 .415.357.415.653.415.654 0 3.266-.772 4.334-2.79l.119-.06c3.265 0 4.037 1.84 4.63 7.955l.832 8.727c-2.137 1.188-10.152 5.996-14.426 8.965C.119 33.72 0 35.502 0 37.223c0 2.018 1.069 3.325 2.85 3.325 1.78 0 4.511-1.96 4.511-2.85 0-.297-.178-.356-.415-.416-.475-.059-1.9-.296-1.9-2.612 0-1.603 0-2.315 6.293-6.53 3.621-2.375 7.065-4.275 10.152-6.234 0 .12.89 9.024 1.009 10.033.475 4.69.89 8.608 5.284 8.608 5.64 0 9.498-4.036 9.498-5.699 0-.415-.296-.415-.653-.415-.593 0-3.206.772-4.334 2.79-.178.06-.474.06-.593.06-3.384 0-3.74-3.74-4.215-8.728-.178-1.9-.356-3.443-.95-9.617C45 7.896 45 5.818 45 3.325 45 1.306 43.931 0 42.21 0c-1.84 0-4.572 2.018-4.572 2.85 0 .356.357.415.416.415.475.06 1.9.297 1.9 2.612 0 1.484-.178 2.375-5.224 5.818-1.781 1.188-4.275 2.731-8.371 5.224l-.89-9.083z" fill="#000"></path></svg>`,
}

type iconKeys = keyof typeof iconMapping

const getIconSvg = (name: iconKeys) => iconMapping[name] ?? null

function extractApexDomain(url: string) {
  const match = url.match(externalLinkRegex)
  return match ? match[1] : ""
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

function createFavouriteIcon() {
  const svg = document.createElement("svg")
  svg.setAttribute("fill", "currentColor")
  svg.setAttribute("preserveAspectRatio", "xMidYMid meet")
  svg.setAttribute("height", "1em")
  svg.setAttribute("width", "1em")
  svg.setAttribute("viewBox", "0 0 40 40")
  svg.style.verticalAlign = "unset"

  const g = document.createElement("g")
  g.innerHTML = getIconSvg("favourite")!
  svg.appendChild(g)

  return svg
}

export const createTitle = (userOpts: Title): HTMLDivElement | HTMLLIElement => {
  const { Link, elementType, addFaIcon } = { ...defaultTitle, ...userOpts }

  if (elementType === undefined) throw new Error("Element type is undefined")

  const item = document.createElement(elementType)
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
      default:
        faIcon.innerHTML = getIconSvg("default")
        break
    }
    item.appendChild(faIcon)
  }

  item.append(header, address)

  if (Link.favorite) {
    const itemIcon = document.createElement("div")
    itemIcon.classList.add("curius-item-icons")
    const icon = createFavouriteIcon()
    itemIcon.appendChild(icon)
    item.appendChild(itemIcon)
  }

  return item
}

function getLocalItem(key: "curiusLinks" | "curiusLastFetch", value: any) {
  return localStorage.getItem(key) ?? value
}

export async function fetchCuriusLinks(refetch: boolean = false): Promise<CuriusResponse> {
  // user metadata
  const user = await fetch("https://raw.aarnphm.xyz/api/curius?query=user", fetchLinksHeaders)
    .then((res): Promise<CuriusResponse> => res.json())
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
    .then((data: CuriusResponse) => {
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

interface TrailInfo {
  trail: Trail
  links: Map<number, Link>
}

const trailMetadata: Map<string, TrailInfo> = new Map()
export async function fetchTrails() {
  const curiusTrails = await fetchCuriusLinks().then((res) => {
    const links = res.links ?? []
    links
      .filter((link) => link.trails.length > 0)
      .map((link) => {
        link.trails.map((trail) => {
          if (!trailMetadata.has(trail.trailName))
            trailMetadata.set(trail.trailName, { trail, links: new Map() })
          trailMetadata.get(trail.trailName)!.links.set(link.id, link)
        })
      })
    return trailMetadata
  })

  const trail = document.getElementById("trail-list") as HTMLUListElement | null
  const total = document.getElementsByClassName("curius-trail")[0] as HTMLDivElement | null
  if (!trail || !total) return

  const limits = parseInt(total.dataset.limits!) ?? 5
  const locale = total.dataset.locale! as ValidLocale

  removeAllChildren(trail)
  for (const [trail_name, { trail: Trail, links: linksMap }] of curiusTrails.entries()) {
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
  headers.innerHTML = `<span class="trail-title">sentier: ${trail_name}</span><span class="trail-description">${info.description!}</span>`

  const trailLink = joinSegments(curiusBase, "trail", info.slug)

  const links = document.createElement("ul")
  links.classList.add("trail-ul")
  links.append(
    ...trails.map((link) => {
      const el = createTitle({ Link: link, elementType: "li" })
      registerMouseHover(el, "focus")

      const openLink = () => window.open(trailLink, "_blank")

      el.addEventListener("click", openLink)
      window.addCleanup(() => el.removeEventListener("click", openLink))

      return el
    }),
  )

  const seeMore = document.createElement("div")
  seeMore.classList.add("see-more")
  seeMore.innerHTML = `<span><a href=${trailLink} target="_blank">${remaining > 0 ? i18n(locale).components.recentNotes.seeRemainingMore({ remaining }) : "Void de plus →"}</a></span>`

  container.append(headers, links, seeMore)

  return container
}
