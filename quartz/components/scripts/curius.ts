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

const iconMapping = {
  favourite: `<svg fill="currentColor" preserveAspectRatio="xMidYMid meet" height="1em" width="1em" viewBox="0 0 40 40" data-tip="Unfavorite" data-for="links" style="vertical-align: unset;"><g><path d="m5.2 18.8l6 5.5-1.7 7.7c-0.2 1 0.2 2 1 2.5 0.3 0.3 0.8 0.5 1.3 0.5 0.4 0 0.7 0 1-0.2 0 0 0.2 0 0.2-0.1l6.8-3.9 6.9 3.9s0.1 0 0.1 0.1c0.9 0.4 1.9 0.4 2.5-0.1 0.9-0.5 1.2-1.5 1-2.5l-1.6-7.7c0.6-0.5 1.6-1.5 2.6-2.5l3.2-2.8 0.2-0.2c0.6-0.7 0.8-1.7 0.5-2.5s-1-1.5-2-1.7h-0.2l-7.8-0.8-3.2-7.2s0-0.1-0.2-0.1c-0.1-1.2-1-1.7-1.8-1.7s-1.7 0.5-2.2 1.3c0 0 0 0.2-0.1 0.2l-3.2 7.2-7.8 0.8h-0.2c-0.8 0.2-1.7 0.8-2 1.7-0.2 1 0 2 0.7 2.6z"></path></g></svg>`,
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

export const createTitle = (
  Link: Link,
  elementType: "div" | "li" = "div",
): HTMLDivElement | HTMLLIElement => {
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

  const icons = document.createElement("div")
  icons.classList.add("curius-item-icons")
  if (Link.favorite) {
    const icon = document.createElement("span")
    icon.classList.add("curius-favourite")
    icon.innerHTML = getIconSvg("favourite")
    icons.appendChild(icon)
  }

  item.append(header, address, icons)
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

const trailMapping: Map<string, Link[]> = new Map()
const trailMetadata: Map<string, Trail> = new Map()
export async function fetchTrails() {
  const curiusTrails = await fetchCuriusLinks().then((res) => {
    const links = res.links ?? []
    links
      .filter((link) => link.trails.length > 0)
      .map((link) => {
        link.trails.map((trail) => {
          if (!trailMapping.has(trail.trailName)) trailMetadata.set(trail.trailName, trail)

          if (trailMapping.has(trail.trailName)) {
            trailMapping.set(trail.trailName, [...trailMapping.get(trail.trailName)!, link])
          } else {
            trailMapping.set(trail.trailName, [link])
          }
        })
      })
    return trailMapping
  })

  const trail = document.getElementById("trail-list") as HTMLUListElement | null
  const total = document.getElementsByClassName("curius-trail")[0] as HTMLDivElement | null
  if (!trail || !total) return

  const limits = parseInt(total.dataset.limits!) ?? 5
  const locale = total.dataset.locale! as ValidLocale

  removeAllChildren(trail)
  for (const [trail_name, links] of curiusTrails.entries()) {
    const info = trailMetadata.get(trail_name)
    if (info === undefined) continue
    const remaining = links.length - limits
    trail.appendChild(
      createTrailEl(
        trail_name,
        links.slice(0, limits),
        info,
        remaining > 0 ? remaining : 0,
        locale,
      ),
    )
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
  headers.style.display = "flex"
  headers.style.gap = "0.5rem"
  headers.innerHTML = `<span class="trail-title">Trail: ${trail_name}</span><span class="trail-description">${info.description!}</span>`

  const links = document.createElement("ul")
  links.classList.add("trail-ul")
  links.append(
    ...trails.map((link) => {
      const el = createTitle(link, "li")
      registerMouseHover(el, "focus")
      return el
    }),
  )

  const seeMore = document.createElement("div")
  seeMore.classList.add("see-more")
  seeMore.innerHTML = `<span><a href=${joinSegments(curiusBase, "trail", info.slug)} target="_blank">${i18n(locale).components.recentNotes.seeRemainingMore({ remaining })}</a></span>`

  container.append(headers, links, seeMore)

  return container
}
